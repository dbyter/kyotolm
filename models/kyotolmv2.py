"""
KyotoLM v2 — optimized architecture.

Changes vs v1:
- FlashAttention via F.scaled_dot_product_attention (O(T) memory, ~3-5x faster at seq_len=2048)
- Cached RoPE cos/sin buffers (computed once per seq_len, not every forward pass)
- Gradient checkpointing kept (required at batch_size=48, seq_len=2048 on 140GB H200)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 32000
    n_embedding_dim: int = 768
    n_head: int = 6
    n_layers: int = 12


class RMSNorm(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(config.n_embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.n_embedding_dim // config.n_head
        self.q_proj = nn.Linear(config.n_embedding_dim, config.n_head * self.d_head, bias=False)
        self.k_proj = nn.Linear(config.n_embedding_dim, config.n_head * self.d_head, bias=False)
        self.v_proj = nn.Linear(config.n_embedding_dim, config.n_head * self.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_head * self.d_head, config.n_embedding_dim, bias=False)

        self._rope_seq_len = 0
        self.register_buffer("_cos", torch.empty(0), persistent=False)
        self.register_buffer("_sin", torch.empty(0), persistent=False)

    def _get_rope(self, t: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if t != self._rope_seq_len:
            inv_freq = 10000.0 ** (-torch.arange(0, self.d_head, 2, device=device).float() / self.d_head)
            positions = torch.arange(t, device=device).float()
            angles = positions[:, None] * inv_freq[None, :]        # (T, d_head/2)
            self._cos = torch.cos(angles)[None, None, :, :]        # (1, 1, T, d_head/2)
            self._sin = torch.sin(angles)[None, None, :, :]
            self._rope_seq_len = t
        return self._cos, self._sin

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        return torch.stack([x_even * cos - x_odd * sin,
                            x_even * sin + x_odd * cos], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)

        cos, sin = self._get_rope(t, x.device)
        q = self._rotate(q, cos, sin)
        k = self._rotate(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_head * self.d_head)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embedding_dim, 4 * config.n_embedding_dim, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embedding_dim, config.n_embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class TransformerBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.ffn_norm  = RMSNorm(config)
        self.attn = MultiHeadAttention(config)
        self.ffn  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class KyotoLM(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embedding_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.rms_norm = RMSNorm(config)
        self.lm_head  = nn.Linear(config.n_embedding_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.transformer_blocks:
            if self.training:
                x = grad_ckpt(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.rms_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        stop_token: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(x)[:, -1, :]

            if repetition_penalty != 1.0:
                for token_id in x[0].tolist():
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / temperature
            if top_k is not None:
                top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
                probs = F.softmax(top_values, dim=-1)
                next_token = torch.gather(top_indices, -1, torch.multinomial(probs, 1))
            else:
                next_token = torch.multinomial(F.softmax(logits, -1), 1)

            x = torch.cat([x, next_token], dim=1)
            if stop_token is not None and next_token.item() == stop_token:
                break
        return x
