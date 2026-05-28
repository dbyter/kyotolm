"""
Ahmed's LLM model implementation.

Embedding
Transformer Blocks
    - Grouped Query Attention
    - RMSNorm
    - MLP
    - RotaryEmbedding
    - SwiGLU
Output Layer
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
    n_head: int = 6 # number of query heads
    n_layers: int = 12

# Start with MultiHeadAttention
class MultiHeadAttention (nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.n_embedding_dim // config.n_head
        self.q_proj = nn.Linear(config.n_embedding_dim, config.n_head * self.d_head, bias = False) # (B, T, D) -> (B, T, H * D_head)
        self.k_proj = nn.Linear(config.n_embedding_dim, config.n_head * self.d_head, bias = False) # (B, T, D) -> (B, T, H * D_head)
        self.v_proj = nn.Linear(config.n_embedding_dim, config.n_head * self.d_head, bias = False) # (B, T, D) -> (B, T, H * D_head)
        self.o_proj = nn.Linear(config.n_head * self.d_head, config.n_embedding_dim) # (B, T, H * D_head) -> (B, T, D)

    @staticmethod
    def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, H, T, D_head)
        returns: rotated q, k with same shape
        """
        b, h, t, d_head = q.shape

        assert d_head % 2 == 0, "RoPE requires even d_head"

        device = q.device

        # Frequencies for each pair of dimensions
        inv_freq = 10000.0 ** (-torch.arange(0, d_head, 2, device=device).float() / d_head)
        # shape: (D_head / 2,)

        # Token positions
        positions = torch.arange(t, device=device).float()
        # shape: (T,)

        # Every position gets every frequency
        angles = positions[:, None] * inv_freq[None, :]
        # shape: (T, D_head / 2)

        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]
        # shape: (1, 1, T, D_head / 2)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]

            x_rot_even = x_even * cos - x_odd * sin
            x_rot_odd = x_even * sin + x_odd * cos

            # interleave even/odd dimensions back together
            x_rot = torch.empty_like(x)
            x_rot[..., 0::2] = x_rot_even
            x_rot[..., 1::2] = x_rot_odd

            return x_rot

        return rotate(q), rotate(k)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, D_head)
        k = self.k_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, D_head)
        v = self.v_proj(x).view(b, t, self.n_head, self.d_head).transpose(1, 2)  # (B, H, T, D_head)
        scaling_factor = self.d_head**-0.5
        q_rope, k_rope = self.apply_rope(q, k)
        qk = q_rope@k_rope.transpose(-2, -1) * scaling_factor # (B, H, T, d_head) * (B, H, d_head, T) -> (B, H, T, T)
        mask = torch.tril(torch.ones(t, t, device=x.device)).bool()
        qk = qk.masked_fill(~mask, float("-inf"))
        qk_soft = F.softmax(qk, dim=-1) # (B, H, T, T)
        attn = qk_soft@v # (B, H, T, T) @ (B, H, T, D_head) -> (B, H, T, D_head)
        out = self.o_proj(attn.transpose(1, 2).contiguous().view(b, t, self.n_head * self.d_head)) # (B, H, T, D_head) -> (B, T, H * D_head) -> (B, T, D_model)
        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embedding_dim, 4 * config.n_embedding_dim, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embedding_dim, config.n_embedding_dim, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)
        self.attn = MultiHeadAttention(config)
        self.ffn = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class RMSNorm(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(config.n_embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight

class LMHead(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.lm_head = nn.Linear(config.n_embedding_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)

class KyotoLM(nn.Module):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embedding_dim)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.rms_norm = RMSNorm(config)
        self.lm_head = LMHead(config)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X -> (B, T, D)
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
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            logits = self.forward(x)

            # only use logits for the last token
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                top_values, top_indices = torch.topk(logits, k=top_k, dim=-1)
                probs = F.softmax(top_values, dim=-1)

                # sample an index within the top-k list
                sampled_idx = torch.multinomial(probs, num_samples=1)

                # convert back to actual token id
                next_token = torch.gather(top_indices, dim=-1, index=sampled_idx)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat([x, next_token], dim=1)

        return x
