"""
Train ``Model`` on (input, next-token) sequences with AdamW on Apple MPS when available.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from llm import KyotoLM, Config


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    embedding_dim: int = 512
    hidden_dim: int = 2048
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int | None = None
    dropout: float = 0.1
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 1
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 100


def train_lm(
    input_sequences: Sequence[Sequence[int]],
    output_sequences: Sequence[Sequence[int]],
    *,
    vocab_size: int,
    seq_len: int,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
    checkpoint_path: Path | str | None = None,
) -> KyotoLM:
    """
    Train a causal LM where each row of ``output_sequences`` is the next-token targets
    for the same row of ``input_sequences`` (same layout as ``main.py``).

    Args:
        input_sequences: List of length-``seq_len`` token id lists.
        output_sequences: List of length-``seq_len`` target id lists (shifted by one).
        vocab_size: Vocabulary size (e.g. ``tokenizer.get_vocab_size()``).
        seq_len: Sequence length (must match each row).
        config: Hyperparameters; defaults are conservative for long contexts on MPS.
        device: If ``None``, uses MPS when available, then CUDA, else CPU.
        checkpoint_path: If set, saves ``model_state_dict`` (CPU tensors) plus
            ``vocab_size``, ``seq_len``, and ``config`` for reloading. When
            ``cfg.save_every > 0``, also writes the same file every ``save_every``
            optimizer steps; always writes once at the end if the last step is not
            already a checkpoint step (or if ``save_every <= 0``).
    """
    if len(input_sequences) != len(output_sequences):
        raise ValueError("input_sequences and output_sequences must have the same length")
    if not input_sequences:
        raise ValueError("need at least one training sequence")

    cfg = config or TrainConfig()
    dev = device or get_device()

    inp = torch.tensor(list(input_sequences), dtype=torch.long)
    tgt = torch.tensor(list(output_sequences), dtype=torch.long)
    if inp.shape[1] != seq_len or tgt.shape[1] != seq_len:
        raise ValueError(f"expected sequence length {seq_len}, got {inp.shape[1]} / {tgt.shape[1]}")

    ds = TensorDataset(inp, tgt)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    n_seq = int(inp.shape[0])
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.max_epochs
    print(
        f"Training data: {n_seq} input rows, sequence length {seq_len}, "
        f"tensor shape {tuple(inp.shape)}, batch_size={cfg.batch_size}"
    )
    print(
        f"Expected optimizer steps: {steps_per_epoch} per epoch × {cfg.max_epochs} epoch(s) "
        f"= {total_steps} total (device={dev})"
    )

    config = Config(
        vocab_size=vocab_size,
        n_embedding_dim=cfg.embedding_dim,
        n_head=cfg.n_heads,
        n_layers=cfg.n_layers,
    )
    model = KyotoLM(config).to(dev)

    optim = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    def save_ckpt(tag: str) -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload = {
            "model_state_dict": state_cpu,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "config": dataclasses.asdict(cfg),
        }
        torch.save(payload, path)
        print(f"saved checkpoint to {path.resolve()} ({tag})")

    model.train()
    step = 0
    for epoch in range(cfg.max_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            if step % cfg.log_every == 0:
                print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f}")
            if checkpoint_path is not None and cfg.save_every > 0 and step % cfg.save_every == 0:
                save_ckpt(f"step {step}")

        mean = epoch_loss / max(n_batches, 1)
        print(f"epoch {epoch + 1} mean loss {mean:.4f} (device={dev})")

    if checkpoint_path is not None and (cfg.save_every <= 0 or step % cfg.save_every != 0):
        save_ckpt("final")

    return model


if __name__ == "__main__":
    import main as main_mod

    vocab_size = main_mod.tokenizer.get_vocab_size(with_added_tokens=True)
    print(
        f"Loaded {len(main_mod.input_sequences)} chunks, "
        f"seq_len={main_mod.SEQ_LEN}, vocab={vocab_size}"
    )

    train_lm(
        main_mod.input_sequences,
        main_mod.output_sequences,
        vocab_size=vocab_size,
        seq_len=main_mod.SEQ_LEN,
        checkpoint_path=Path("checkpoints/lm.pt"),
    )
