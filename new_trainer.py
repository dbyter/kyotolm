"""
Train our LLM using torch DDP

Handles data loading, model loading, and training
"""
import contextlib
import math
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from llm import KyotoLM, Config
from data_loader import make_dataset


parser = ArgumentParser()
parser.add_argument("--wandb_project", type=str, default="", help="W&B project name (empty = disabled)")
parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (auto-generated if empty)")
parser.add_argument("--no_wandb_artifacts", action="store_true", help="Disable uploading checkpoints as W&B artifacts")
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_heads", type=int, default=6, help="Number of heads in the model")
parser.add_argument("--n_embedding_dim", type=int, default=768, help="Number of embedding dimensions in the model")
parser.add_argument("--seq_length", type=int, default=2048, help="Max sequence length")
parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size (must match tokenizer.json)")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout parameter")
parser.add_argument("--batch_size", type=int, default=48, help="Per-GPU micro-batch size")
parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for optimization")
parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW optimizer decay")
parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs")
parser.add_argument("--grad_clip", type=float, default=0.1, help="Clip gradient at")
parser.add_argument("--log_every", type=int, default=10, help="Log step progress + loss every N steps")
parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint model every N steps")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/lm.pt", help="Path to save checkpoint to")
parser.add_argument("--n_shards", type=int, default=40, help="Number of HF parquet shards to stream")
parser.add_argument("--steps_per_epoch", type=int, default=0, help="Max optimizer steps per epoch (0 = stream until data exhausted)")
args = parser.parse_args()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# DDP setup — must happen before any CUDA calls
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_master = rank == 0

use_ddp = world_size > 1
if use_ddp:
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

if torch.cuda.is_available():
    device = torch.device("cuda", rank)
    if is_master:
        print(f"CUDA loaded, starting trainers: {world_size} torch instances running")
else:
    device = torch.device("cpu")

# Only validate tokenizer vocab on master to avoid redundant disk reads
if is_master:
    tokenizer_vocab = Tokenizer.from_file("tokenizer.json").get_vocab_size(with_added_tokens=True)
    if args.vocab_size != tokenizer_vocab:
        print(
            f"Warning: --vocab_size={args.vocab_size} does not match tokenizer "
            f"({tokenizer_vocab}); using tokenizer vocab size"
        )
        args.vocab_size = tokenizer_vocab

if use_ddp:
    # Broadcast the (possibly corrected) vocab_size from master to all ranks
    vocab_tensor = torch.tensor([args.vocab_size], dtype=torch.long, device=device)
    dist.broadcast(vocab_tensor, src=0)
    args.vocab_size = int(vocab_tensor.item())

if is_master:
    print(f"Building streaming dataset (rank {rank}/{world_size})")
ds = make_dataset(rank, world_size, seq_len=args.seq_length, n_shards=args.n_shards)
loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())

# Streaming dataset has no known length upfront; used for LR schedule
total_steps = args.max_epochs * (args.steps_per_epoch if args.steps_per_epoch > 0 else 10_000)
if is_master:
    print(f"Streaming {args.n_shards} shards, seq_len={args.seq_length}, batch_size={args.batch_size}, device={device}")

config = Config(
    vocab_size=args.vocab_size,
    n_embedding_dim=args.n_embedding_dim,
    n_head=args.n_heads,
    n_layers=args.n_layers,
)
model = KyotoLM(config).to(device)

optim = AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.95),
)


def get_lr(current_step: int) -> float:
    """Linear warmup then cosine decay to 10% of peak lr."""
    warmup_steps = max(1, int(0.02 * total_steps))
    if current_step < warmup_steps:
        return current_step / warmup_steps
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=get_lr)

start_epoch = 0
step = 0
checkpoint_path = Path(args.checkpoint_path)
if checkpoint_path.is_file():
    d = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(d["model_state_dict"])
    optim.load_state_dict(d["optimizer_state_dict"])
    if "scheduler_state_dict" in d:
        scheduler.load_state_dict(d["scheduler_state_dict"])
    start_epoch = d.get("epoch", 0)
    step = d.get("step", 0)
    if is_master:
        print(f"resumed from {checkpoint_path} (epoch {start_epoch + 1}, step {step})")

# Init wandb on master only, after checkpoint load so `step` reflects any resume
use_wandb = is_master and bool(args.wandb_project)
if use_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config=vars(args),
        resume="allow",
    )

# Wrap model in DDP after loading checkpoint so all ranks start with identical weights
if use_ddp:
    model = DDP(model, device_ids=[rank])

raw_model = model.module if use_ddp else model


def save_ckpt(tag: str) -> None:
    """Save checkpoint — only called on rank 0."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_cpu = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
    payload = {
        "model_state_dict": state_cpu,
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "vocab_size": args.vocab_size,
        "seq_length": args.seq_length,
        "config": vars(args),
    }
    torch.save(payload, path)
    print(f"saved checkpoint to {path.resolve()} ({tag})")
    if use_wandb and not args.no_wandb_artifacts:
        artifact = wandb.Artifact(
            name=f"checkpoint-{tag.replace(' ', '-')}",
            type="model",
            metadata={"step": step, "epoch": epoch},
        )
        artifact.add_file(str(path.resolve()))
        wandb.log_artifact(artifact)


use_amp = torch.cuda.is_available()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)

model.train()
train_t0 = time.perf_counter()
for epoch in range(start_epoch, args.max_epochs):
    epoch_loss = 0.0
    n_batches = 0
    accum_loss = 0.0
    epoch_step = 0

    for batch_idx, (x, y) in enumerate(loader):
        if args.steps_per_epoch > 0 and epoch_step >= args.steps_per_epoch:
            break

        x = x.to(device)
        y = y.to(device)

        is_accum_step = (batch_idx + 1) % args.grad_accum_steps != 0

        # Skip gradient sync on accumulation steps — saves unnecessary all-reduce communication
        ddp_sync_ctx = model.no_sync() if (use_ddp and is_accum_step) else contextlib.nullcontext()

        with ddp_sync_ctx, autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss = loss / args.grad_accum_steps

        loss.backward()
        accum_loss += loss.item()

        if not is_accum_step:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)

            epoch_loss += accum_loss
            n_batches += 1
            step += 1
            epoch_step += 1
            accum_loss = 0.0

            if is_master and step % args.log_every == 0:
                wall = time.perf_counter() - train_t0
                current_lr = scheduler.get_last_lr()[0]
                mean_loss = epoch_loss / n_batches
                print(
                    f"epoch {epoch + 1} step {step} loss {mean_loss:.4f} "
                    f"lr {current_lr:.2e} wall {wall:.1f}s"
                )
                if use_wandb:
                    wandb.log({"train/loss": mean_loss, "train/lr": current_lr, "train/wall": wall}, step=step)
            if is_master and args.save_every > 0 and step % args.save_every == 0:
                save_ckpt(f"step {step}")

    if is_master:
        mean = epoch_loss / max(n_batches, 1)
        wall = time.perf_counter() - train_t0
        print(f"epoch {epoch + 1} mean loss {mean:.4f} (device={device}) wall {wall:.1f}s")
        if use_wandb:
            wandb.log({"epoch/mean_loss": mean, "epoch/wall": wall}, step=step)

if is_master and (args.save_every <= 0 or step % args.save_every != 0):
    save_ckpt("final")

if use_wandb:
    wandb.finish()

if use_ddp:
    dist.destroy_process_group()
