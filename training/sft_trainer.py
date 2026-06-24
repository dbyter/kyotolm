"""
SFT trainer for KyotoLM.

Fine-tunes a pre-trained checkpoint on smol-smoltalk using DDP.
Loss is computed only on assistant tokens (prompt tokens are masked to -100).
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

from models.kyotov1 import KyotoLM, Config
from data_loader import make_sft_dataset


parser = ArgumentParser()
parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Local path or wandb artifact name (e.g. checkpoint-step-1000:latest)")
parser.add_argument("--sft_checkpoint_path", type=str, default="checkpoints/sft.pt", help="Path to save SFT checkpoints")
parser.add_argument("--seq_length", type=int, default=2048, help="Max sequence length")
parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU micro-batch size")
parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate")
parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs over the SFT dataset")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
parser.add_argument("--log_every", type=int, default=10, help="Log every N optimizer steps")
parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N optimizer steps (0 = disable)")
parser.add_argument("--steps_per_epoch", type=int, default=0, help="Max optimizer steps per epoch (0 = full dataset)")
parser.add_argument("--wandb_project", type=str, default="", help="W&B project name (empty = disabled)")
parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (auto-generated if empty)")
parser.add_argument("--no_wandb_artifacts", action="store_true", help="Disable uploading checkpoints as W&B artifacts")
args = parser.parse_args()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# DDP setup
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_master = rank == 0

use_ddp = world_size > 1
if use_ddp:
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
if is_master:
    print(f"{'CUDA' if torch.cuda.is_available() else 'CPU'} — {world_size} ranks")

# Init wandb early (before checkpoint load) so we can download artifacts if needed
use_wandb = is_master and bool(args.wandb_project)
if use_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config=vars(args),
        resume="allow",
    )

# Load pretrained checkpoint — local path or wandb artifact
ckpt_path = Path(args.pretrained_checkpoint)
if not ckpt_path.is_file():
    if not use_wandb:
        raise FileNotFoundError(
            f"Checkpoint '{args.pretrained_checkpoint}' not found locally and --wandb_project is not set for artifact download."
        )
    if is_master:
        print(f"Downloading wandb artifact: {args.pretrained_checkpoint}")
    artifact = wandb.use_artifact(args.pretrained_checkpoint, type="model")
    artifact_dir = artifact.download()
    # artifact contains a single .pt file
    pt_files = list(Path(artifact_dir).glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in downloaded artifact at {artifact_dir}")
    ckpt_path = pt_files[0]

if is_master:
    print(f"Loading pretrained checkpoint from {ckpt_path}")

d = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Reconstruct model from saved config so arch args don't need to be re-specified
saved_cfg = d.get("config", {})
config = Config(
    vocab_size=d.get("vocab_size", saved_cfg.get("vocab_size", 32000)),
    n_embedding_dim=saved_cfg.get("n_embedding_dim", 768),
    n_head=saved_cfg.get("n_heads", 6),
    n_layers=saved_cfg.get("n_layers", 12),
)
model = KyotoLM(config).to(device)
model.load_state_dict(d["model_state_dict"])
if is_master:
    print(f"Loaded model: {config}")

# Dataset
if is_master:
    print("Building SFT dataset (smol-smoltalk, streaming)...")
ds = make_sft_dataset(rank, world_size, seq_len=args.seq_length)
loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, pin_memory=torch.cuda.is_available())

total_steps = args.max_epochs * (args.steps_per_epoch if args.steps_per_epoch > 0 else 28_000)

optim = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))


def get_lr(current_step: int) -> float:
    warmup_steps = max(1, int(0.03 * total_steps))
    if current_step < warmup_steps:
        return current_step / warmup_steps
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=get_lr)

if use_ddp:
    model = DDP(model, device_ids=[rank])

raw_model = model.module if use_ddp else model
checkpoint_path = Path(args.sft_checkpoint_path)
step = 0


def save_ckpt(tag: str) -> None:
    path = checkpoint_path
    path.parent.mkdir(parents=True, exist_ok=True)
    state_cpu = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
    payload = {
        "model_state_dict": state_cpu,
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "vocab_size": config.vocab_size,
        "seq_length": args.seq_length,
        "config": vars(args),
    }
    torch.save(payload, path)
    print(f"saved SFT checkpoint to {path.resolve()} ({tag})")
    if use_wandb and not args.no_wandb_artifacts:
        artifact = wandb.Artifact(
            name=f"sft-checkpoint-{tag.replace(' ', '-')}",
            type="model",
            metadata={"step": step, "epoch": epoch},
        )
        artifact.add_file(str(path.resolve()))
        wandb.log_artifact(artifact)


use_amp = torch.cuda.is_available()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)

model.train()
train_t0 = time.perf_counter()

for epoch in range(args.max_epochs):
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
        ddp_sync_ctx = model.no_sync() if (use_ddp and is_accum_step) else contextlib.nullcontext()

        with ddp_sync_ctx, autocast_ctx:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-100)
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
                print(f"epoch {epoch + 1} step {step} loss {mean_loss:.4f} lr {current_lr:.2e} wall {wall:.1f}s")
                if use_wandb:
                    wandb.log({"train/loss": mean_loss, "train/lr": current_lr, "train/wall": wall}, step=step)

            if is_master and args.save_every > 0 and step % args.save_every == 0:
                save_ckpt(f"step {step}")

    if is_master:
        mean = epoch_loss / max(n_batches, 1)
        wall = time.perf_counter() - train_t0
        print(f"epoch {epoch + 1} mean loss {mean:.4f} wall {wall:.1f}s")
        if use_wandb:
            wandb.log({"epoch/mean_loss": mean, "epoch/wall": wall}, step=step)

if is_master and (args.save_every <= 0 or step % args.save_every != 0):
    save_ckpt("final")

if use_wandb:
    wandb.finish()

if use_ddp:
    dist.destroy_process_group()
