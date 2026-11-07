"""
SFT trainer for KyotoLM.

Fine-tunes a pre-trained checkpoint on smol-smoltalk using DDP.
Loss is computed only on assistant tokens (prompt tokens are masked to -100).
"""
import contextlib
import itertools
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
from torch.utils.data import DataLoader

from models import MODELS, resolve_checkpoint
from data_loader import make_sft_dataset
from tokenizers import Tokenizer
from training.optim import build_optimizers
from training.wandb_utils import stage, upload_checkpoint


parser = ArgumentParser()
parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw", help="Match the pretraining optimizer")
parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Local path or wandb artifact name (e.g. lm:latest)")
parser.add_argument("--sft_checkpoint_path", type=str, default="checkpoints/sft.pt", help="Path to save SFT checkpoints")
parser.add_argument("--seq_length", type=int, default=2048, help="Max sequence length")
parser.add_argument("--fp8", action="store_true", help="Use fp8 training for ~2x throughput (requires H100/H200 and torch>=2.11)")
parser.add_argument("--batch_size", type=int, default=16, help="Per-GPU micro-batch size")
parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate")
parser.add_argument("--adamw_lr", type=float, default=None, help="Muon only: lr for the embedding and norms (default: --learning_rate)")
parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs over the SFT dataset")
parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
parser.add_argument("--log_every", type=int, default=10, help="Log every N optimizer steps")
parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N optimizer steps (0 = disable)")
parser.add_argument("--eval_every", type=int, default=250, help="Compute val loss every N optimizer steps (0 = only at the end)")
parser.add_argument("--eval_batches", type=int, default=20, help="Val batches per rank per eval")
parser.add_argument("--steps_per_epoch", type=int, default=0, help="Max optimizer steps per epoch (0 = full dataset)")
parser.add_argument("--wandb_project", type=str, default="", help="W&B project name (empty = disabled)")
parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (auto-generated if empty)")
parser.add_argument("--no_wandb_artifacts", action="store_true", help="Disable uploading the final checkpoint as W&B artifact 'sft'")
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

# Load pretrained checkpoint — local path or wandb artifact (e.g. lm:step-20000). Master downloads
# once and shares the local path, since only master has a wandb run.
ckpt_ref = [str(resolve_checkpoint(args.pretrained_checkpoint)) if is_master else None]
if use_ddp:
    dist.broadcast_object_list(ckpt_ref, src=0)
ckpt_path = Path(ckpt_ref[0])

if is_master:
    print(f"Loading pretrained checkpoint from {ckpt_path}")

d = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Reconstruct model from saved config so arch args don't need to be re-specified
saved_cfg = d.get("config", {})
model_version = saved_cfg.get("model", "v2")
arch = MODELS[model_version]
config = arch.Config(
    vocab_size=d.get("vocab_size", saved_cfg.get("vocab_size", 32000)),
    n_embedding_dim=saved_cfg.get("n_embedding_dim", 768),
    n_head=saved_cfg.get("n_heads", 6),
    n_layers=saved_cfg.get("n_layers", 12),
)
model = arch.KyotoLM(config).to(device)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in d["model_state_dict"].items()}
model.load_state_dict(state_dict)
if is_master:
    print(f"Loaded model: {config}")

if args.fp8 and torch.cuda.is_available():
    from torchao.float8 import convert_to_float8_training
    convert_to_float8_training(model)
    if is_master:
        print("fp8 training enabled")

model = torch.compile(model, dynamic=False)
if is_master:
    print("torch.compile enabled")

# Dataset
if is_master:
    print("Building SFT dataset (smol-smoltalk, streaming)...")
ds = make_sft_dataset(rank, world_size, seq_len=args.seq_length)
loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, prefetch_factor=8, pin_memory=torch.cuda.is_available())
# Fixed val set, read once so each eval is just forward passes. num_workers=1 matters: a partially
# read HF stream in the main process hangs interpreter exit.
val_ds = make_sft_dataset(rank, world_size, seq_len=args.seq_length, split="test")
val_batches = list(itertools.islice(DataLoader(val_ds, batch_size=args.batch_size, num_workers=1), args.eval_batches))

if args.steps_per_epoch == 0:
    # Interleaving runs until smoltalk (460k examples, p=0.7) is exhausted.
    n_examples = int(460_341 / 0.7)
    # Packing makes the sequence count data-dependent, so estimate it by counting <bos> (one per
    # conversation) in ~800 packed sequences. The sample comes from the unsharded stream so every
    # rank computes the same number, and the 3% margin stops all ranks before any one runs dry
    # (a rank exiting early would hang DDP).
    bos_id = Tokenizer.from_file("tokenizer.json").token_to_id("<bos>")
    sample = itertools.islice(DataLoader(make_sft_dataset(0, 1, seq_len=args.seq_length), batch_size=64, num_workers=1), 13)
    x_sample = torch.cat([x for x, _ in sample])
    examples_per_seq = (x_sample == bos_id).sum().item() / len(x_sample)
    args.steps_per_epoch = int(0.97 * n_examples / examples_per_seq / (args.batch_size * world_size * args.grad_accum_steps))
    if is_master:
        print(f"Packing ~{examples_per_seq:.1f} conversations per sequence")
total_steps = args.max_epochs * args.steps_per_epoch
if is_master:
    print(f"SFT total steps: {total_steps}")

optimizers = build_optimizers(model._orig_mod, args.optimizer, args.learning_rate, args.weight_decay, args.adamw_lr)


def get_lr(current_step: int) -> float:
    warmup_steps = max(1, int(0.03 * total_steps))
    if current_step < warmup_steps:
        return current_step / warmup_steps
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lr_lambda=get_lr) for o in optimizers]

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
        "optimizer_state_dict": [o.state_dict() for o in optimizers],
        "scheduler_state_dict": [s.state_dict() for s in schedulers],
        "epoch": epoch,
        "step": step,
        "vocab_size": config.vocab_size,
        "seq_length": args.seq_length,
        "config": {
            "model": model_version,
            "n_embedding_dim": config.n_embedding_dim,
            "n_heads": config.n_head,
            "n_layers": config.n_layers,
        },
    }
    torch.save(payload, path)
    print(f"saved SFT checkpoint to {path.resolve()} ({tag})")


use_amp = torch.cuda.is_available()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def val_loss() -> float:
    """Token-weighted loss on the first eval_batches val batches, reduced across ranks. Call on every rank."""
    totals = torch.zeros(2, device=device)  # [loss sum, assistant token count]
    for x, y in val_batches:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            # raw_model bypasses DDP; no dropout, so no eval()/train() toggle (that would force a recompile)
            logits = raw_model(x)
            totals[0] += F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1), ignore_index=-100, reduction="sum")
        totals[1] += (y != -100).sum()
    if use_ddp:
        dist.all_reduce(totals)
    return (totals[0] / totals[1]).item()


def log_val_loss() -> None:
    loss = val_loss()
    if is_master:
        print(f"step {step} val_loss {loss:.4f}")
        if use_wandb:
            wandb.log({"val/loss": loss}, step=step)


model.train()
train_t0 = time.perf_counter()
window_loss = 0.0

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
        accum_loss += loss.detach()

        if not is_accum_step:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            for o, s in zip(optimizers, schedulers):
                o.step()
                s.step()
            model.zero_grad(set_to_none=True)

            epoch_loss += accum_loss
            window_loss += accum_loss
            n_batches += 1
            step += 1
            epoch_step += 1
            accum_loss = 0.0

            if step % args.log_every == 0:
                if is_master:
                    wall = time.perf_counter() - train_t0
                    current_lr = schedulers[0].get_last_lr()[0]
                    mean_loss = window_loss.item() / args.log_every  # mean over this window, not the epoch
                    print(f"epoch {epoch + 1} step {step} loss {mean_loss:.4f} lr {current_lr:.2e} wall {wall:.1f}s")
                    if use_wandb:
                        wandb.log({"train/loss": mean_loss, "train/lr": current_lr, "train/wall": wall}, step=step)
                window_loss = 0.0

            if args.eval_every > 0 and step % args.eval_every == 0:
                log_val_loss()

            if is_master and args.save_every > 0 and step % args.save_every == 0:
                save_ckpt(f"step {step}")

    if is_master:
        mean = float(epoch_loss) / max(n_batches, 1)
        wall = time.perf_counter() - train_t0
        print(f"epoch {epoch + 1} mean loss {mean:.4f} wall {wall:.1f}s")
        if use_wandb:
            wandb.log({"epoch/mean_loss": mean, "epoch/wall": wall}, step=step)

if args.eval_every <= 0 or step % args.eval_every != 0:
    log_val_loss()

if is_master and (args.save_every <= 0 or step % args.save_every != 0):
    save_ckpt("final")
if is_master and use_wandb and not args.no_wandb_artifacts:
    upload_checkpoint(stage(checkpoint_path, step), "sft", step, keep_history=True, final=True)

if use_wandb:
    wandb.finish()

if use_ddp:
    dist.destroy_process_group()
