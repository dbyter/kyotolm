"""
Train our LLM using torch DDP

Handles data loading, model loading, and training
"""
from inspect import Arguments
from numpy import argsort
import torch
import time 
import os 
from argparse import ArgumentParser
from tokenizers import Tokenizer
from llm import KyotoLM, Config
from data_loader import load_data
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from llm import KyotoLM, Config
from pathlib import Path
import dataclasses
from dataclasses import dataclass



parser = ArgumentParser()
parser.add_argument("--n_layers", type=int, default=12, help="Number of layers in the model")
parser.add_argument("--n_heads", type=int, default=6, help="Number of heads in the model")
parser.add_argument("--n_embedding_dim", type=int, default=768, help="Number of embedding dimensions in the model")
parser.add_argument("--seq_length", type=int, default=2048, help="Max sequence length")
parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size (must match tokenizer.json)")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout parameter")
parser.add_argument("--batch_size", type=int, default=24, help="Vocabulary size")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for opimization")
parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW optimizer decay")
parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs")
parser.add_argument("--grad_clip", type=float, default=0.1, help="Clip gradient at")
parser.add_argument("--log_every", type=int, default=10, help="Log step progress + loss every N steps")
parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint model every N steps")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/lm.pt", help="Path to save checkpoint to")
args = parser.parse_args()

tokenizer_vocab = Tokenizer.from_file("tokenizer.json").get_vocab_size(with_added_tokens=True)
if args.vocab_size != tokenizer_vocab:
    print(
        f"Warning: --vocab_size={args.vocab_size} does not match tokenizer "
        f"({tokenizer_vocab}); using tokenizer vocab size"
    )
    args.vocab_size = tokenizer_vocab

# DDP setup
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_master = rank == 0


if torch.cuda.is_available():
    device = torch.device("cuda", rank)
    # Load CUDA device 
    device = torch.device("cuda", rank)
    if is_master: print(f"CUDA loaded, starting trainers: {world_size} torch instances running")
else:
    if torch.backends.mps.is_available():
        print (f"Loading MPS")
        device = torch.device("mps")
    device = torch.device("cpu")


# Load data; tokenize first and then load the dataset to device 
print(f"Beginning data load")
input_sequences, output_sequences = load_data(rank, world_size, seq_len=args.seq_length)
inp = torch.tensor(list(input_sequences), dtype=torch.long)
tgt = torch.tensor(list(output_sequences), dtype=torch.long)
max_id = max(inp.max().item(), tgt.max().item())
min_id = min(inp.min().item(), tgt.min().item())
if min_id < 0 or max_id >= args.vocab_size:
    raise ValueError(
        f"Token ids out of range [{min_id}, {max_id}] for vocab_size={args.vocab_size}"
    )

ds = TensorDataset(inp, tgt) 
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

n_seq = int(inp.shape[0])
steps_per_epoch = len(loader)
total_steps = steps_per_epoch * args.max_epochs
print(
    f"Training data: {n_seq} input rows, sequence length {args.seq_length}, "
    f"tensor shape {tuple(inp.shape)}, batch_size={args.batch_size}"
)
print(
    f"Expected optimizer steps: {steps_per_epoch} per epoch × {args.max_epochs} epoch(s) "
    f"= {total_steps} total (device={device})"
)

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

def save_ckpt(tag: str) -> None:
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "model_state_dict": state_cpu,
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
        "step": step,
        "vocab_size": args.vocab_size,
        "seq_length": args.seq_length,
        "config": dataclasses.asdict(args),
    }
    torch.save(payload, path)
    print(f"saved checkpoint to {path.resolve()} ({tag})")

start_epoch = 0
step = 0
checkpoint_path = Path(args.checkpoint_path)
if checkpoint_path is not None and Path(checkpoint_path).is_file():
    d = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(d["model_state_dict"])
    optim.load_state_dict(d["optimizer_state_dict"])
    start_epoch = d.get("epoch", 0)
    step = d.get("step", 0)
    print(f"resumed from {checkpoint_path} (epoch {start_epoch + 1}, step {step})")

model.train()
train_t0 = time.perf_counter()
for epoch in range(start_epoch, args.max_epochs):
    epoch_loss = 0.0
    n_batches = 0
    batches_done = step % steps_per_epoch
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx < batches_done:
            continue
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        epoch_loss += loss.item()
        n_batches += 1
        step += 1
        if step % args.log_every == 0:
            wall = time.perf_counter() - train_t0
            print(
                f"epoch {epoch + 1} step {step} loss {loss.item():.4f} "
                f"wall {wall:.1f}s"
            )
        if checkpoint_path is not None and args.save_every > 0 and step % args.save_every == 0:
            save_ckpt(f"step {step}")

    mean = epoch_loss / max(n_batches, 1)
    wall = time.perf_counter() - train_t0
    print(
        f"epoch {epoch + 1} mean loss {mean:.4f} (device={device}) wall {wall:.1f}s"
    )

if checkpoint_path is not None and (args.save_every <= 0 or step % args.save_every != 0):
    save_ckpt("final")



