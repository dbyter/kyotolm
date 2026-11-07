"""Generate text with a trained checkpoint.

Usage:
  python generate.py [checkpoint_path | wandb artifact like lm:step-20000] [--chat]

  --chat   Use SFT chat format (User/Assistant turns). Pass for SFT checkpoints.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from models import load_model
from tokenizers import Tokenizer

parser = ArgumentParser()
parser.add_argument("checkpoint", nargs="?", help="Path or W&B artifact, e.g. lm:step-20000 (default: latest in checkpoints/)")
parser.add_argument("--chat", action="store_true", help="Use chat format for SFT models")
args = parser.parse_args()

if args.checkpoint:
    ckpt_path = Path(args.checkpoint)
else:
    pts = list(Path("checkpoints").glob("*.pt"))
    ckpt_path = max(pts, key=lambda p: p.stat().st_mtime) if pts else Path("checkpoints/lm.pt")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(ckpt_path, dev)

tokenizer = Tokenizer.from_file("tokenizer.json")
bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")

history = ""  # accumulated conversation for multi-turn chat

while True:
    user_input = input("You: " if args.chat else "Enter a prompt: ").rstrip()

    if args.chat:
        history += f"User: {user_input}\n\nAssistant:"
        ids = [bos_id] + tokenizer.encode(history).ids
    else:
        ids = tokenizer.encode(user_input).ids

    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out = model.generate(x, max_new_tokens=200, stop_token=eos_id, repetition_penalty=1.3)
    generated_ids = out[0].tolist()[len(ids):]
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]

    response = tokenizer.decode(generated_ids).strip()

    if args.chat:
        print(f"Assistant: {response}\n")
        history += f" {response}\n\n"
    else:
        print(user_input + tokenizer.decode(generated_ids))
