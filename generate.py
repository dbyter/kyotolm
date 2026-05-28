"""Generate text with a trained checkpoint (optional path as argv[1])."""

import sys
from pathlib import Path

import torch
from llm import KyotoLM, Config
from tokenizers import Tokenizer
from trainer import get_device

if len(sys.argv) > 1:
    ckpt_path = Path(sys.argv[1])
else:
    pts = list(Path("checkpoints").glob("*.pt"))
    ckpt_path = max(pts, key=lambda p: p.stat().st_mtime) if pts else Path("checkpoints/lm.pt")

d = torch.load(ckpt_path, map_location="cpu")
tc = d["config"]
model = KyotoLM(
    Config(
        vocab_size=d["vocab_size"],
        n_embedding_dim=tc.get("n_embedding_dim", tc.get("embedding_dim", 768)),
        n_head=tc.get("n_heads", 6),
        n_layers=tc.get("n_layers", 12),
    )
)
model.load_state_dict(d["model_state_dict"])
dev = get_device()
model.to(dev)
model.eval()

tokenizer = Tokenizer.from_file("tokenizer.json")
eos_id = tokenizer.encode("<eos>").ids[0]

while True:
    prompt = input("Enter a prompt: ")
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out = model.generate(x, max_new_tokens=200, stop_token=eos_id, repetition_penalty=1.3)
    out_ids = out[0].tolist()
    if eos_id in out_ids[len(ids):]:
        out_ids = out_ids[:len(ids) + out_ids[len(ids):].index(eos_id)]
    # Decode the full sequence to avoid broken byte boundaries, then strip the prompt
    full_text = tokenizer.decode(out_ids)
    prompt_decoded = tokenizer.decode(ids)
    print(full_text[len(prompt_decoded):])
