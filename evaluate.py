"""
Benchmark an SFT checkpoint on MMLU and GSM8K, using the same chat format as SFT.

MMLU:  picks the choice whose "The answer is X." completion has the highest log-prob.
GSM8K: greedy generation, exact match on the number after "####".

Usage:
  python evaluate.py checkpoints/sft.pt [--mmlu_n 1000] [--gsm8k_n 200]
"""
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer

from data_loader import _format_mmlu
from models import load_model

parser = ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--mmlu_n", type=int, default=1000, help="MMLU test questions to score (0 = skip)")
parser.add_argument("--gsm8k_n", type=int, default=200, help="GSM8K test problems to solve (0 = skip)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
model = load_model(args.checkpoint, device)
tokenizer = Tokenizer.from_file("tokenizer.json")
bos_id = tokenizer.token_to_id("<bos>")
eos_id = tokenizer.token_to_id("<eos>")


def prompt_ids(question: str) -> list[int]:
    return [bos_id] + tokenizer.encode(f"User: {question}\n\nAssistant: ").ids


@torch.no_grad()
def completion_logprob(prompt: list[int], completion: list[int]) -> float:
    ids = torch.tensor([prompt + completion], device=device)
    with autocast_ctx:
        logits = model(ids[:, :-1])
    logprobs = F.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return logprobs[0, len(prompt) - 1 :].sum().item()  # position i predicts token i + 1


def mmlu_accuracy(n: int) -> float:
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(n))
    candidates = [tokenizer.encode(f"The answer is {letter}.").ids for letter in "ABCD"]
    correct = 0
    for ex in ds:
        prompt = prompt_ids(_format_mmlu(ex)["messages"][0]["content"])
        scores = [completion_logprob(prompt, c) for c in candidates]
        correct += scores.index(max(scores)) == ex["answer"]
    return correct / n


def final_number(text: str) -> str | None:
    match = re.search(r"####\s*(-?[\d,.]+)", text)
    return match.group(1).replace(",", "").rstrip(".") if match else None


@torch.no_grad()
def gsm8k_accuracy(n: int) -> float:
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    correct = 0
    for ex in ds:
        prompt = prompt_ids(ex["question"])
        with autocast_ctx:
            out = model.generate(torch.tensor([prompt], device=device), max_new_tokens=300, top_k=1, stop_token=eos_id)
        answer = tokenizer.decode(out[0, len(prompt) :].tolist())
        correct += final_number(answer) == final_number(ex["answer"])
    return correct / n


if args.mmlu_n:
    print(f"MMLU  ({args.mmlu_n} questions): {mmlu_accuracy(args.mmlu_n):.1%}  (chance 25%)")
if args.gsm8k_n:
    print(f"GSM8K ({args.gsm8k_n} problems):  {gsm8k_accuracy(args.gsm8k_n):.1%}")
