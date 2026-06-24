"""
Streaming data loaders for pre-training and SFT.

Pre-training: pulls parquet shards from HF Hub on demand, tokenizes into fixed-length chunks.
SFT: streams HuggingFaceTB/smol-smoltalk, formats as User/Assistant turns, masks prompt tokens.
Each DDP rank gets its own non-overlapping slice via split_dataset_by_node.
"""

from argparse import ArgumentParser

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from datasets.distributed import split_dataset_by_node
from tokenizers import Tokenizer

parser = ArgumentParser()
parser.add_argument("--n", type=int, default=40, help="Number of shards to stream")
args, _ = parser.parse_known_args()

BASE = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"


class StreamingTokenDataset(IterableDataset):
    """
    Streams documents from HF, tokenizes on the fly, and yields (input, target) token
    sequences of length seq_len. Maintains a rolling buffer across documents so no
    tokens are wasted at document boundaries.
    """

    def __init__(self, hf_ds, tokenizer_path: str, seq_len: int):
        self.hf_ds = hf_ds
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len

    def __iter__(self):
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        eos_ids = tokenizer.encode("<eos>").ids

        buffer = []
        for doc in self.hf_ds:
            buffer.extend(tokenizer.encode(doc["text"]).ids)
            buffer.extend(eos_ids)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


class SFTDataset(IterableDataset):
    """
    Streams smol-smoltalk, formats each conversation as:
        <bos>User: {content}\n\nAssistant: {content}<eos>User: ...
    Yields (x, y) where y has -100 at prompt/user positions so only
    assistant tokens contribute to the loss.
    """

    def __init__(self, hf_ds, tokenizer_path: str, seq_len: int):
        self.hf_ds = hf_ds
        self.tokenizer_path = tokenizer_path
        self.seq_len = seq_len

    def __iter__(self):
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")

        for sample in self.hf_ds:
            token_ids = [bos_id]
            mask = [0]  # 1 = assistant token (contributes to loss), 0 = prompt token

            for turn in sample["messages"]:
                role = turn["role"]
                content = turn["content"]
                if role == "system":
                    ids = tokenizer.encode(f"System: {content}\n\n").ids
                    token_ids.extend(ids)
                    mask.extend([0] * len(ids))
                elif role == "user":
                    ids = tokenizer.encode(f"User: {content}\n\nAssistant: ").ids
                    token_ids.extend(ids)
                    mask.extend([0] * len(ids))
                elif role == "assistant":
                    ids = tokenizer.encode(content).ids
                    token_ids.extend(ids)
                    mask.extend([1] * len(ids))
                    token_ids.append(eos_id)
                    mask.append(1)

            # truncate to seq_len + 1 (need the extra token for x/y shift)
            token_ids = token_ids[: self.seq_len + 1]
            mask = mask[: self.seq_len + 1]

            # skip if no assistant tokens remain after truncation
            if len(token_ids) < 2 or sum(mask[1:]) == 0:
                continue

            # pad to fixed length so DataLoader default collation works
            pad = self.seq_len + 1 - len(token_ids)
            token_ids += [0] * pad
            mask += [0] * pad

            x = torch.tensor(token_ids[:-1], dtype=torch.long)
            y = torch.tensor(token_ids[1:], dtype=torch.long)
            loss_mask = torch.tensor(mask[1:], dtype=torch.bool)
            y[~loss_mask] = -100
            yield x, y


def _to_messages(role_user: str, role_assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": role_user}, {"role": "assistant", "content": role_assistant}]}


def _format_gsm8k(example: dict) -> dict:
    return _to_messages(example["question"], example["answer"])


def _format_mmlu(example: dict) -> dict:
    letters = ["A", "B", "C", "D"]
    choices = "\n".join(f"{l}) {c}" for l, c in zip(letters, example["choices"]))
    answer = f"The answer is {letters[example['answer']]}."
    return _to_messages(f"{example['question']}\n{choices}", answer)


def make_sft_dataset(rank: int, world_size: int, seq_len: int = 2048, split: str = "train") -> SFTDataset:
    smoltalk = load_dataset("HuggingFaceTB/smol-smoltalk", split=split, streaming=True)

    gsm8k_split = "train" if split == "train" else "test"
    gsm8k = load_dataset("openai/gsm8k", "main", split=gsm8k_split, streaming=True).map(
        _format_gsm8k, remove_columns=["question", "answer"]
    )

    mmlu_split = "auxiliary_train" if split == "train" else "test"
    mmlu = load_dataset("cais/mmlu", "all", split=mmlu_split, streaming=True).map(
        _format_mmlu, remove_columns=["question", "choices", "answer", "subject"]
    )

    # ~0.7 smoltalk / ~0.2 mmlu / ~0.1 gsm8k — roughly matches nanochat's mixture ratios
    combined = interleave_datasets(
        [smoltalk, mmlu, gsm8k],
        probabilities=[0.7, 0.2, 0.1],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    combined = split_dataset_by_node(combined, rank=rank, world_size=world_size)
    return SFTDataset(combined, "tokenizer.json", seq_len)


def make_dataset(rank: int, world_size: int, seq_len: int = 2048, n_shards: int | None = None) -> StreamingTokenDataset:
    n = n_shards if n_shards is not None else args.n
    hf_ds = load_dataset(
        "parquet",
        data_files={"train": [f"{BASE}/shard_{i:05d}.parquet" for i in range(n)]},
        split="train",
        streaming=True,
    )
    hf_ds = split_dataset_by_node(hf_ds, rank=rank, world_size=world_size)
    return StreamingTokenDataset(hf_ds, "tokenizer.json", seq_len)
