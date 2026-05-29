"""
Streaming data loader for DDP training.

Pulls parquet shards from HF Hub on demand — no pre-download or pre-tokenization needed.
Each DDP rank gets its own non-overlapping slice of the stream via split_dataset_by_node.
"""

from argparse import ArgumentParser

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
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
