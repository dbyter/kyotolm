"""
Handles data loading for DDP training and tokenizes the data before returning the data loader
"""

import time
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer

parser = ArgumentParser()
parser.add_argument("--n", type=int, default=9, help="Number of shards to download")
args = parser.parse_args()

BASE = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
RAW_DATASET_PATH = Path("./datasets/saved_dataset")
TOKENIZED_PATH = Path("./tokenized_dataset")


def _load_raw_dataset():
    if RAW_DATASET_PATH.exists():
        return load_from_disk(str(RAW_DATASET_PATH))

    return load_dataset(
        "parquet",
        data_files={
            "train": [f"{BASE}/shard_{i:05d}.parquet" for i in range(args.n)]
        },
        split="train",
        cache_dir="./hf_cache",
    )


def _get_data_batch(ds, rank, batch_size):
    start = rank * batch_size
    end = start + batch_size
    return ds.select(range(start, end))


def _build_sequences(batch, eos_ids, seq_len):
    all_tokens = []
    n_docs = len(batch)
    t0 = time.perf_counter()

    for doc_idx, row in enumerate(batch):
        all_tokens.extend(row["text"])
        all_tokens.extend(eos_ids)
        if doc_idx > 0 and doc_idx % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"\t\tProcessed {doc_idx}/{n_docs} docs ({elapsed:.1f}s)")

    token_arr = np.array(all_tokens, dtype=np.int32)
    del all_tokens

    n_seq = (len(token_arr) - 1) // seq_len
    usable = n_seq * seq_len + 1
    token_arr = token_arr[:usable]

    input_sequences  = token_arr[:-1].reshape(n_seq, seq_len)
    output_sequences = token_arr[1:].reshape(n_seq, seq_len)

    elapsed = time.perf_counter() - t0
    print(
        f"\t\tBuilt {n_seq} sequences from {n_docs} docs "
        f"({len(token_arr):,} tokens, {elapsed:.1f}s)"
    )
    return input_sequences, output_sequences


def load_data(rank, world_size, seq_len=2048):
    tokenizer = Tokenizer.from_file("tokenizer.json")
    eos_ids = tokenizer.encode("<eos>").ids
    print(f"\t\tLoaded Tokenizer")

    if TOKENIZED_PATH.exists():
        print(f"\t\tLoading pre-tokenized dataset from {TOKENIZED_PATH}")
        encoded_ds = load_from_disk(str(TOKENIZED_PATH))
    else:
        print("\t\tNo tokenized cache found; tokenizing this rank's shard (batched)")
        raw_ds = _load_raw_dataset()
        print(f"\t\tDataset length: {len(raw_ds)}")
        batch_size = len(raw_ds) // world_size
        batch = _get_data_batch(raw_ds, rank, batch_size)

        def encode_batch(examples):
            encodings = tokenizer.encode_batch(examples["text"])
            return {"text": [e.ids for e in encodings]}

        t0 = time.perf_counter()
        batch = batch.map(
            encode_batch,
            batched=True,
            batch_size=1000,
            desc=f"tokenize rank {rank}",
        )
        print(f"\t\tTokenized {len(batch)} docs in {time.perf_counter() - t0:.1f}s")

    if TOKENIZED_PATH.exists():
        print(f"\t\tDataset length: {len(encoded_ds)}")
        batch_size = len(encoded_ds) // world_size
        batch = _get_data_batch(encoded_ds, rank, batch_size)

    return _build_sequences(batch, eos_ids, seq_len)
