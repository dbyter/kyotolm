"""
Handles data loading for DDP training and tokenizes the data before returning the data loader

"""

import datasets
from argparse import ArgumentParser
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument("--n", type=int, default=9, help="Number of shards to download")
args = parser.parse_args()

def _get_data_batch(ds, rank, batch_size):
    start = rank * batch_size
    end = start + batch_size
    return ds.select(range(start, end))

def load_data(rank, world_size, seq_len=2048):
    BASE = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": [
                f"{BASE}/shard_{i:05d}.parquet" for i in range(args.n)
            ]
        },
        split="train",
        cache_dir="./hf_cache",
    )
    print(f"\t\tDataset length: {len(ds)}")
    batch_size = len(ds) // world_size
    batch = _get_data_batch(ds, rank, batch_size)
    tokenizer = Tokenizer.from_file("tokenizer.json")
    print(f"\t\tLoaded Tokenizer")

    input_sequences = []
    output_sequences = []
    all_tokens = []
    
    eos_ids = tokenizer.encode("<eos>").ids
    #print(f"Length of batch: {batch}; example: {batch[0]}")
    #print(f"EOS ids: {eos_ids}")
    for row in batch:
        text = row["text"]
        all_tokens.extend(tokenizer.encode(text).ids)
        all_tokens.extend(eos_ids)

    i = 0
    while i + seq_len + 1 <= len(all_tokens):
        input_sequences.append(all_tokens[i : i + seq_len])
        output_sequences.append(all_tokens[i + 1 : i + seq_len + 1])
        i += seq_len

    return input_sequences, output_sequences