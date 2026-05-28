import time
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer

parser = ArgumentParser()
parser.add_argument("--n", type=int, default=9, help="Number of shards to download")
args = parser.parse_args()

BASE = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
RAW_DATASET_PATH = Path("./datasets/saved_dataset")
TOKENIZED_PATH = Path("./tokenized_dataset")

if RAW_DATASET_PATH.exists():
    print(f"Loading raw dataset from {RAW_DATASET_PATH}")
    ds = load_from_disk(str(RAW_DATASET_PATH))
else:
    print(f"Loading {args.n} parquet shards from Hugging Face")
    ds = load_dataset(
        "parquet",
        data_files={
            "train": [f"{BASE}/shard_{i:05d}.parquet" for i in range(args.n)]
        },
        split="train",
        cache_dir="./hf_cache",
    )

tokenizer = Tokenizer.from_file("tokenizer.json")
print(f"Tokenizing {len(ds)} documents (batched, one-time)...")
t0 = time.perf_counter()


def encode_batch(examples):
    encodings = tokenizer.encode_batch(examples["text"])
    return {"text": [e.ids for e in encodings]}


encoded_ds = ds.map(encode_batch, batched=True, batch_size=1000, desc="tokenize")
encoded_ds.save_to_disk(str(TOKENIZED_PATH))
print(f"Saved tokenized dataset to {TOKENIZED_PATH} ({time.perf_counter() - t0:.1f}s)")
