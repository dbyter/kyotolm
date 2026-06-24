from pathlib import Path
from argparse import ArgumentParser
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

# n arg
parser = ArgumentParser()
parser.add_argument("--n", type=int, default=9, help="Number of shards to download")
args = parser.parse_args()

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

ds.save_to_disk(f"./datasets/saved_dataset")