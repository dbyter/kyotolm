from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

BASE = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

# Arrow dataset on disk — delete this folder if you change shards or tokenizer.json
TOKENIZED_PATH = Path("./tokenized_dataset")
CHECKPOINT_PATH = Path("./checkpoints/lm.pt")

ds = load_dataset(
    "parquet",
    data_files={
        "train": [
            f"{BASE}/shard_00000.parquet",
            f"{BASE}/shard_00001.parquet",
            f"{BASE}/shard_00002.parquet",
            f"{BASE}/shard_00003.parquet",
            f"{BASE}/shard_00004.parquet",
            f"{BASE}/shard_00005.parquet",
            f"{BASE}/shard_00006.parquet",
            f"{BASE}/shard_00007.parquet",
            f"{BASE}/shard_00008.parquet",
            f"{BASE}/shard_00009.parquet",
        ]
    },
    split="train",
    cache_dir="./hf_cache",
)

tokenizer = Tokenizer.from_file("tokenizer.json")

# batched=True → each column is a list; use encode_batch
def encode_batch(examples):
    encodings = tokenizer.encode_batch(examples["text"])
    return {"text": [e.ids for e in encodings]}

encoded_ds = ds.map(encode_batch, batched=True)
encoded_ds.save_to_disk(str(TOKENIZED_PATH))
