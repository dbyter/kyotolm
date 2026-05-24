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

if TOKENIZED_PATH.exists():
    encoded_ds = load_from_disk(str(TOKENIZED_PATH))
else:
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
                # f"{BASE}/shard_00006.parquet",
                # f"{BASE}/shard_00007.parquet",
                # f"{BASE}/shard_00008.parquet",
                # f"{BASE}/shard_00009.parquet",
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

tokenizer = Tokenizer.from_file("tokenizer.json")

# Create input & output tensors
SEQ_LEN = 2048

input_sequences = []
output_sequences = []
all_tokens = []

eos_ids = tokenizer.encode("<eos>").ids

for row in encoded_ds:
    text = row["text"]
    all_tokens.extend(text)
    all_tokens.extend(eos_ids)

i = 0
while i + SEQ_LEN + 1 <= len(all_tokens):
    input_sequences.append(all_tokens[i : i + SEQ_LEN])
    output_sequences.append(all_tokens[i + 1 : i + SEQ_LEN + 1])
    i += SEQ_LEN

if __name__ == "__main__":

    from trainer import train_lm

    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    train_lm(
        input_sequences[:100],
        output_sequences[:100],
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        checkpoint_path=CHECKPOINT_PATH,
    )