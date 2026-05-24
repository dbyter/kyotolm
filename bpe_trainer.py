import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

BASE = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"

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

def text_iterator():
    for row in ds:
        text = row.get("text")
        if text:
            yield text

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],
)

tokenizer.train_from_iterator(
    text_iterator(),
    trainer=trainer,
    length=len(ds),
)

tokenizer.save("tokenizer.json")
print(tokenizer.decode(tokenizer.encode("Hello, world!")))