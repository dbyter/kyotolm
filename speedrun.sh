# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

wandb login

# Run the BPE trainer
uv run bpe_trainer.py

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
N_SHARDS=53
PRETRAIN_STEPS=$(python -c "print(int($N_SHARDS * 60_000_000 / (24 * 2048 * $NGPU)))")

uv run torchrun --nproc_per_node=$NGPU -m training.pretrain \
    --n_layers 20 --n_heads 10 --n_embedding_dim 1280 \
    --n_shards $N_SHARDS \
    --steps_per_epoch $PRETRAIN_STEPS \
    --batch_size 24 \
    --learning_rate 1e-4 \
    --save_every 500 \
    --wandb_project kyotolm \
    --no_wandb_artifacts \
    --fp8

# SFT — fine-tune the pretrained checkpoint on smol-smoltalk + MMLU + GSM8K
SFT_STEPS=$(python -c "print(460341 // 8 // $NGPU)")
uv run torchrun --nproc_per_node=$NGPU -m training.sft_trainer \
    --pretrained_checkpoint checkpoints/lm.pt \
    --steps_per_epoch $SFT_STEPS \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_every 500 \
    --wandb_project kyotolm \
    --no_wandb_artifacts \
    --fp8

