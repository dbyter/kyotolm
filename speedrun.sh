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

# Run the trainer — streams data from HF Hub on demand, no pre-download needed
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
uv run torchrun --nproc_per_node=$NGPU -m training.pretrain \
    --n_layers 18 --n_heads 8 --n_embedding_dim 1024 \
    --n_shards 53 \
    --learning_rate 1e-4 \
    --save_every 500 \
    --wandb_project kyotolm \
    --fp8

