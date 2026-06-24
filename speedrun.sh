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
uv run torchrun --nproc_per_node=$NGPU new_trainer.py --n_shards 40 --wandb_project kyotolm