# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Run the BPE trainer
uv run bpe_trainer.py

# Run the data downloader
uv run data_downloader.py --n 40

# Tokenize once and save to disk (trainer loads this instead of re-tokenizing)
uv run data_tokenizer.py --n 40

# Run the trainer (use uv so torchrun uses the project venv)
uv run torchrun --nproc_per_node=8 new_trainer.py