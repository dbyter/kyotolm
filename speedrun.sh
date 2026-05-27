#!/usr/bin/env bash
set -euo pipefail

# Install deps from pyproject.toml into .venv
uv sync

# Run the BPE trainer
uv run bpe_trainer.py

# Run the data downloader
uv run data_downloader.py

# Run the trainer (use uv so torchrun uses the project venv)
uv run torchrun --nproc_per_node=8 new_trainer.py