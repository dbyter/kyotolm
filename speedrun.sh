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

[ -n "$WANDB_API_KEY" ] || wandb login

# Run the BPE trainer
uv run bpe_trainer.py

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
N_SHARDS=${N_SHARDS:-153}  # ~9.2B tokens, ~10 tokens per param at d26
BATCH_SIZE=${BATCH_SIZE:-12}  # 12 x 5 x 2048 x 8 GPUs ~= 1M tokens/step, the batch the LRs were tuned at
GRAD_ACCUM=${GRAD_ACCUM:-5}
PRETRAIN_STEPS=$(python -c "print(int($N_SHARDS * 60_000_000 / ($BATCH_SIZE * $GRAD_ACCUM * 2048 * $NGPU)))")

uv run torchrun --nproc_per_node=$NGPU -m training.pretrain \
    --model v5 --optimizer muon \
    --n_layers 26 --n_heads 13 --n_embedding_dim 1664 \
    --n_shards $N_SHARDS \
    --steps_per_epoch $PRETRAIN_STEPS \
    --batch_size $BATCH_SIZE --grad_accum_steps $GRAD_ACCUM \
    --learning_rate 1e-3 --adamw_lr 1e-3 --lm_head_lr 3e-4 \
    --save_every 500 --artifact_every 1000 \
    --wandb_project kyotolm \
    --fp8

# SFT — fine-tune the pretrained checkpoint on smol-smoltalk + MMLU + GSM8K
# (steps are estimated from the packed dataset, so no --steps_per_epoch)
uv run torchrun --nproc_per_node=$NGPU -m training.sft_trainer \
    --optimizer muon \
    --pretrained_checkpoint checkpoints/lm.pt \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --save_every 500 \
    --wandb_project kyotolm \
    --fp8

uv run python evaluate.py checkpoints/sft.pt
