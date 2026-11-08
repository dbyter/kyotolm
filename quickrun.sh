uv sync

NGPU=$(uv run python -c "import torch; print(torch.cuda.device_count())")
N_SHARDS=${N_SHARDS:-153}  # ~9.2B tokens, ~10 tokens per param at d26
BATCH_SIZE=${BATCH_SIZE:-12}  # 12 x 5 x 2048 x 8 GPUs ~= 1M tokens/step, the batch the LRs were tuned at
GRAD_ACCUM=${GRAD_ACCUM:-5}
PRETRAIN_STEPS=$(uv run python -c "print(int($N_SHARDS * 60_000_000 / ($BATCH_SIZE * $GRAD_ACCUM * 2048 * $NGPU)))")

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