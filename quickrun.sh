uv sync

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
N_SHARDS=160
PRETRAIN_STEPS=$(python -c "print(int($N_SHARDS * 60_000_000 / (24 * 2048 * $NGPU)))")

uv run torchrun --nproc_per_node=$NGPU -m training.pretrain \
    --n_layers 20 --n_heads 10 --n_embedding_dim 1280 \
    --n_shards $N_SHARDS \
    --steps_per_epoch $PRETRAIN_STEPS \
    --batch_size 24 \
    --learning_rate 1e-4 \
    --save_every 500 \
    --wandb_project kyotolm \
    --fp8