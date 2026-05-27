# Run the BPE trainer
python bpe_trainer.py

# Run the data downloader
python data_downloader.py

# Run the trainer
torchrun --nproc_per_node=8 new_trainer.py