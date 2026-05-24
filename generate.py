"""
Generate text using the trained model.
"""

import torch
from llm import KyotoLM, Config
from tokenizers import Tokenizer

model = KyotoLM(Config(vocab_size=30000))
model.load_state_dict(torch.load("checkpoints/lm.pt"))
model.eval()

tokenizer = Tokenizer.from_file("tokenizer.json") 
if not tokenizer: 
    raise ValueError("Tokenizer not found")
    
while True:
    text = input("Enter a prompt: ")
    text = tokenizer.encode(text).ids
    # Add EOS token to the end of the prompt
    text = text + [tokenizer.encode("<eos>").ids[0]]
    print(tokenizer.decode(model.generate(text, max_new_tokens=100)))