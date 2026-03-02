# model/tokenizer.py

import os

# Read the data
with open(os.path.join(os.path.dirname(__file__), '../data/input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

# Build vocabulary
chars      = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

# Functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Only runs when executing tokenizer.py directly
if __name__ == '__main__':
    print(f"Vocab size: {vocab_size}")
    print(f"All characters: {''.join(chars)}")
    print(f"First 10 characters: {chars[:10]}")
    print(f"Encoding 'hello': {encode('hello')}")
    print(f"Decoding [7, 4, 11, 11, 14]: {decode([7, 4, 11, 11, 14])}")