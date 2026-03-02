# model/bangla_tokenizer.py

import os

# ─────────────────────────────────────────
# READ BANGLA DATASET
# ─────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), '../data/bangla_clean.txt')

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# ─────────────────────────────────────────
# BUILD VOCABULARY
# ─────────────────────────────────────────
chars      = sorted(list(set(text)))
vocab_size = len(chars)

# ─────────────────────────────────────────
# MAPPINGS
# ─────────────────────────────────────────
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

# ─────────────────────────────────────────
# ENCODE / DECODE
# ─────────────────────────────────────────
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# ─────────────────────────────────────────
# TEST WHEN RUN DIRECTLY
# ─────────────────────────────────────────
if __name__ == '__main__':
    print(f"Vocab size      : {vocab_size}")
    print(f"Total chars     : {len(text):,}")

    # show bangla specific stats
    bangla_chars = [c for c in chars if '\u0980' <= c <= '\u09FF']
    print(f"Bangla chars    : {len(bangla_chars)}")
    print(f"\nSample text     :")
    print(text[:200])
    print(f"\nEncoded sample  : {encode(text[:10])}")
    print(f"Decoded back    : {decode(encode(text[:10]))}")