# model/gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import encode, decode, vocab_size, text

# ─────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────
batch_size    = 32
block_size    = 64    # ↑ increased from 8 — more context
max_iters     = 5000  # ↑ increased — more training
eval_interval = 500
learning_rate = 3e-4  # ↓ decreased — more careful learning
eval_iters    = 200
n_embd        = 128   # ↑ increased — richer embeddings
n_head        = 4
n_layer       = 4     # ← NEW: number of transformer blocks
dropout       = 0.2   # ← NEW: regularization
device        = 'cpu'

torch.manual_seed(1337)

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ─────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i+block_size]     for i in ix])
    y    = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ─────────────────────────────────────────
# LOSS ESTIMATOR
# ─────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y         = get_batch(split)
            logits, loss = model(X, Y)
            losses[k]    = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ─────────────────────────────────────────
# SINGLE ATTENTION HEAD
# ─────────────────────────────────────────
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.query   = nn.Linear(n_embd, head_size, bias=False)
        self.key     = nn.Linear(n_embd, head_size, bias=False)
        self.value   = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # ← NEW: dropout on attention weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # ← NEW: randomly drop some attention connections during training
        wei = self.dropout(wei)

        v   = self.value(x)
        out = wei @ v
        return out

# ─────────────────────────────────────────
# MULTI HEAD ATTENTION
# ─────────────────────────────────────────
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        # ← NEW: dropout after projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # ← NEW: dropout applied after projection
        out = self.dropout(self.proj(out))
        return out

# ─────────────────────────────────────────
# FEEDFORWARD
# ─────────────────────────────────────────
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # ← NEW: dropout at end of feedforward
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────
# TRANSFORMER BLOCK ← NEW CLASS
# ─────────────────────────────────────────
class Block(nn.Module):
    """
    One full transformer block:
    - LayerNorm before attention     (normalize inputs)
    - Multi-Head Attention           (tokens communicate)
    - Residual connection            (skip connection)
    - LayerNorm before feedforward   (normalize again)
    - FeedForward                    (tokens compute)
    - Residual connection            (skip connection)
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size  = n_embd // n_head
        self.sa    = MultiHeadAttention(n_head, head_size)
        self.ffwd  = FeedForward(n_embd)
        # ← NEW: two layer norms per block
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connection + pre layer norm before attention
        x = x + self.sa(self.ln1(x))    # ← NEW pattern
        # residual connection + pre layer norm before feedforward
        x = x + self.ffwd(self.ln2(x))  # ← NEW pattern
        return x

# ─────────────────────────────────────────
# FULL GPT MODEL ← MAIN NEW CLASS
# ─────────────────────────────────────────
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # ← NEW: stack n_layer blocks sequentially
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        # ← NEW: final layer norm after all blocks
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                               # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x       = tok_emb + pos_emb    # (B, T, n_embd)

        # ← NEW: pass through ALL blocks sequentially
        x = self.blocks(x)             # (B, T, n_embd)

        # ← NEW: final layer norm
        x = self.ln_f(x)               # (B, T, n_embd)

        logits = self.lm_head(x)       # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond     = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits       = logits[:, -1, :]
            probs        = F.softmax(logits, dim=-1)
            idx_next     = torch.multinomial(probs, num_samples=1)
            idx          = torch.cat((idx, idx_next), dim=1)
        return idx

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
model = GPTLanguageModel(vocab_size)
m     = model.to(device)

# print number of parameters
total_params = sum(p.numel() for p in m.parameters()) / 1e6
print(f"Model parameters: {total_params:.2f}M")
print(f"Vocab size: {vocab_size} | Layers: {n_layer} | Heads: {n_head} | Device: {device}\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training Full GPT...")
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb       = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete! Generating text...\n")
print("-" * 50)

# ─────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
print("-" * 50)

# ─────────────────────────────────────────
# SAVE THE MODEL
# ─────────────────────────────────────────
torch.save(model.state_dict(), '../model/gpt_weights.pth')
print("\nModel saved to gpt_weights.pth")