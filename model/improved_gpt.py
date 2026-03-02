# model/improved_gpt.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
import math   # ← NEW: needed for cosine scheduler

sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import encode, decode, vocab_size, text

# ─────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────
batch_size    = 32
block_size    = 64
max_iters     = 5000
eval_interval = 500
learning_rate = 3e-4
min_lr        = 3e-5   # ← NEW: minimum learning rate for scheduler
eval_iters    = 200
n_embd        = 128
n_head        = 4
n_layer       = 4
dropout       = 0.2
device        = 'cpu'
max_grad_norm = 1.0    # ← NEW: gradient clipping threshold

# ← NEW: checkpoint settings
checkpoint_dir  = '../checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

torch.manual_seed(1337)

# ─────────────────────────────────────────
# CREATE CHECKPOINT DIRECTORY
# ─────────────────────────────────────────
os.makedirs(checkpoint_dir, exist_ok=True)

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
# NEW FEATURE 1: LEARNING RATE SCHEDULER
# ─────────────────────────────────────────
def get_lr(iteration):
    """
    Cosine learning rate scheduler with warmup.

    3 phases:
    1. Warmup    — LR increases linearly from 0 to max
    2. Cosine    — LR decreases smoothly from max to min
    3. Minimum   — LR stays at min_lr floor
    """
    warmup_iters = 200   # first 200 steps: warmup phase

    # Phase 1: linear warmup
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / warmup_iters

    # Phase 3: minimum learning rate floor
    if iteration > max_iters:
        return min_lr

    # Phase 2: cosine decay
    # maps iteration to a value between 0 and 1
    progress = (iteration - warmup_iters) / (max_iters - warmup_iters)
    # cosine curve goes from 1.0 down to 0.0
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    # scale between min_lr and learning_rate
    return min_lr + coeff * (learning_rate - min_lr)

# ─────────────────────────────────────────
# MODEL CLASSES — same as gpt.py
# ─────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query   = nn.Linear(n_embd, head_size, bias=False)
        self.key     = nn.Linear(n_embd, head_size, bias=False)
        self.value   = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    # ─────────────────────────────────────────
    # NEW FEATURE 4: TEMPERATURE CONTROL
    # ─────────────────────────────────────────
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        temperature: controls randomness
            < 1.0 = more conservative, repetitive
            = 1.0 = normal
            > 1.0 = more creative, random

        top_k: only sample from top k most likely tokens
            None = sample from all tokens
            40   = only consider top 40 tokens
        """
        for _ in range(max_new_tokens):
            idx_cond     = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits       = logits[:, -1, :]  # (B, vocab_size)

            # ← NEW: apply temperature
            logits = logits / temperature

            # ← NEW: apply top_k filtering
            if top_k is not None:
                # find the kth largest value
                v = torch.topk(logits, min(top_k, logits.size(-1)))
                # set everything below kth value to -inf
                logits[logits < v[0][:, [-1]]] = float('-inf')

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx

# ─────────────────────────────────────────
# INITIALIZE MODEL
# ─────────────────────────────────────────
model     = GPTLanguageModel(vocab_size)
m         = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in m.parameters()) / 1e6
print(f"Model parameters: {total_params:.2f}M")
print(f"Vocab size: {vocab_size} | Layers: {n_layer} | Heads: {n_head} | Device: {device}\n")

# ─────────────────────────────────────────
# NEW FEATURE 2: CHECKPOINT TRACKING
# ─────────────────────────────────────────
best_val_loss = float('inf')   # track best validation loss
history       = {              # track loss history
    'train': [],
    'val':   [],
    'lr':    [],
}

# ─────────────────────────────────────────
# TRAINING LOOP — WITH ALL IMPROVEMENTS
# ─────────────────────────────────────────
print("Training Improved GPT...")
print("-" * 60)

for iter in range(max_iters):

    # ── NEW FEATURE 1: Update learning rate ──
    current_lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    # ── Evaluate and checkpoint ──
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        train_loss = losses['train'].item()
        val_loss   = losses['val'].item()

        # track history
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['lr'].append(current_lr)

        print(f"step {iter:5d} | lr {current_lr:.6f} | train loss {train_loss:.4f} | val loss {val_loss:.4f}", end="")

        # ── NEW FEATURE 2: Save best checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'iteration':  iter,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':   val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f" ← best saved! ✓", end="")

        print()  # newline

    # ── Forward pass ──
    xb, yb       = get_batch('train')
    logits, loss = model(xb, yb)

    # ── Backward pass ──
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # ── NEW FEATURE 3: Gradient clipping ──
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

# ─────────────────────────────────────────
# TRAINING SUMMARY
# ─────────────────────────────────────────
print("-" * 60)
print(f"\nTraining complete!")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Best model saved to: {checkpoint_path}")

# ─────────────────────────────────────────
# LOAD BEST MODEL FOR GENERATION
# ─────────────────────────────────────────
print("\nLoading best model for generation...")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from step {checkpoint['iteration']} with val loss {checkpoint['val_loss']:.4f}")

# ─────────────────────────────────────────
# GENERATE WITH DIFFERENT TEMPERATURES
# ─────────────────────────────────────────
print("\n" + "=" * 60)
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("\n[Temperature 0.5 — Conservative]")
print("-" * 40)
print(decode(model.generate(context, max_new_tokens=200, temperature=0.5, top_k=40)[0].tolist()))

print("\n[Temperature 1.0 — Normal]")
print("-" * 40)
print(decode(model.generate(context, max_new_tokens=200, temperature=1.0, top_k=40)[0].tolist()))

print("\n[Temperature 1.5 — Creative]")
print("-" * 40)
print(decode(model.generate(context, max_new_tokens=200, temperature=1.5, top_k=40)[0].tolist()))