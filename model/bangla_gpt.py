# model/bangla_gpt.py
# Full GPT trained on Bangla Wikipedia text

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))
from bangla_tokenizer import encode, decode, vocab_size, text

# ─────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────
batch_size    = 32
block_size    = 64
max_iters     = 5000
eval_interval = 500
learning_rate = 3e-4
min_lr        = 3e-5
eval_iters    = 200
n_embd        = 128
n_head        = 4
n_layer       = 4
dropout       = 0.2
device        = 'cpu'
max_grad_norm = 1.0

checkpoint_dir  = '../checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, 'bangla_best_model.pth')

torch.manual_seed(1337)
os.makedirs(checkpoint_dir, exist_ok=True)

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Dataset loaded!")
print(f"Total tokens  : {len(data):,}")
print(f"Train tokens  : {len(train_data):,}")
print(f"Val tokens    : {len(val_data):,}")
print(f"Vocab size    : {vocab_size}")

# ─────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────
def get_batch(split):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i+block_size]     for i in ix])
    y  = torch.stack([d[i+1:i+block_size+1] for i in ix])
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
# LEARNING RATE SCHEDULER
# ─────────────────────────────────────────
def get_lr(iteration):
    warmup_iters = 200
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / warmup_iters
    if iteration > max_iters:
        return min_lr
    progress = (iteration - warmup_iters) / (max_iters - warmup_iters)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)

# ─────────────────────────────────────────
# MODEL CLASSES
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
        self.blocks  = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )
        x      = tok_emb + pos_emb
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond     = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits       = logits[:, -1, :]
            logits       = logits / temperature
            if top_k is not None:
                v      = torch.topk(logits, min(top_k, logits.size(-1)))
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
print(f"\nModel parameters  : {total_params:.2f}M")
print(f"Layers            : {n_layer}")
print(f"Heads             : {n_head}")
print(f"Embedding dim     : {n_embd}")
print(f"Device            : {device}\n")

# ─────────────────────────────────────────
# RESUME FROM CHECKPOINT
# ─────────────────────────────────────────
best_val_loss = float('inf')
start_iter    = 0
history       = {'train': [], 'val': [], 'lr': []}

if os.path.exists(checkpoint_path):
    print(f"Found checkpoint — resuming...")
    checkpoint    = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['val_loss']
    start_iter    = checkpoint['iteration'] + 1
    print(f"Resumed from step {checkpoint['iteration']} | best val loss: {best_val_loss:.4f}\n")
else:
    print("Starting fresh training...\n")

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
print("=" * 60)
print("  Training Bangla GPT")
print("=" * 60)

for iter in range(start_iter, max_iters):

    # update learning rate
    current_lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    # evaluate and checkpoint
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses     = estimate_loss()
        train_loss = losses['train'].item()
        val_loss   = losses['val'].item()

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['lr'].append(current_lr)

        print(
            f"step {iter:5d} | "
            f"lr {current_lr:.6f} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f}",
            end=""
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'iteration':            iter,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'train_loss':           train_loss,
                'vocab_size':           vocab_size,
            }, checkpoint_path)
            print(f" ← best saved! ✓", end="")

        print()

    # training step
    xb, yb       = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

# ─────────────────────────────────────────
# TRAINING SUMMARY
# ─────────────────────────────────────────
print("=" * 60)
print(f"Training complete!")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Checkpoint    : {checkpoint_path}")

# ─────────────────────────────────────────
# LOAD BEST MODEL AND GENERATE
# ─────────────────────────────────────────
print("\nLoading best model...")
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("\n" + "=" * 60)
print("  GENERATED BANGLA TEXT")
print("=" * 60)

print("\n[Temperature 0.5 — Conservative]")
print("-" * 40)
print(decode(model.generate(
    context, max_new_tokens=200,
    temperature=0.5, top_k=40
)[0].tolist()))

print("\n[Temperature 1.0 — Normal]")
print("-" * 40)
print(decode(model.generate(
    context, max_new_tokens=200,
    temperature=1.0, top_k=40
)[0].tolist()))

print("\n[Temperature 1.5 — Creative]")
print("-" * 40)
print(decode(model.generate(
    context, max_new_tokens=200,
    temperature=1.5, top_k=40
)[0].tolist()))