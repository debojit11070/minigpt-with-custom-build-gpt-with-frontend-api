# model/bangla_gpt.py
# Improved Bangla GPT with bigger model, longer training, larger context

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
block_size    = 128   # ↑ 64  → 128  (more context)
max_iters     = 10000 # ↑ 5000 → 10000 (longer training)
eval_interval = 500
learning_rate = 3e-4
min_lr        = 3e-5
eval_iters    = 200
n_embd        = 256   # ↑ 128 → 256  (richer embeddings)
n_head        = 8     # ↑ 4   → 8    (more attention heads)
n_layer       = 6     # ↑ 4   → 6    (deeper model)
dropout       = 0.2
device        = 'cpu'
max_grad_norm = 1.0

# new checkpoint name — different from old model
checkpoint_dir  = '../checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, 'bangla_improved_model.pth')

torch.manual_seed(1337)
os.makedirs(checkpoint_dir, exist_ok=True)

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print("=" * 60)
print("  Improved Bangla GPT")
print("=" * 60)
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
    """
    Cosine LR schedule with linear warmup:
    Phase 1 (0     → 200)      : linear warmup
    Phase 2 (200   → 10000)    : cosine decay
    Phase 3 (10000 → beyond)   : floor at min_lr
    """
    warmup_iters = 200

    # phase 1: linear warmup
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / warmup_iters

    # phase 3: minimum floor
    if iteration > max_iters:
        return min_lr

    # phase 2: cosine decay
    progress = (iteration - warmup_iters) / (max_iters - warmup_iters)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)

# ─────────────────────────────────────────
# SINGLE ATTENTION HEAD
# ─────────────────────────────────────────
class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.query   = nn.Linear(n_embd, head_size, bias=False)
        self.key     = nn.Linear(n_embd, head_size, bias=False)
        self.value   = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)

        # attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted aggregation
        v   = self.value(x)  # (B, T, head_size)
        out = wei @ v        # (B, T, head_size)
        return out

# ─────────────────────────────────────────
# MULTI HEAD ATTENTION
# ─────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run all heads in parallel then concatenate
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# ─────────────────────────────────────────
# FEEDFORWARD NETWORK
# ─────────────────────────────────────────
class FeedForward(nn.Module):
    """ expand → ReLU → compress """

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

# ─────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────
class Block(nn.Module):
    """
    One transformer block:
    communication (attention) + computation (feedforward)
    with residual connections and layer normalization
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # 256 // 8 = 32
        self.sa   = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual + pre-norm pattern
        x = x + self.sa(self.ln1(x))    # attention
        x = x + self.ffwd(self.ln2(x))  # feedforward
        return x

# ─────────────────────────────────────────
# FULL GPT MODEL
# ─────────────────────────────────────────
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

        # ← GPT-2 style weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        GPT-2 style initialization:
        - Linear: normal(mean=0, std=0.02)
        - Embedding: normal(mean=0, std=0.02)
        - Biases: zero
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # embeddings
        tok_emb = self.token_embedding_table(idx)                # (B, T, 256)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )                                                         # (T, 256)
        x = tok_emb + pos_emb                                    # (B, T, 256)

        # transformer blocks
        x = self.blocks(x)                                       # (B, T, 256)

        # final layer norm
        x = self.ln_f(x)                                         # (B, T, 256)

        # project to vocabulary
        logits = self.lm_head(x)                                 # (B, T, 151)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        temperature : controls randomness (0.5=safe, 1.0=normal, 1.5=creative)
        top_k       : only sample from top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # crop to last block_size tokens
            idx_cond = idx[:, -block_size:]

            # forward pass
            logits, loss = self(idx_cond)
            logits       = logits[:, -1, :]  # (B, vocab_size)

            # apply temperature
            logits = logits / temperature

            # apply top_k
            if top_k is not None:
                v      = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[0][:, [-1]]] = float('-inf')

            # sample
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
print(f"Block size        : {block_size}")
print(f"Max iterations    : {max_iters}")
print(f"Device            : {device}")

# ─────────────────────────────────────────
# RESUME FROM CHECKPOINT IF EXISTS
# ─────────────────────────────────────────
best_val_loss = float('inf')
start_iter    = 0
history       = {'train': [], 'val': [], 'lr': []}

if os.path.exists(checkpoint_path):
    print(f"\nFound checkpoint — resuming from {checkpoint_path}")
    checkpoint    = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['val_loss']
    start_iter    = checkpoint['iteration'] + 1
    print(f"Resumed from step {checkpoint['iteration']}")
    print(f"Best val loss so far: {best_val_loss:.4f}\n")
else:
    print("\nNo checkpoint found — starting fresh\n")

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
print("=" * 60)
print("  Training Improved Bangla GPT")
print(f"  block_size={block_size} | n_embd={n_embd} | n_layer={n_layer} | n_head={n_head}")
print("=" * 60)

for iter in range(start_iter, max_iters):

    # ── update learning rate every step ──
    current_lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    # ── evaluate and checkpoint ──
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses     = estimate_loss()
        train_loss = losses['train'].item()
        val_loss   = losses['val'].item()

        # track history
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

        # ── save best checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'iteration':            iter,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'train_loss':           train_loss,
                'vocab_size':           vocab_size,
                'hyperparameters': {
                    'block_size': block_size,
                    'n_embd':     n_embd,
                    'n_head':     n_head,
                    'n_layer':    n_layer,
                    'dropout':    dropout,
                }
            }, checkpoint_path)
            print(f" ← best saved! ✓", end="")

        print()

    # ── forward pass ──
    xb, yb       = get_batch('train')
    logits, loss = model(xb, yb)

    # ── backward pass ──
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # ── gradient clipping ──
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # ── update weights ──
    optimizer.step()

# ─────────────────────────────────────────
# TRAINING SUMMARY
# ─────────────────────────────────────────
print("=" * 60)
print(f"Training complete!")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Checkpoint    : {checkpoint_path}")
