import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import encode, decode, vocab_size, text


batch_size    = 32
block_size    = 8
max_iters     = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters    = 200
device        = 'cpu'
n_embd        = 32   # ← NEW: embedding dimension size
torch.manual_seed(1337)

# data
data       = torch.tensor(encode(text), dtype=torch.long)
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# data loader

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i+block_size]     for i in ix])
    y    = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# loss estimetor

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


# self attention head

class Head(nn.Module):
    """ one head of self attention """

    def __init__(self, head_size):
        super().__init__()
        # these three linear layers create Q, K, V
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # tril is a lower triangular matrix — used for masking
        # it's not a parameter so we register it as a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # every token produces a query and a key
        q = self.query(x)  # (B, T, head_size) — "what am I looking for?"
        k = self.key(x)    # (B, T, head_size) — "what do I contain?"

        # compute attention scores — how much does each token attend to others?
        # scale by head_size**-0.5 to keep values stable (prevent softmax saturation)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)

        # MASKING — future tokens cannot communicate with past tokens
        # replace upper triangle with -inf so softmax makes them 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # softmax turns scores into probabilities
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # weighted aggregation of values
        v   = self.value(x)   # (B, T, head_size) — "what do I share?"
        out = wei @ v         # (B, T, head_size)
        return out


## model with self attention

class BigramWithAttention(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # token embedding — now projects to n_embd instead of vocab_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        # position embedding — each position gets its own embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # one head of self attention
        self.sa_head = Head(n_embd)
        # final linear layer to project back to vocab_size
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # token embeddings — what is each character?
        tok_emb = self.token_embedding_table(idx)             # (B, T, n_embd)
        # position embeddings — where is each character?
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)

        # add them together — now each token knows WHAT it is AND WHERE it is
        x = tok_emb + pos_emb        # (B, T, n_embd)

        # pass through self attention
        x = self.sa_head(x)          # (B, T, n_embd)

        # project to vocabulary scores
        logits = self.lm_head(x)     # (B, T, vocab_size)

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
            # crop to last block_size tokens — attention has a size limit
            idx_cond     = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits       = logits[:, -1, :]
            probs        = F.softmax(logits, dim=-1)
            idx_next     = torch.multinomial(probs, num_samples=1)
            idx          = torch.cat((idx, idx_next), dim=1)
        return idx
    
# training loop
model     = BigramWithAttention(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training with Self-Attention...")
print(f"Vocab size: {vocab_size} | Device: {device}\n")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb       = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete! Generating text...\n")
print("-" * 50)


# generate

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
print("-" * 50)