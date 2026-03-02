# api/model_loader.py
# Loads and manages both English and Bangla GPT models

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import math

# ─────────────────────────────────────────
# MODEL ARCHITECTURE
# must match exactly what we trained
# ─────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
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
        q   = self.query(x)
        k   = self.key(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads   = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout)
             for _ in range(num_heads)]
        )
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa   = MultiHeadAttention(
            n_embd, n_head, head_size, block_size, dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer,
                 block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout)
              for _ in range(n_layer)]
        )
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T    = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x      = tok_emb + pos_emb
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss    = F.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens,
                 temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature
            if top_k is not None:
                v      = torch.topk(
                    logits, min(top_k, logits.size(-1))
                )
                logits[logits < v[0][:, [-1]]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx

# ─────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CHECKPOINTS = {
#     'english': os.path.join(
#         os.path.dirname(__file__),
#         '../checkpoints/english_maxed_model.pth'
#     ),
#     'bangla': os.path.join(
#         os.path.dirname(__file__),
#         '../checkpoints/bangla_improved_model.pth'
#     ),
# }
from huggingface_hub import hf_hub_download

# ── Hugging Face repo ──
HF_REPO_ID = "debojitbasak/minigpt-models"

# ── local cache directory ──
CACHE_DIR = os.path.join(os.path.dirname(__file__), '../checkpoints')
os.makedirs(CACHE_DIR, exist_ok=True)

# ── model filenames on HF ──
HF_FILENAMES = {
    'english': 'english_maxed_model.pth',
    'bangla':  'bangla_improved_model.pth',
}

def get_checkpoint_path(language: str) -> str:
    """
    Returns local path to checkpoint.
    Downloads from Hugging Face if not found locally.
    """
    filename   = HF_FILENAMES[language]
    local_path = os.path.join(CACHE_DIR, filename)

    # if already exists locally — use it
    if os.path.exists(local_path):
        print(f"  Found locally: {local_path}")
        return local_path

    # otherwise download from Hugging Face
    print(f"  Downloading {filename} from Hugging Face...")
    print(f"  Repo: {HF_REPO_ID}")

    downloaded_path = hf_hub_download(
        repo_id   = HF_REPO_ID,
        filename  = filename,
        repo_type = "model",
        local_dir = CACHE_DIR,
    )

    print(f"  ✓ Downloaded to {downloaded_path}")
    return downloaded_path

# cache loaded models — load once, reuse forever
_models  = {}
_configs = {}

# def load_model(language: str):
#     """
#     Load a model by language key.
#     Returns (model, stoi, itos) tuple.
#     Caches in memory after first load.
#     """
#     if language in _models:
#         return _models[language]

#     path = CHECKPOINTS.get(language)
#     if not path or not os.path.exists(path):
#         raise FileNotFoundError(
#             f"Checkpoint not found for '{language}': {path}"
#         )

#     print(f"Loading {language} model from {path}...")
#     checkpoint = torch.load(path, map_location=device)

def load_model(language: str):
    if language in _models:
        return _models[language]

    print(f"Loading {language} model...")
    path       = get_checkpoint_path(language)
    checkpoint = torch.load(path, map_location=device)

    # get hyperparameters from checkpoint
    hp         = checkpoint['hyperparameters']
    vocab_size = checkpoint['vocab_size']
    stoi       = checkpoint['stoi']
    itos       = checkpoint['itos']

    # build model with saved hyperparameters
    model = GPTLanguageModel(
        vocab_size = vocab_size,
        n_embd     = hp['n_embd'],
        n_head     = hp['n_head'],
        n_layer    = hp['n_layer'],
        block_size = hp['block_size'],
        dropout    = hp['dropout'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded! {params:.2f}M params | "
          f"vocab={vocab_size} | "
          f"val_loss={checkpoint['val_loss']:.4f}")

    _models[language] = (model, stoi, itos)
    return _models[language]

def generate_text(
    language:    str,
    prompt:      str   = "",
    max_tokens:  int   = 200,
    temperature: float = 1.0,
    top_k:       int   = 40,
) -> str:
    """
    Generate text using the specified model.
    """
    model, stoi, itos = load_model(language)

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    # encode prompt or start from zero token
    if prompt and prompt.strip():
        encoded = encode(prompt)
        if not encoded:
            encoded = [0]
        context = torch.tensor(
            encoded, dtype=torch.long, device=device
        ).unsqueeze(0)
    else:
        context = torch.zeros(
            (1, 1), dtype=torch.long, device=device
        )

    # generate
    output = model.generate(
        context,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    full_text = decode(output[0].tolist())

    # return only generated part if prompt was given
    if prompt and prompt.strip():
        return full_text[len(prompt):]
    return full_text

def get_model_info(language: str) -> dict:
    """
    Return model metadata for display in UI.
    """
    try:
        path = get_checkpoint_path(language)
    except Exception:
        return {'error': f'Model not found: {language}'}

    checkpoint = torch.load(path, map_location=device)
    hp         = checkpoint['hyperparameters']

    return {
        'language':   language,
        'val_loss':   round(checkpoint['val_loss'], 4),
        'vocab_size': checkpoint['vocab_size'],
        'parameters': round(sum(
            p.numel() for p in
            GPTLanguageModel(
                checkpoint['vocab_size'],
                hp['n_embd'], hp['n_head'],
                hp['n_layer'], hp['block_size'],
                hp['dropout']
            ).parameters()
        ) / 1e6, 2),
        'hyperparameters': hp,
    }