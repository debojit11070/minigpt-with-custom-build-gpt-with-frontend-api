# 🤖 MiniGPT — Built From Scratch

A GPT language model implemented from scratch using PyTorch, trained on English (Shakespeare) and Bangla (Wikipedia) text. Built as a portfolio project demonstrating transformer architecture from first principles.

## 🌐 Live Demo
[debojitbasak-minigpt.hf.space](https://debojitbasak-minigpt.hf.space)

---

## 📌 Project Overview

This project follows the full ML lifecycle:
- **Architecture** — built transformer from scratch, starting from bigram model up to full GPT
- **Training** — trained on Kaggle T4 GPU with mixed precision and cosine LR scheduling
- **Backend** — FastAPI serving both models via REST API
- **Frontend** — Next.js + Tailwind CSS web interface
- **Deployment** — Hugging Face Spaces with models hosted on Hugging Face Hub

---

## 📈 Model Progress

| Model | Val Loss | Params | Device |
|---|---|---|---|
| Bigram | 2.49 | tiny | CPU |
| Self-Attention | 2.43 | tiny | CPU |
| Multi-Head Attention | 2.25 | 0.8M | CPU |
| Full GPT | 1.76 | 0.82M | CPU |
| Improved GPT | 1.87 | 0.82M | CPU |
| **English GPT (Kaggle)** | **1.51** | **25.4M** | **T4 GPU** |
| **Bangla GPT (Kaggle)** | **1.66** | **10.8M** | **T4 GPU** |

---

## 🏗️ Architecture
```
Input tokens
     ↓
Token Embedding + Positional Embedding
     ↓
[ Transformer Block ] × N layers
   ├── Multi-Head Self-Attention
   │     ├── Query, Key, Value projections
   │     ├── Scaled dot-product attention
   │     └── Causal masking (no future tokens)
   ├── Residual connection + LayerNorm
   ├── Feed Forward (Linear → GELU → Linear)
   └── Residual connection + LayerNorm
     ↓
LayerNorm
     ↓
Linear → Logits → Softmax → Next token
```

### English Model
- 25.4M parameters
- 8 layers, 8 heads, 512 embedding dim
- Trained on Tiny Shakespeare (1MB)
- Context length: 256 tokens

### Bangla Model
- 10.8M parameters
- 6 layers, 6 heads, 384 embedding dim
- Trained on Bangla Wikipedia (5MB, 80+ articles)
- Context length: 256 tokens
- Vocab size: 160 (Bangla Unicode + ASCII)

---

## ⚙️ Training Details

| Setting | English | Bangla |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-4 → 3e-5 | 3e-4 → 3e-5 |
| LR Schedule | Cosine decay + warmup | Cosine decay + warmup |
| Batch size | 64 | 64 |
| Mixed precision | FP16 | FP16 |
| Gradient clipping | 1.0 | 1.0 |
| Weight init | GPT-2 style | GPT-2 style |
| GPU | Kaggle T4 | Kaggle T4 |
| Training time | ~85 min | ~15 min |

---

## 🗂️ Project Structure
```
minigpt/
├── model/                        # Model architecture — progression
│   ├── bigram.py                 # Step 1: bigram baseline (loss 2.49)
│   ├── attention.py              # Step 2: single head attention (2.43)
│   ├── multihead.py              # Step 3: multi-head attention (2.25)
│   ├── gpt.py                    # Step 4: full GPT (1.76)
│   ├── improved_gpt.py           # Step 5: LR scheduler + checkpointing
│   ├── bangla_gpt.py             # Step 6: Bangla GPT
│   ├── tokenizer.py              # English char-level tokenizer
│   └── bangla_tokenizer.py       # Bangla Unicode tokenizer
│
├── api/                          # FastAPI backend
│   ├── main.py                   # API routes
│   └── model_loader.py           # Model loading + HF Hub download
│
├── webapp/                       # Next.js frontend
│   └── app/
│       ├── page.js               # Main UI
│       ├── layout.js
│       └── globals.css
│
├── scraper.py                    # Bangla Wikipedia scraper
├── data_inspector.py             # Data cleaning tool
├── upload_models.py              # Upload models to HF Hub
└── requirements.txt
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### Backend
```bash
# clone the repo
git clone https://github.com/YOURUSERNAME/minigpt.git
cd minigpt

# create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install -r requirements.txt

# run the API
cd api
uvicorn main:app --reload --port 8000
```

Models are automatically downloaded from Hugging Face Hub on first run.

### Frontend
```bash
cd webapp
npm install
npm run dev
```

Open `http://localhost:3000`

---

## 🔌 API Reference

### `POST /generate`
Generate text using either model.

**Request:**
```json
{
  "language": "english",
  "prompt": "ROMEO:",
  "max_tokens": 200,
  "temperature": 0.8,
  "top_k": 40
}
```

**Response:**
```json
{
  "generated_text": "ROMEO:\nIf you bid me good...",
  "prompt": "ROMEO:",
  "language": "english",
  "time_taken": 12.4,
  "tokens_generated": 200
}
```

### `GET /models`
Returns info about both loaded models.

### `GET /health`
Health check endpoint.

---

## 💡 Key Concepts Implemented

- **Scaled dot-product attention** — Q·Kᵀ/√d
- **Causal masking** — prevents attending to future tokens
- **Multi-head attention** — parallel attention heads
- **Residual connections** — prevents vanishing gradients
- **Layer normalization** — pre-norm style (GPT-2)
- **GELU activation** — smoother than ReLU for transformers
- **Cosine LR decay** — with linear warmup
- **Mixed precision** — FP16 forward, FP32 gradients
- **Top-k sampling** — controls generation diversity
- **Temperature scaling** — controls randomness

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch (from scratch) |
| Training | Kaggle T4 GPU |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 14 + Tailwind CSS v4 |
| Model Storage | Hugging Face Hub |
| Deployment | Hugging Face Spaces |

---

## 📝 Sample Output

**English (prompt: "ROMEO:"):**
```
ROMEO:
If you bid me good aboad, and so this injury time,
You should be most grown to play the children;
If not long the hungry mercy of you
As like an unstainting of your brother's virtues
```

**Bangla (prompt: "বাংলাদেশ একটি"):**
```
বাংলাদেশ একটি স্বাধীন সার্বভৌম রাষ্ট্র।
এই দেশের রাজধানী ঢাকা এবং এখানে বাংলা ভাষায় কথা বলা হয়।
```

---

## 🙏 Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy/nanoGPT) — nanoGPT tutorial
- [Bangla Wikipedia](https://bn.wikipedia.org) — training data
- [Tiny Shakespeare](https://github.com/karpathy/char-rnn) — English training data