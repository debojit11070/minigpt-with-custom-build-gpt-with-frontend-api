# api/main.py
# FastAPI backend for MiniGPT web app

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import time

from model_loader import generate_text, get_model_info, load_model

# ─────────────────────────────────────────
# LIFESPAN — preload both models on startup
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("Preloading models...")
    for lang in ['english', 'bangla']:
        try:
            load_model(lang)
            print(f"  {lang} model ready ✓")
        except Exception as e:
            print(f"  {lang} model failed: {e}")
    yield
    # shutdown — nothing to clean up

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI(
    title       = "MiniGPT API",
    description = "GPT text generation — English & Bangla",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# allow both Vite and Next.js dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [
        "http://localhost:3000",   # Next.js
        "http://localhost:5173",   # Vite (just in case)
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────
class GenerateRequest(BaseModel):
    language:    str   = Field(
        default     = "english",
        description = "Model to use: 'english' or 'bangla'"
    )
    prompt:      str   = Field(
        default     = "",
        description = "Starting text for generation"
    )
    max_tokens:  int   = Field(
        default = 200,
        ge      = 10,
        le      = 500,
        description = "Number of tokens to generate"
    )
    temperature: float = Field(
        default = 1.0,
        ge      = 0.1,
        le      = 2.0,
        description = "Sampling temperature"
    )
    top_k:       int   = Field(
        default = 40,
        ge      = 1,
        le      = 100,
        description = "Top-k sampling"
    )

class GenerateResponse(BaseModel):
    generated_text:   str
    prompt:           str
    language:         str
    time_taken:       float
    tokens_generated: int

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message"   : "MiniGPT API is running!",
        "endpoints" : {
            "generate" : "POST /generate",
            "models"   : "GET  /models",
            "health"   : "GET  /health",
            "docs"     : "GET  /docs",
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def models():
    """Return info about both available models"""
    return {
        "english" : get_model_info("english"),
        "bangla"  : get_model_info("bangla"),
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate text using the specified model"""

    if req.language not in ["english", "bangla"]:
        raise HTTPException(
            status_code = 400,
            detail      = "language must be 'english' or 'bangla'"
        )

    try:
        start = time.time()

        text = generate_text(
            language    = req.language,
            prompt      = req.prompt,
            max_tokens  = req.max_tokens,
            temperature = req.temperature,
            top_k       = req.top_k,
        )

        elapsed = round(time.time() - start, 2)

        return GenerateResponse(
            generated_text   = text,
            prompt           = req.prompt,
            language         = req.language,
            time_taken       = elapsed,
            tokens_generated = len(text),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))