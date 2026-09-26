"""Configuration: models, pricing, thresholds."""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# Models to compare during evaluation. Swap freely.
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "llama-3.1-8b-instant")
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "llama-3.3-70b-versatile")

# USD per 1M tokens. VERIFY against current provider pricing before you
# quote these numbers anywhere - they change often.
PRICING = {
    "llama-3.1-8b-instant":    {"in": 0.05, "out": 0.08},
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
    "mock":                    {"in": 0.00, "out": 0.00},
}

# GPT-OSS models reason before answering, and those tokens come out of the same
# budget as the JSON. "low" leaves room for the output; raise it if extraction
# quality drops.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))

CONFIDENCE_REVIEW_THRESHOLD = 0.70
URGENT_DEADLINE_DAYS = 7
PROMPT_VERSION = "v1"


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p = PRICING.get(model, {"in": 0.0, "out": 0.0})
    return round((tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000, 6)
