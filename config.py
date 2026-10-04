"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- LLM (OpenAI-compatible endpoint: OpenRouter / OpenAI) ---
# Key OpenRouter (sk-or-v1...) tự đi qua https://openrouter.ai/api/v1.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or OPENAI_API_KEY
if not LLM_BASE_URL and LLM_API_KEY.startswith("sk-or-v1"):
    LLM_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm_client():
    """OpenAI-compatible client theo LLM_BASE_URL (OpenRouter/Groq/OpenAI)."""
    from openai import OpenAI
    if LLM_BASE_URL:
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    return OpenAI()


def get_ragas_llm_kwargs() -> dict:
    """Kwargs ChatOpenAI cho RAGAS judge (chỉ truyền base khi có)."""
    kwargs: dict = {
        "model_name": LLM_MODEL,
        "temperature": 0,
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1000")),
    }
    if LLM_API_KEY:
        kwargs["openai_api_key"] = LLM_API_KEY
    if LLM_BASE_URL:
        kwargs["openai_api_base"] = LLM_BASE_URL
    return kwargs


def has_llm_key() -> bool:
    return bool(LLM_API_KEY)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
