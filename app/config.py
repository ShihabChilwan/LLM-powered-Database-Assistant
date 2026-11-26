"""
All settings and limits for the app, in one place.

Values that differ per machine (the API key, which model to use) come from
the .env file in the project root. Everything else is a plain constant here,
so no other module contains magic numbers.

Usage from any module:
    from app import config
    config.LLM_MODEL
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- Paths ----------

# Path(__file__) is this file; .parents[1] walks up two levels to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "app" / "database" / "processed.db"

# Load .env from the project root, not from whatever folder the app was started in.
# load_dotenv never overwrites a variable that is already set in the real environment.
load_dotenv(PROJECT_ROOT / ".env")

# ---------- Model provider (OpenRouter, OpenAI-compatible API) ----------

# May be empty here: the LLM client only checks for it when the first call is made,
# so importing this module without a key does not fail.
OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "",
)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# Slug checked against OpenRouter's model list (GET /api/v1/models), which marks it
# as supporting the "tools" parameter.
LLM_MODEL = os.getenv("LLM_MODEL", "z-ai/glm-5.3-flash")

# Checked against OpenRouter's embedding model list (GET /api/v1/embeddings/models).
# The same model must embed both the documents and the questions, otherwise the
# vectors are not comparable.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# ---------- Limits ----------

# Maximum number of model calls in one question-answering loop.
MAX_TURNS = 10

# Maximum number of rows a SQL tool call returns to the model (used from Part A).
MAX_SQL_ROWS = 200

# Number of document chunks the retriever returns (used from Part B3).
TOP_K = 4

# Chunks with a cosine similarity below this are dropped (used from Part B3).
# Starting value only; it will be tuned once real embeddings can be inspected.
MIN_SCORE = 0.3
