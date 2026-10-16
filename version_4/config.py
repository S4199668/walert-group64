import os
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 768
RERANK_MODEL = "zeroentropy/zerank-1-small"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

CANDIDATES_PER_METHOD = int(os.getenv("CANDIDATES_PER_METHOD", 30))   # top-N from BM25 and from cosine each
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.5))   # min confidence (0-1) to keep a chunk
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 10))

DATA_DIR = BASE / "data"
SESSION_DIR = BASE / "sessions"
DEFAULT_FOLDER = BASE / "processed_data"
DB_PATH = DATA_DIR / "rag.db"
DATA_DIR.mkdir(exist_ok=True)
SESSION_DIR.mkdir(exist_ok=True)

RERANK_URL = os.getenv("RERANK_URL", "http://127.0.0.1:8002")  