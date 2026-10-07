"""Central configuration. Every tunable lives here so the pipeline stays deterministic."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = ROOT_DIR / "app"
STATIC_DIR = APP_DIR / "static"

RAW_CSV = ROOT_DIR / "laptops.csv"
DATA_DIR = ROOT_DIR / "data"
CLEAN_CSV = DATA_DIR / "laptops_clean.csv"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"
EMBEDDINGS_META = DATA_DIR / "embeddings_meta.json"

# Embedding model - override with env EMBED_MODEL (e.g. BAAI/bge-small-en-v1.5 for small servers).
# ENABLE_DENSE=false runs BM25-only: no model download, ~0.5 GB RAM.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
ENABLE_DENSE = os.getenv("ENABLE_DENSE", "true").strip().lower() not in ("0", "false", "no", "off")
EMBED_MAX_SEQ_LEN = 512      # chunks are ~100 tokens; 512 is plenty and keeps CPU encoding fast
EMBED_BATCH_SIZE = 32

# Retrieval
RRF_K = 60                   # Reciprocal Rank Fusion constant
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# Below this cosine similarity, a query with no structured constraints is treated as off-topic.
# Calibrated on BGE-M3: off-topic questions ("weather", "biryani recipe") score 0.35-0.39,
# real laptop questions score 0.44+.
OFF_TOPIC_DENSE_THRESHOLD = 0.42

# Server (Docker sets HOST=0.0.0.0)
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
