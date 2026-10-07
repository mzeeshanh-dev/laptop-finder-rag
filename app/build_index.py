"""Optional: pre-build the index (clean -> chunk -> tokenize -> embed) without starting the server.

    python app/build_index.py

The server also does this automatically on startup; embeddings are cached in data/.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import kb  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kb.build()
    print("\nStatus:", kb.status["message"])
    print("Stats: ", kb.stats)
