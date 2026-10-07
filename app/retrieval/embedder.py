"""Dense retrieval with BAAI/bge-m3 (dense vectors, cosine similarity)."""
from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

from app import config

log = logging.getLogger(__name__)


class DenseIndex:
    def __init__(self):
        self.model = None
        self.embeddings: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.embeddings is not None

    @property
    def tokenizer(self):
        return self.model.tokenizer if self.model is not None else None

    def load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s (first run downloads ~2.3GB)...", config.EMBED_MODEL)
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.model.max_seq_length = config.EMBED_MAX_SEQ_LEN

    def build(self, texts: list[str]) -> None:
        """Embed all chunks, reusing the on-disk cache when the chunk texts are unchanged."""
        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
        meta_path, npy_path = config.EMBEDDINGS_META, config.EMBEDDINGS_NPY
        if npy_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("digest") == digest and meta.get("model") == config.EMBED_MODEL:
                self.embeddings = np.load(npy_path)
                log.info("Loaded %d cached embeddings from %s", len(self.embeddings), npy_path)
                return

        log.info("Embedding %d chunks with %s (one-time, cached afterwards)...", len(texts), config.EMBED_MODEL)
        self.embeddings = self.model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(npy_path, self.embeddings)
        meta_path.write_text(json.dumps({"digest": digest, "model": config.EMBED_MODEL, "count": len(texts)}))

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every chunk (vectors are L2-normalized)."""
        return self.embeddings @ self.encode_query(query)
