"""Sparse keyword retrieval with BM25 (Okapi)."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "by", "is", "are", "be",
    "i", "me", "my", "want", "need", "looking", "show", "find", "give", "get", "laptop", "laptops",
    "which", "what", "that", "this", "it", "can", "you", "please", "some", "any", "under", "over",
    "than", "less", "more", "about", "around", "between", "good", "best", "eur", "euro", "euros",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.bm25.corpus_size, dtype=np.float32)
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
