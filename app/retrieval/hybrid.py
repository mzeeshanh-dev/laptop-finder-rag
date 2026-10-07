"""Hybrid retrieval: dense (BGE-M3) + sparse (BM25) fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import numpy as np

from app import config


def _ranks(scores: np.ndarray, candidates: np.ndarray) -> dict[int, int]:
    """1-based rank of each candidate by descending score; ties broken by id for determinism."""
    order = sorted(candidates.tolist(), key=lambda i: (-float(scores[i]), i))
    return {idx: rank for rank, idx in enumerate(order, start=1)}


def rrf_fuse(
    candidates: np.ndarray,
    dense_scores: np.ndarray | None,
    bm25_scores: np.ndarray,
) -> dict[int, float]:
    """Return {laptop_id: fused_score} for every candidate. Higher is better."""
    fused = {int(i): 0.0 for i in candidates}
    if len(candidates) == 0:
        return fused
    if dense_scores is not None:
        for idx, rank in _ranks(dense_scores, candidates).items():
            fused[idx] += config.DENSE_WEIGHT / (config.RRF_K + rank)
    if np.any(bm25_scores[candidates] > 0):
        for idx, rank in _ranks(bm25_scores, candidates).items():
            fused[idx] += config.BM25_WEIGHT / (config.RRF_K + rank)
    return fused
