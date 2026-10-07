"""Deterministic ranker + constraint filtering. Same input -> same output, always."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.query.parser import Constraint, ParsedQuery

# Score weights when the user gave no explicit sort order
W_RELEVANCE = 0.55
W_PERFORMANCE = 0.25
W_VALUE = 0.20
W_PREFERENCE = 0.35


def constraint_mask(df: pd.DataFrame, c: Constraint) -> pd.Series:
    col = df[c.field]
    if c.op == "<=":
        return col <= c.value
    if c.op == ">=":
        return col >= c.value
    if c.op == "<":
        return col < c.value
    if c.op == ">":
        return col > c.value
    if c.op == "==":
        return col == c.value
    if c.op == "between":
        lo, hi = c.value
        return (col >= lo) & (col <= hi)
    if c.op == "in":
        return col.isin(c.value)
    if c.op == "not_in":
        return ~col.isin(c.value)
    if c.op == "is":
        return col == c.value
    if c.op == "contains":
        if c.field == "use_cases":
            return col.map(lambda tags: c.value in tags)
        return col.str.lower().str.contains(str(c.value).lower(), regex=False)
    raise ValueError(f"Unknown operator {c.op}")


def filter_with_relaxation(df: pd.DataFrame, constraints: list[Constraint]) -> tuple[np.ndarray, list[Constraint], list[Constraint]]:
    """Apply all hard constraints. If nothing matches, drop the lowest-priority constraint
    one at a time until something does. Returns (ids, applied, dropped)."""
    applied = list(constraints)
    dropped: list[Constraint] = []
    while True:
        mask = pd.Series(True, index=df.index)
        for c in applied:
            mask &= constraint_mask(df, c)
        ids = df.index[mask].to_numpy()
        if len(ids) or not applied:
            return ids, applied, dropped
        # drop the least important constraint (lowest priority, latest-added on ties)
        victim = min(reversed(applied), key=lambda c: c.priority)
        applied.remove(victim)
        dropped.append(victim)


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    return np.ones_like(values) if hi - lo < 1e-9 else (values - lo) / (hi - lo)


def rank(df: pd.DataFrame, ids: np.ndarray, fused: dict[int, float], pq: ParsedQuery) -> list[dict]:
    if len(ids) == 0:
        return []
    sub = df.loc[ids]
    rel = np.array([fused.get(int(i), 0.0) for i in ids], dtype=np.float64)
    rel = rel / rel.max() if rel.max() > 0 else rel
    perf = _minmax(sub["perf_score"].to_numpy(dtype=np.float64))
    value = _minmax(sub["value_score"].to_numpy(dtype=np.float64))
    price = _minmax(sub["price_eur"].to_numpy(dtype=np.float64))

    score = W_RELEVANCE * rel + W_PERFORMANCE * perf + W_VALUE * value
    if "cheap" in pq.preferences:
        score += W_PREFERENCE * (1 - price)
    if "powerful" in pq.preferences:
        score += W_PREFERENCE * perf

    rows = [
        {"id": int(i), "score": float(s), "relevance": float(r), "sort_value": None}
        for i, s, r in zip(ids, score, rel)
    ]
    if pq.sort:
        field, ascending, _ = pq.sort
        for row in rows:
            row["sort_value"] = float(df.at[row["id"], field])
        sign = 1 if ascending else -1
        rows.sort(key=lambda r: (sign * r["sort_value"], -r["score"], r["id"]))
    else:
        rows.sort(key=lambda r: (-round(r["score"], 9), float(df.at[r["id"], "price_eur"]), r["id"]))
    return rows
