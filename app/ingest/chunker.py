"""Stage 2 - turn each cleaned laptop row into a retrievable chunk.

One laptop = one chunk. A spec card is ~100 tokens, far below BGE-M3's 8192 limit,
so splitting a laptop across chunks would only hurt retrieval. Each chunk carries:
  * text      - natural-language spec card (embedded + BM25-indexed)
  * metadata  - the structured fields used for hard filtering and ranking
"""
from __future__ import annotations

import json

import pandas as pd

from app import config

# Deterministic use-case rules. Each tag is added to the chunk text (helps semantic and
# keyword search) and stored as metadata (lets the query parser filter on it).
USE_CASE_RULES = {
    "gaming": lambda r: r.category == "Gaming" or (r.gaming_gpu and r.ram_gb >= 8),
    "video editing": lambda r: r.ram_gb >= 16 and r.dedicated_gpu and r.cpu_tier >= 5,
    "programming": lambda r: r.ram_gb >= 8 and r.has_ssd and r.cpu_tier >= 5,
    "office": lambda r: r.cpu_tier >= 3 and r.has_ssd and r.category in ("Notebook", "Ultrabook", "2 in 1 Convertible"),
    "student": lambda r: r.price_eur <= 800 and r.weight_kg <= 2.2 and r.ram_gb >= 4,
    "travel": lambda r: 0 < r.weight_kg <= 1.5,
    "machine learning": lambda r: r.ram_gb >= 16 and r.gpu_brand == "Nvidia" and r.cpu_tier >= 5,
    "workstation": lambda r: r.category == "Workstation" or r.workstation_gpu,
    "budget": lambda r: r.price_eur <= 500,
    "premium": lambda r: r.price_eur >= 1500,
}

USE_CASE_TEXT = {
    "gaming": "good for gaming and playing games",
    "video editing": "good for video editing, photo editing, design and content creation",
    "programming": "good for programming, software development and coding",
    "office": "good for office work, business and everyday productivity",
    "student": "good for students, school and university",
    "travel": "lightweight and portable, good for travel and commuting",
    "machine learning": "good for machine learning, deep learning, AI and data science (Nvidia CUDA GPU)",
    "workstation": "workstation class for engineering, CAD, 3D rendering and machine learning",
    "budget": "budget friendly, cheap and affordable",
    "premium": "premium high-end flagship",
}


def use_cases(row) -> list[str]:
    return [tag for tag, rule in USE_CASE_RULES.items() if rule(row)]


def _fmt_storage(gb: float) -> str:
    return f"{gb / 1024:g}TB" if gb >= 1024 else f"{gb:g}GB"


def spec_card(row, tags: list[str]) -> str:
    screen_bits = [f'{row.screen_inches:g}" {row.resolution_class}']
    if row.resolution:
        screen_bits.append(f"({row.resolution})")
    if row.ips:
        screen_bits.append("IPS panel")
    if row.touchscreen:
        screen_bits.append("touchscreen")
    if row.retina:
        screen_bits.append("Retina display")
    gpu_kind = "dedicated graphics" if row.dedicated_gpu else "integrated graphics"
    storage_kind = "SSD" if row.has_ssd else "HDD"
    lines = [
        f"{row.title} - {row.category} laptop by {row.brand}.",
        f"Price: {row.price_eur:.2f} EUR.",
        f"Processor: {row.cpu} ({row.cpu_family}).",
        f"Memory: {row.ram_gb:g}GB RAM.",
        f"Storage: {row.storage} (total {_fmt_storage(row.total_storage_gb)}, {storage_kind}).",
        f"Graphics: {row.gpu} ({gpu_kind}).",
        f"Display: {' '.join(screen_bits)}.",
        f"Operating system: {row.os_full or 'none'}.",
        f"Weight: {row.weight_kg:g} kg.",
    ]
    if tags:
        lines.append("Suitable for: " + "; ".join(USE_CASE_TEXT[t] for t in tags) + ".")
    return " ".join(lines)


def build_chunks(df: pd.DataFrame, tokenizer=None) -> list[dict]:
    chunks = []
    for row in df.itertuples(index=False):
        tags = use_cases(row)
        text = spec_card(row, tags)
        chunk = {
            "chunk_id": f"laptop-{row.id}",
            "laptop_id": int(row.id),
            "text": text,
            "use_cases": tags,
        }
        if tokenizer is not None:
            chunk["n_tokens"] = len(tokenizer.encode(text, add_special_tokens=True))
        chunks.append(chunk)
    return chunks


def save_chunks(chunks: list[dict], path=config.CHUNKS_JSONL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path=config.CHUNKS_JSONL) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
