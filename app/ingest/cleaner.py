"""Stage 1 - clean the raw CSV into typed, normalized columns.

Every rule here is deterministic: regexes and lookup tables, no models.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from app import config

RAW_COLUMNS = {
    "Manufacturer": "brand",
    "Model Name": "model",
    "Category": "category",
    "Screen Size": "screen_size_raw",
    "Screen": "screen",
    "CPU": "cpu",
    "RAM": "ram_raw",
    "Storage": "storage",
    "GPU": "gpu",
    "Operating System": "os_raw",
    "Operating System Version": "os_version",
    "Weight": "weight_raw",
    "Price (Euros)": "price_raw",
}


def _num(pattern: str, text: str, default: float | None = None) -> float | None:
    m = re.search(pattern, text or "", re.IGNORECASE)
    return float(m.group(1)) if m else default


def _to_gb(amount: float, unit: str) -> float:
    return amount * 1024 if unit.upper() == "TB" else amount


def parse_storage(raw: str) -> dict:
    """'128GB SSD +  1TB HDD' -> {'ssd_gb': 128, 'hdd_gb': 1024, ...}"""
    out = {"ssd_gb": 0.0, "hdd_gb": 0.0, "flash_gb": 0.0, "hybrid_gb": 0.0}
    for part in (raw or "").split("+"):
        m = re.search(r"([\d.]+)\s*(GB|TB)\s*(SSD|HDD|Flash Storage|Hybrid)", part, re.IGNORECASE)
        if not m:
            continue
        size = _to_gb(float(m.group(1)), m.group(2))
        kind = m.group(3).lower()
        key = {"ssd": "ssd_gb", "hdd": "hdd_gb", "flash storage": "flash_gb", "hybrid": "hybrid_gb"}[kind]
        out[key] += size
    out["total_storage_gb"] = sum(out.values())
    # SSD-like = solid state (SSD or flash). Hybrid drives are counted as HDD-like.
    out["has_ssd"] = (out["ssd_gb"] + out["flash_gb"]) > 0
    return out


def parse_screen(raw: str) -> dict:
    text = raw or ""
    w, h = 0, 0
    m = re.search(r"(\d{3,4})x(\d{3,4})", text)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    if w >= 3840:
        label = "4K Ultra HD"
    elif w >= 2560:
        label = "QHD+"
    elif w >= 1920:
        label = "Full HD"
    elif w >= 1600:
        label = "HD+"
    else:
        label = "HD"
    return {
        "res_w": w,
        "res_h": h,
        "resolution": f"{w}x{h}" if w else "",
        "resolution_class": label,
        "ips": "ips" in text.lower(),
        "touchscreen": "touch" in text.lower(),
        "retina": "retina" in text.lower(),
    }


CPU_FAMILIES = [
    # (regex, family label) - order matters, first match wins
    (r"core i9", "Core i9"),
    (r"core i7", "Core i7"),
    (r"core i5", "Core i5"),
    (r"core i3", "Core i3"),
    (r"core m|core m\d", "Core M"),
    (r"xeon", "Xeon"),
    (r"celeron", "Celeron"),
    (r"pentium", "Pentium"),
    (r"atom", "Atom"),
    (r"ryzen", "Ryzen"),
    (r"a\d+-series|\ba\d{1,2}\b", "AMD A-Series"),
    (r"e-series", "AMD E-Series"),
    (r"\bfx\b", "AMD FX"),
    (r"samsung", "Samsung Cortex"),
]

# Rough, deterministic performance tier used by the ranker (higher = faster).
CPU_TIER = {
    "Core i9": 9, "Xeon": 8, "Core i7": 7, "Ryzen": 6, "Core i5": 5, "AMD FX": 4,
    "Core i3": 3, "AMD A-Series": 3, "Core M": 3, "Pentium": 2, "Celeron": 1,
    "Atom": 1, "AMD E-Series": 1, "Samsung Cortex": 1, "Other": 2,
}


def parse_cpu(raw: str) -> dict:
    text = (raw or "").lower()
    family = "Other"
    for pattern, label in CPU_FAMILIES:
        if re.search(pattern, text):
            family = label
            break
    brand = "Intel" if "intel" in text else "AMD" if "amd" in text else "Samsung" if "samsung" in text else "Other"
    # Intel 'U' suffix (e.g. 7200U) = low-power; 'HQ'/'HK' = high-performance mobile
    model = re.search(r"\b(\d{4}[A-Z]{0,2})\b", raw or "")
    suffix = re.sub(r"\d", "", model.group(1)) if model else ""
    return {
        "cpu_brand": brand,
        "cpu_family": family,
        "cpu_ghz": _num(r"([\d.]+)\s*GHz", raw, 0.0),
        "cpu_tier": CPU_TIER[family] + (1 if suffix in ("HQ", "HK") else 0),
    }


def parse_gpu(raw: str, cpu_brand: str) -> dict:
    text = (raw or "").lower()
    brand = "Nvidia" if "nvidia" in text else "AMD" if "amd" in text else "Intel" if "intel" in text else "ARM" if "arm" in text else "Other"
    # Nvidia is always discrete. AMD is discrete when paired with an Intel CPU,
    # or when it is a Radeon Pro / RX / FirePro part. Otherwise it is an APU iGPU.
    dedicated = brand == "Nvidia" or (
        brand == "AMD" and (cpu_brand == "Intel" or re.search(r"radeon (pro|rx)|firepro|\brx\b", text) is not None)
    )
    gaming_gpu = bool(re.search(r"gtx|rtx|rx 5\d0|radeon rx", text))
    workstation_gpu = bool(re.search(r"quadro|firepro|radeon pro", text))
    return {"gpu_brand": brand, "dedicated_gpu": dedicated, "gaming_gpu": gaming_gpu, "workstation_gpu": workstation_gpu}


def normalize_os(os_raw: str, version: str) -> tuple[str, str]:
    os_name = (os_raw or "").strip()
    if os_name.lower() in ("mac os", "macos"):
        os_name = "macOS"
    version = (version or "").strip()
    if os_name == "macOS" and version == "X":
        version = ""
    full = f"{os_name} {version}".strip()
    return os_name, full


# Some cells contain escaped code points such as 'GTX 960<U+039C>' (Greek capital Mu used as 'M').
_CODEPOINT_RE = re.compile(r"<U\+([0-9A-Fa-f]{4,6})>")
_LOOKALIKES = {"Μ": "M", "μ": "u"}


def _ascii(text: str) -> str:
    """Decode '<U+XXXX>' escapes and fold accents so 'Portégé' is searchable as 'Portege'."""
    text = _CODEPOINT_RE.sub(lambda m: chr(int(m.group(1), 16)), text)
    text = "".join(_LOOKALIKES.get(ch, ch) for ch in text)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def read_raw(raw_csv) -> pd.DataFrame:
    # The source file is not UTF-8 (it contains cp1252 bytes such as 'é'); try both.
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(raw_csv, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {raw_csv}")


def clean(raw_csv=config.RAW_CSV) -> pd.DataFrame:
    df = read_raw(raw_csv)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k.strip(): v for k, v in RAW_COLUMNS.items()})
    for col in df.columns:
        df[col] = df[col].astype(str).map(_ascii).str.replace(r"\s+", " ", regex=True).str.strip()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dupes = before - len(df)

    rows = []
    for i, r in df.iterrows():
        cpu = parse_cpu(r["cpu"])
        os_name, os_full = normalize_os(r["os_raw"], r["os_version"])
        row = {
            "id": i,
            "brand": r["brand"],
            "model": r["model"],
            "title": f"{r['brand']} {r['model']}",
            "category": r["category"],
            "screen_inches": _num(r"([\d.]+)", r["screen_size_raw"], 0.0),
            "screen": r["screen"],
            **parse_screen(r["screen"]),
            "cpu": r["cpu"],
            **cpu,
            "ram_gb": _num(r"(\d+)\s*GB", r["ram_raw"], 0.0),
            "storage": r["storage"].replace(" + ", " + "),
            **parse_storage(r["storage"]),
            "gpu": r["gpu"],
            **parse_gpu(r["gpu"], cpu["cpu_brand"]),
            "os": os_name,
            "os_full": os_full,
            "weight_kg": _num(r"([\d.]+)", r["weight_raw"], 0.0),
            "price_eur": float(r["price_raw"].replace(".", "").replace(",", ".")) if r["price_raw"] else 0.0,
        }
        rows.append(row)

    clean_df = pd.DataFrame(rows)
    clean_df.attrs["duplicates_removed"] = dupes
    return clean_df


if __name__ == "__main__":
    out = clean()
    config.DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(config.CLEAN_CSV, index=False)
    print(f"Cleaned {len(out)} rows ({out.attrs['duplicates_removed']} duplicates removed) -> {config.CLEAN_CSV}")
