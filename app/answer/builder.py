"""Answer builder - turns ranked rows / statistics into a response using templates only.

Every answer is a dict the frontend renders:
  text         - short markdown-lite paragraph (**bold**, line breaks, "- " bullets)
  cards        - laptop cards (search results)
  table        - {"columns": [...], "rows": [[...]]} for stats / comparisons
  suggestions  - follow-up questions the user can click
"""
from __future__ import annotations

import re

import pandas as pd

from app import config

UNSUPPORTED = {
    r"battery|battery life": "battery life",
    r"keyboard|backlit": "keyboard details",
    r"webcam|camera": "webcam",
    r"\bports?\b|usb|thunderbolt|hdmi": "ports",
    r"warranty|guarantee": "warranty",
    r"colou?r": "colours",
    r"release|year|newest|latest|202\d|201\d": "release dates",
    r"review|rating|reliab": "reviews or ratings",
    r"refresh rate|\d+ ?hz\b": "refresh rate",
    r"fingerprint|face unlock": "biometrics",
}

DEFAULT_SUGGESTIONS = [
    "Best gaming laptop under €1200 with 16GB RAM",
    "Cheapest Lenovo with SSD",
    "Lightweight laptop for students under 700 euros",
    "Compare Dell XPS 13 vs MacBook Pro",
]


def _eur(v: float) -> str:
    return f"€{v:,.0f}" if v >= 100 else f"€{v:,.2f}"


def _storage(gb: float) -> str:
    return f"{gb / 1024:g}TB" if gb >= 1024 else f"{gb:g}GB"


def card(row: pd.Series, reasons: list[str] | None = None, rank: int | None = None, score: float | None = None) -> dict:
    return {
        "id": int(row["id"]),
        "rank": rank,
        "title": row["title"],
        "brand": row["brand"],
        "category": row["category"],
        "price": round(float(row["price_eur"]), 2),
        "price_text": f"€{row['price_eur']:,.2f}",
        "cpu": row["cpu"],
        "ram": f"{row['ram_gb']:g}GB",
        "storage": row["storage"],
        "gpu": row["gpu"],
        "screen": f'{row["screen_inches"]:g}" {row["resolution_class"]}'
                  + (f" ({row['resolution']})" if row["resolution"] else "")
                  + (" · IPS" if row["ips"] else "") + (" · Touch" if row["touchscreen"] else ""),
        "os": row["os_full"] or "No OS",
        "weight": f"{row['weight_kg']:g} kg",
        "tags": list(row["use_cases"]),
        "reasons": reasons or [],
        "score": None if score is None else round(score, 3),
    }


def _match_reasons(row: pd.Series, applied) -> list[str]:
    return [c.label for c in applied]


def _unsupported_note(raw: str) -> str:
    missing = sorted({label for pat, label in UNSUPPORTED.items() if re.search(pat, raw.lower())})
    if not missing:
        return ""
    return f"\n\nNote: the dataset has no information about {', '.join(missing)}, so I could not filter on that."


def _constraint_text(applied) -> str:
    return ", ".join(c.label for c in applied)


# ======================================================================== search
def search(df: pd.DataFrame, pq, ranked: list[dict], applied, dropped, offset: int = 0) -> dict:
    total = len(ranked)
    if offset:
        return _next_page(df, pq, ranked, applied, offset)
    k = min(pq.top_k, total)
    top = ranked[:k]
    lines = []
    if dropped:
        lines.append(
            "No laptop matched **all** of your requirements, so I relaxed: "
            + ", ".join(f"~~{c.label}~~" for c in dropped) + "."
        )
    if total == 0:
        return {"type": "text", "text": "I couldn't find any laptop in the dataset for that request." + _unsupported_note(pq.raw),
                "cards": [], "suggestions": DEFAULT_SUGGESTIONS}

    criteria = _constraint_text(applied)
    sub = df.loc[[r["id"] for r in ranked]]
    if criteria:
        lines.append(f"Found **{total}** laptop{'s' if total != 1 else ''} matching {criteria}.")
    else:
        lines.append(f"Searched all **{len(df)}** laptops for the closest matches.")

    best = df.loc[top[0]["id"]]
    if pq.sort:
        lines.append(f"Ranked by **{pq.sort[2]}**. The top pick is **{best['title']}** at **{_eur(best['price_eur'])}** "
                     f"({best['ram_gb']:g}GB RAM, {best['cpu_family']}, {best['weight_kg']:g} kg).")
    else:
        basis = "relevance to your question, performance and value"
        if "cheap" in pq.preferences:
            basis += " (favouring lower prices)"
        if "powerful" in pq.preferences:
            basis += " (favouring performance)"
        lines.append(f"Ranked by {basis}. Best match: **{best['title']}** at **{_eur(best['price_eur'])}**.")
    if total > 1:
        lines.append(f"Matching prices range from {_eur(sub['price_eur'].min())} to {_eur(sub['price_eur'].max())} "
                     f"(median {_eur(sub['price_eur'].median())}).")
    if total > k:
        lines.append(f"Showing the top {k} — say **\"show more\"** for the next ones.")
    text = "\n".join(lines) + _unsupported_note(pq.raw)

    reasons = _match_reasons(best, applied)
    cards = [card(df.loc[r["id"]], reasons, rank=i + 1, score=r["score"]) for i, r in enumerate(top)]
    return {"type": "results", "text": text, "cards": cards, "suggestions": _follow_ups(pq, best, total, k)}


def _next_page(df: pd.DataFrame, pq, ranked: list[dict], applied, offset: int) -> dict:
    total = len(ranked)
    if offset >= total:
        return {"type": "text", "cards": [], "text": f"That's all — I've already shown all **{total}** matches. "
                "Want to loosen a filter (e.g. **\"any brand\"** or **\"remove the price limit\"**)?",
                "suggestions": ["Any brand", "Remove the price limit", "Start over"]}
    page = ranked[offset: offset + pq.top_k]
    reasons = _match_reasons(None, applied)
    cards = [card(df.loc[r["id"]], reasons, rank=offset + i + 1, score=r["score"]) for i, r in enumerate(page)]
    end = offset + len(page)
    more = " Say **\"show more\"** for the next ones." if end < total else " That's the last of them."
    return {"type": "results", "cards": cards, "text": f"Here are results **{offset + 1}–{end}** of {total}.{more}",
            "suggestions": ["Show more", "Tell me more about the first one", "Which one is the lightest?"] if end < total
            else ["Compare the first two", "Start over"]}


def _follow_ups(pq, best: pd.Series, total: int, k: int) -> list[str]:
    out = []
    if total > 1 and (not pq.sort or pq.sort[0] != "price_eur"):
        out.append("Show cheaper ones")
    out.append("Tell me more about the first one")
    if k >= 2:
        out.append("Compare the first two")
    if total > k:
        out.append("Show more")
    if not any(c.field == "brand" for c in pq.constraints):
        out.append("Only Lenovo" if best["brand"] != "Lenovo" else "What about Dell?")
    out.append("Which one is the lightest?")
    return out[:5]


# ======================================================================== conversation
USE_CASE_REQUIREMENT = {
    "gaming": "a Gaming-category machine, or a gaming GPU (GTX/RTX/RX) with at least 8GB RAM",
    "video editing": "16GB+ RAM, a dedicated GPU and an i5-class CPU or better",
    "programming": "8GB+ RAM, an SSD and an i5-class CPU or better",
    "office": "an SSD and at least an i3-class CPU",
    "student": "a price of €800 or less, at most 2.2 kg and at least 4GB RAM",
    "travel": "a weight of 1.5 kg or less",
    "machine learning": "16GB+ RAM, an Nvidia (CUDA) GPU and an i5-class CPU or better",
    "workstation": "a workstation GPU (Quadro/FirePro/Radeon Pro) or the Workstation category",
    "budget": "a price of €500 or less",
    "premium": "a price of €1,500 or more",
}


def _summary(r: pd.Series) -> str:
    return (f"**{r['title']}** is a {r['category'].lower()} laptop by {r['brand']} for **€{r['price_eur']:,.2f}**: "
            f"{r['cpu']}, {r['ram_gb']:g}GB RAM, {r['storage']}, {r['gpu']}, "
            f"{r['screen_inches']:g}\" {r['resolution_class']} screen, {r['os_full'] or 'no OS'}, {r['weight_kg']:g} kg.")


def detail(df: pd.DataFrame, r: pd.Series, attribute: str | None, use_case: str | None, question: str,
           position: int | None, total: int | None) -> dict:
    from app.ingest.chunker import USE_CASE_RULES

    q = question.lower()
    yes_no = re.match(r"^(does|do|has|is|can|will|would|are)\b", q) is not None
    where = f" (#{position} of {total} in your results)" if position else ""
    if use_case:
        ok = bool(USE_CASE_RULES[use_case](r))
        need = USE_CASE_REQUIREMENT[use_case]
        if ok:
            text = f"**Yes** — **{r['title']}**{where} meets my rule for {use_case}: {need}."
        else:
            text = (f"**Not really.** For {use_case} I look for {need}. **{r['title']}** has {r['ram_gb']:g}GB RAM, "
                    f"{r['cpu_family']}, {r['gpu']} and costs €{r['price_eur']:,.0f}.")
        text += f"\n{_summary(r)}"
    elif attribute == "price":
        avg = df["price_eur"].mean()
        text = (f"**{r['title']}**{where} costs **€{r['price_eur']:,.2f}** — "
                f"{'below' if r['price_eur'] < avg else 'above'} the dataset average of €{avg:,.0f}.")
    elif attribute == "weight":
        avg = df["weight_kg"].mean()
        light = r["weight_kg"] <= 1.5
        text = (f"{'**Yes**, it' if yes_no and light else '**No**, it' if yes_no else 'It'} weighs **{r['weight_kg']:g} kg** "
                f"({'lighter' if r['weight_kg'] < avg else 'heavier'} than the average {avg:.2f} kg"
                f"{', light enough for daily carrying' if light else ''}).")
    elif attribute == "ram":
        text = f"**{r['title']}** has **{r['ram_gb']:g}GB RAM** (the most common amount in the dataset is 8GB)."
    elif attribute == "storage":
        ssd = "has an SSD" if r["has_ssd"] else "has **no SSD** (HDD only)"
        prefix = ("**Yes** — " if r["has_ssd"] else "**No** — ") if yes_no and "ssd" in q else ""
        text = f"{prefix}Storage is **{r['storage']}**; it {ssd}."
    elif attribute == "gpu":
        kind = "a dedicated GPU" if r["dedicated_gpu"] else "integrated graphics"
        prefix = ("**Yes** — " if r["dedicated_gpu"] else "**No** — ") if yes_no and re.search(r"dedicated|discrete|gpu|graphics card|nvidia", q) else ""
        text = f"{prefix}Graphics: **{r['gpu']}** ({kind})."
    elif attribute == "cpu":
        text = f"Processor: **{r['cpu']}** ({r['cpu_family']}, performance tier {r['cpu_tier']}/9 in my scale)."
    elif attribute == "screen":
        feats = [f'{r["screen_inches"]:g}"', r["resolution_class"], f"({r['resolution']})" if r["resolution"] else ""]
        feats += ["IPS" if r["ips"] else "", "touchscreen" if r["touchscreen"] else "not touch", "Retina" if r["retina"] else ""]
        prefix = ("**Yes** — " if r["touchscreen"] else "**No** — ") if yes_no and "touch" in q else ""
        text = f"{prefix}Display: **{' '.join(f for f in feats if f)}**."
    elif attribute == "os":
        text = f"It comes with **{r['os_full'] or 'no operating system'}**."
    else:
        text = _summary(r) + (f"\nIt's #{position} of {total} in your current results." if position else "")
        if r["use_cases"]:
            text += "\nGood for: " + ", ".join(r["use_cases"]) + "."
    if attribute and not use_case:
        text += f"\n{_summary(r)}"
    text += _unsupported_note(question)
    sugg = [f"Is it good for {t}?" for t in ("gaming", "programming", "students") if t.rstrip("s") not in r["use_cases"]][:1]
    sugg += ["Show similar but cheaper", "Why this one?", "Compare the first two"]
    return {"type": "detail", "text": text, "cards": [card(r, rank=position)], "suggestions": sugg[:4]}


def explain(df: pd.DataFrame, r: pd.Series, pq, ranked: list[dict]) -> dict:
    ids = [x["id"] for x in ranked]
    pos = ids.index(int(r["id"])) + 1 if int(r["id"]) in ids else None
    lines = [f"Here's why **{r['title']}** is {'#' + str(pos) if pos else 'in the list'}"
             f"{' of ' + str(len(ids)) if pos else ''}:"]
    if pq and pq.constraints:
        lines.append("- It satisfies every filter: " + ", ".join(c.label for c in pq.constraints) + ".")
    if pq and pq.sort:
        lines.append(f"- Results are sorted by **{pq.sort[2]}**; its value is {r[pq.sort[0]]:g}.")
    else:
        entry = next((x for x in ranked if x["id"] == int(r["id"])), None)
        if entry:
            lines.append(f"- Score = 0.55 × relevance + 0.25 × performance + 0.20 × value = **{entry['score']:.3f}** "
                         f"(text relevance {entry['relevance']:.2f}).")
    sub = df.loc[ids] if ids else df
    perf_rank = int((sub["perf_score"] > r["perf_score"]).sum()) + 1
    value_rank = int((sub["value_score"] > r["value_score"]).sum()) + 1
    lines.append(f"- Performance: #{perf_rank} of {len(sub)} ({r['cpu_family']}, {r['ram_gb']:g}GB RAM, "
                 f"{'SSD' if r['has_ssd'] else 'HDD'}, {'dedicated' if r['dedicated_gpu'] else 'integrated'} GPU).")
    lines.append(f"- Value for money: #{value_rank} of {len(sub)} at €{r['price_eur']:,.0f}.")
    return {"type": "detail", "text": "\n".join(lines), "cards": [card(r, rank=pos)],
            "suggestions": ["Show similar but cheaper", "Tell me more about it", "Show more"]}


def pick(df: pd.DataFrame, ids: list[int], fld: str, ascending: bool, phrase: str, ranked_ids: list[int]) -> dict:
    sub = df.loc[ids]
    best_id = sorted(ids, key=lambda i: ((1 if ascending else -1) * float(df.at[i, fld]), float(df.at[i, "price_eur"]), i))[0]
    r = df.loc[best_id]
    shown_value = {
        "price_eur": f"€{r['price_eur']:,.2f}", "weight_kg": f"{r['weight_kg']:g} kg", "ram_gb": f"{r['ram_gb']:g}GB RAM",
        "total_storage_gb": r["storage"], "screen_inches": f'{r["screen_inches"]:g}" screen',
        "res_w": r["resolution"], "perf_score": f"{r['cpu_family']}, {r['ram_gb']:g}GB RAM, {r['gpu']}",
        "value_score": f"€{r['price_eur']:,.0f} for {r['cpu_family']} / {r['ram_gb']:g}GB",
    }[fld]
    pos = ranked_ids.index(best_id) + 1 if best_id in ranked_ids else ids.index(best_id) + 1
    text = f"Of the {len(sub)} laptops I just showed, **{r['title']}** (#{pos}) is {phrase} — {shown_value}."
    return {"type": "detail", "text": text, "cards": [card(r, rank=pos)],
            "suggestions": ["Tell me more about it", "Show similar but cheaper", "Why this one?"]}


def reset() -> dict:
    return {"type": "text", "cards": [], "suggestions": DEFAULT_SUGGESTIONS,
            "text": "Okay, starting fresh — I've cleared your filters. What are you looking for?"}


# ======================================================================== stats
def count(full: pd.DataFrame, sub: pd.DataFrame, pq, applied, dropped) -> dict:
    n = len(sub)
    criteria = _constraint_text(applied)
    text = (f"There {'is' if n == 1 else 'are'} **{n}** laptop{'s' if n != 1 else ''} matching {criteria}"
            f" (out of {len(full)} in the dataset)." if criteria else f"The dataset contains **{n}** laptops.")
    table = None
    if n:
        group = "category" if any(c.field == "brand" and c.op == "in" for c in applied) else "brand"
        counts = sub[group].value_counts().head(10)
        table = {"columns": [group.title(), "Laptops", "Share"],
                 "rows": [[k, int(v), f"{v / n:.0%}"] for k, v in counts.items()]}
        text += f"\nBreakdown by {group}:"
    return {"type": "stats", "text": text + _unsupported_note(pq.raw), "table": table,
            "cards": [], "suggestions": DEFAULT_SUGGESTIONS[:3]}


METRIC_LABEL = {"price_eur": "price", "weight_kg": "weight", "ram_gb": "RAM",
                "total_storage_gb": "storage", "screen_inches": "screen size"}


def _fmt_metric(metric: str, v: float) -> str:
    return {"price_eur": _eur, "weight_kg": lambda x: f"{x:.2f} kg", "ram_gb": lambda x: f"{x:.1f}GB",
            "total_storage_gb": _storage, "screen_inches": lambda x: f'{x:.1f}"'}[metric](v)


def aggregate(sub: pd.DataFrame, pq, applied, dropped) -> dict:
    metric = pq.aggregate_metric or "price_eur"
    label = METRIC_LABEL[metric]
    criteria = _constraint_text(applied)
    if sub.empty:
        return {"type": "text", "text": "No laptops match those criteria.", "cards": []}
    s = sub[metric]
    lo_row, hi_row = sub.loc[s.idxmin()], sub.loc[s.idxmax()]
    text = (f"Across **{len(sub)}** laptops{(' matching ' + criteria) if criteria else ''}, the average {label} is "
            f"**{_fmt_metric(metric, s.mean())}** (median {_fmt_metric(metric, s.median())}).\n"
            f"- Lowest: {_fmt_metric(metric, s.min())} — {lo_row['title']}\n"
            f"- Highest: {_fmt_metric(metric, s.max())} — {hi_row['title']}")
    if dropped:
        text = "Some filters matched nothing and were ignored: " + ", ".join(c.label for c in dropped) + ".\n" + text
    group = "category" if re.search(r"categor|type", pq.normalized) else "brand"
    if not any(c.field == group and c.op == "in" and len(c.value) == 1 for c in applied):
        g = sub.groupby(group)[metric].agg(["count", "mean", "min", "max"]).sort_values("count", ascending=False).head(10)
        table = {"columns": [group.title(), "Laptops", f"Avg {label}", "Min", "Max"],
                 "rows": [[k, int(r["count"]), _fmt_metric(metric, r["mean"]), _fmt_metric(metric, r["min"]),
                           _fmt_metric(metric, r["max"])] for k, r in g.iterrows()]}
    else:
        table = None
    return {"type": "stats", "text": text, "table": table, "cards": [],
            "suggestions": ["Average weight of Ultrabook laptops", "Average price by category", "Price range of gaming laptops"]}


LIST_LABEL = {"brand": "Brand", "category": "Category", "os": "Operating system", "gpu": "GPU", "cpu_family": "CPU family"}


def list_values(sub: pd.DataFrame, pq, applied) -> dict:
    fld = pq.list_field or "brand"
    counts = sub[fld].value_counts()
    criteria = _constraint_text(applied)
    shown = counts.head(25)
    text = (f"There are **{len(counts)}** distinct {LIST_LABEL[fld].lower()} values"
            f"{(' among laptops matching ' + criteria) if criteria else ' in the dataset'}.")
    rows = []
    for k, v in shown.items():
        avg = sub.loc[sub[fld] == k, "price_eur"].mean()
        rows.append([k, int(v), _eur(avg)])
    return {"type": "stats", "text": text, "cards": [],
            "table": {"columns": [LIST_LABEL[fld], "Laptops", "Avg price"], "rows": rows},
            "suggestions": DEFAULT_SUGGESTIONS[:3]}


def overview(df: pd.DataFrame) -> dict:
    text = (f"The knowledge base holds **{len(df)}** laptops from **{df['brand'].nunique()}** brands across "
            f"**{df['category'].nunique()}** categories.\n"
            f"- Prices: {_eur(df['price_eur'].min())} to {_eur(df['price_eur'].max())} (median {_eur(df['price_eur'].median())})\n"
            f"- RAM: {df['ram_gb'].min():g}GB to {df['ram_gb'].max():g}GB (most common {df['ram_gb'].mode()[0]:g}GB)\n"
            f"- Weight: {df['weight_kg'].min():g} kg to {df['weight_kg'].max():g} kg\n"
            f"- {df['has_ssd'].mean():.0%} have an SSD, {df['dedicated_gpu'].mean():.0%} have a dedicated GPU")
    g = df.groupby("category")["price_eur"].agg(["count", "median"]).sort_values("count", ascending=False)
    table = {"columns": ["Category", "Laptops", "Median price"], "rows": [[k, int(r["count"]), _eur(r["median"])] for k, r in g.iterrows()]}
    return {"type": "stats", "text": text, "table": table, "cards": [], "suggestions": DEFAULT_SUGGESTIONS}


# ======================================================================== compare
def compare(sides: list[dict], shared) -> dict:
    if all(s["kind"] == "model" for s in sides):
        return _compare_models(sides)
    return _compare_groups(sides, shared)


def _compare_models(sides: list[dict]) -> dict:
    rows = [s["row"] for s in sides]
    # Many configurations share a model name - add the price so the two sides are distinguishable
    titles_raw = [r["title"] for r in rows]
    names = [f"{t} (€{r['price_eur']:,.0f})" if titles_raw.count(t) > 1 else t for t, r in zip(titles_raw, rows)]
    spec = [
        ("Price", lambda r: f"€{r['price_eur']:,.2f}"), ("Category", lambda r: r["category"]),
        ("CPU", lambda r: r["cpu"]), ("RAM", lambda r: f"{r['ram_gb']:g}GB"), ("Storage", lambda r: r["storage"]),
        ("GPU", lambda r: r["gpu"]), ("Screen", lambda r: f'{r["screen_inches"]:g}" {r["resolution"]}'),
        ("OS", lambda r: r["os_full"] or "No OS"), ("Weight", lambda r: f"{r['weight_kg']:g} kg"),
    ]
    table = {"columns": ["Spec"] + names, "rows": [[name] + [f(r) for r in rows] for name, f in spec]}

    def winner(field, low=True):
        vals = [r[field] for r in rows]
        best = min(vals) if low else max(vals)
        if vals.count(best) == len(vals):
            return None
        return names[vals.index(best)]

    verdict = []
    for label, field, low in [("Cheaper", "price_eur", True), ("More RAM", "ram_gb", False),
                              ("Lighter", "weight_kg", True), ("Faster CPU tier", "cpu_tier", False),
                              ("More storage", "total_storage_gb", False), ("Higher resolution", "res_w", False)]:
        if w := winner(field, low):
            verdict.append(f"- {label}: **{w}**")
    order = range(len(rows))
    perf = names[max(order, key=lambda i: (rows[i]["perf_score"], -rows[i]["price_eur"]))]
    value = names[max(order, key=lambda i: (rows[i]["value_score"], -rows[i]["price_eur"]))]
    titles = " vs ".join("**" + n + "**" for n in names)
    picked = "" if all(s.get("ref") for s in sides) else " (best matching configuration for each)"
    text = (f"Comparing {titles}{picked}:\n" + "\n".join(verdict)
            + f"\n\nOverall stronger specs: **{perf}**. Better value for money: **{value}**.")
    return {"type": "compare", "text": text, "table": table, "cards": [card(r) for r in rows], "suggestions": DEFAULT_SUGGESTIONS[:3]}


def _compare_groups(sides: list[dict], shared) -> dict:
    def stats(s):
        if s["kind"] == "model":
            sub = pd.DataFrame([s["row"]])
        else:
            sub = s["rows"]
        cheapest = sub.loc[sub["price_eur"].idxmin()]
        return {
            "Laptops": str(len(sub)),
            "Avg price": _eur(sub["price_eur"].mean()),
            "Median price": _eur(sub["price_eur"].median()),
            "Price range": f"{_eur(sub['price_eur'].min())} – {_eur(sub['price_eur'].max())}",
            "Avg RAM": f"{sub['ram_gb'].mean():.1f}GB",
            "Avg weight": f"{sub['weight_kg'].mean():.2f} kg",
            "With SSD": f"{sub['has_ssd'].mean():.0%}",
            "Dedicated GPU": f"{sub['dedicated_gpu'].mean():.0%}",
            "Most common type": sub["category"].mode()[0],
            "Cheapest model": f"{cheapest['title']} ({_eur(cheapest['price_eur'])})",
            "_avg": sub["price_eur"].mean(), "_ram": sub["ram_gb"].mean(), "_w": sub["weight_kg"].mean(),
        }

    st = [stats(s) for s in sides]
    names = [s["label"] for s in sides]
    keys = [k for k in st[0] if not k.startswith("_")]
    table = {"columns": ["Metric"] + names, "rows": [[k] + [x[k] for x in st] for k in keys]}
    pick = lambda key, low: names[min(range(len(st)), key=lambda i: st[i][key] if low else -st[i][key])]
    scope = f" (filtered to {_constraint_text(shared)})" if shared else ""
    text = (f"Comparing {' vs '.join(f'**{n}**' for n in names)}{scope}:\n"
            f"- Lower average price: **{pick('_avg', True)}**\n"
            f"- More RAM on average: **{pick('_ram', False)}**\n"
            f"- Lighter on average: **{pick('_w', True)}**")
    return {"type": "compare", "text": text, "table": table, "cards": [], "suggestions": DEFAULT_SUGGESTIONS[:3]}


# ======================================================================== small talk
def greeting(stats: dict) -> dict:
    return {"type": "text", "cards": [], "suggestions": DEFAULT_SUGGESTIONS,
            "text": f"Hi! I'm a laptop knowledge base with **{stats.get('laptops', 0)}** laptops. "
                    "Ask me to find, filter, count, average or compare laptops — e.g. by price, RAM, brand, GPU or use case."}


def thanks() -> dict:
    return {"type": "text", "text": "You're welcome! Ask me anything else about the laptops.", "cards": [], "suggestions": DEFAULT_SUGGESTIONS[:2]}


def help_text(stats: dict) -> dict:
    text = ("I answer questions using only the laptop dataset — no generative AI, every answer is computed from the data.\n"
            "- **Find**: \"gaming laptop under €1000 with 16GB RAM and GTX 1060\"\n"
            "- **Sort**: \"cheapest / lightest / most powerful Dell ultrabook\"\n"
            "- **Count**: \"how many HP laptops have an SSD?\"\n"
            "- **Average**: \"average price of 2-in-1 laptops by brand\"\n"
            "- **List**: \"which brands make workstations?\"\n"
            "- **Compare**: \"Dell vs HP\" or \"XPS 13 vs MacBook Air\"\n"
            "Filters understood: price, RAM, SSD/HDD size, screen size, weight, brand (incl. \"not Apple\"), "
            "category, CPU (i3/i5/i7, Ryzen...), GPU (Nvidia, GTX 1050...), OS, touchscreen, IPS, Full HD/4K, and use cases "
            "(gaming, students, programming, video editing, office, travel, machine learning).")
    return {"type": "text", "text": text, "cards": [], "suggestions": DEFAULT_SUGGESTIONS}


def unsupported(raw: str) -> dict | None:
    missing = sorted({label for pat, label in UNSUPPORTED.items() if re.search(pat, raw.lower())})
    if not missing:
        return None
    return {"type": "text", "cards": [], "suggestions": ["Lightest laptops under 1.3 kg", "Laptops with SSD and 8GB RAM under €600"],
            "text": f"The dataset doesn't contain {', '.join(missing)}, so I can't answer that reliably. "
                    "I can filter by price, brand, CPU, RAM, storage, GPU, screen, OS and weight — for example, "
                    "lighter laptops with SSDs and low-power CPUs are usually good for portability."}


def off_topic(raw: str) -> dict:
    return {"type": "text", "cards": [], "suggestions": DEFAULT_SUGGESTIONS,
            "text": "I can only answer questions about the laptops in this knowledge base "
                    "(specs, prices, brands, comparisons). Try one of the suggestions below."}


def not_ready(message: str) -> dict:
    return {"type": "text", "cards": [], "suggestions": [], "text": f"The knowledge base is still loading: {message}"}
