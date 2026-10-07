"""Knowledge base: builds the index once, then answers questions end-to-end.

    query -> parse (rules) -> hard filter (+relaxation) -> hybrid retrieval (BGE-M3 + BM25, RRF)
          -> deterministic ranker -> template answer builder
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from app import config, ranker
from app.answer import builder
from app.conversation import Resolver, SessionStore
from app.ingest import chunker, cleaner
from app.query.parser import ParsedQuery, QueryParser
from app.retrieval.bm25 import BM25Index
from app.retrieval.embedder import DenseIndex
from app.retrieval.hybrid import rrf_fuse

log = logging.getLogger(__name__)

# Product-line names worth recognising in questions ("XPS", "ThinkPad", ...)
SERIES = [
    "inspiron", "ideapad", "thinkpad", "probook", "aspire", "elitebook", "latitude", "yoga", "xps", "rog",
    "vostro", "zenbook", "vivobook", "legion", "macbook", "alienware", "tecra", "portege", "zbook", "spectre",
    "satellite", "pavilion", "omen", "precision", "strix", "envy", "swift", "blade", "stream", "surface",
    "smartbook", "travelmate", "stealth", "lapbook", "leopard", "pixelbook", "lifebook", "gram", "predator",
    "dominator", "nitro", "spin", "thinkbook", "chromebook flip", "x1 carbon", "macbook air", "macbook pro",
]
NOT_CODES = {"w10", "i3", "i5", "i7", "i9", "gb", "tb", "fhd", "x360", "4k", "2k", "1tb", "hd", "uhd", "win10"}


class KnowledgeBase:
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.chunks: list[dict] = []
        self.bm25: BM25Index | None = None
        self.dense = DenseIndex()
        self.parser: QueryParser | None = None
        self.resolver: Resolver | None = None
        self.sessions = SessionStore()
        self.status = {"stage": "starting", "ready": False, "dense": False, "message": "Starting up..."}
        self.stats: dict = {}
        self._lock = threading.Lock()

    # ================================================================ build
    def build(self) -> None:
        t0 = time.time()
        self._set("cleaning", "Cleaning the raw CSV...")
        df = cleaner.clean()
        df["solid_gb"] = df["ssd_gb"] + df["flash_gb"]
        df["gpu_norm"] = df["gpu"].str.lower().str.replace(" ", "", regex=False)
        df["perf_score"] = (
            0.40 * df["cpu_tier"] / 10
            + 0.25 * df["ram_gb"].clip(upper=32) / 32
            + 0.15 * df["has_ssd"].astype(float)
            + 0.12 * df["dedicated_gpu"].astype(float)
            + 0.08 * (df["gaming_gpu"] | df["workstation_gpu"]).astype(float)
        )
        df["value_score"] = df["perf_score"] / np.sqrt(df["price_eur"].clip(lower=100) / 1000)
        config.DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(config.CLEAN_CSV, index=False)

        self._set("chunking", "Building spec-card chunks...")
        chunks = chunker.build_chunks(df)
        df["use_cases"] = [c["use_cases"] for c in chunks]

        self._set("bm25", "Building BM25 keyword index...")
        texts = [c["text"] for c in chunks]
        self.bm25 = BM25Index(texts)
        self.df, self.chunks = df, chunks
        df["text"] = texts
        self.parser = self._make_parser(df)
        self.resolver = Resolver(df, self.parser.brand_aliases)
        self.stats = {"laptops": len(df), "duplicates_removed": int(df.attrs.get("duplicates_removed", 0)), "chunks": len(chunks)}
        # Keyword-only mode is usable right away while the embedding model loads.
        self.status.update(ready=True)

        self._set("embedding", f"Loading {config.EMBED_MODEL} and embedding chunks (first run downloads the model)...")
        if not config.ENABLE_DENSE:
            log.info("ENABLE_DENSE=false - running BM25-only, no embedding model loaded.")
            self.status["dense_error"] = "disabled with ENABLE_DENSE=false"
        else:
            try:
                self.dense.load_model()
                for c in chunks:
                    c["n_tokens"] = len(self.dense.tokenizer.encode(c["text"], add_special_tokens=True))
                self.dense.build(texts)
                self.status["dense"] = True
                tok = [c["n_tokens"] for c in chunks]
                self.stats.update(tokens_avg=round(float(np.mean(tok)), 1), tokens_max=int(np.max(tok)))
            except Exception as exc:  # noqa: BLE001 - fall back to BM25-only rather than failing
                log.exception("Dense index unavailable, running BM25-only: %s", exc)
                self.status["dense_error"] = str(exc)
        chunker.save_chunks(chunks)
        self.stats["build_seconds"] = round(time.time() - t0, 1)
        mode = "hybrid (BGE-M3 + BM25)" if self.status["dense"] else "keyword-only (BM25)"
        self._set("ready", f"Ready - {len(df)} laptops indexed, {mode} retrieval.")

    def _set(self, stage: str, message: str) -> None:
        self.status.update(stage=stage, message=message)
        log.info(message)

    @staticmethod
    def _make_parser(df: pd.DataFrame) -> QueryParser:
        models = " ".join(df["model"].str.lower())
        series = [s for s in SERIES if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", models)]
        series.sort(key=len, reverse=True)
        hw_tokens = set(re.findall(r"[a-z0-9]+", " ".join(df["cpu"].str.lower()) + " " + " ".join(df["gpu"].str.lower())))
        codes = {
            t for t in re.findall(r"[a-z0-9]+", models)
            if re.search(r"[a-z]", t) and re.search(r"\d", t) and len(t) >= 3 and t not in NOT_CODES and t not in hw_tokens
            and not re.fullmatch(r"\d+(gb|tb|g|t)", t)
        }
        return QueryParser(sorted(df["brand"].unique()), series, codes, sorted(set(df["model"].str.lower())))

    # ================================================================ query
    def reset_session(self, session_id: str) -> None:
        self.sessions.reset(session_id)

    def ask(self, query: str, session_id: str = "default") -> dict:
        t0 = time.time()
        if not self.status["ready"]:
            return builder.not_ready(self.status["message"])
        query = (query or "").strip()[:500]
        with self._lock:
            state = self.sessions.get(session_id)
            state.turn += 1
            pq = self.parser.parse(query)
            plan = self.resolver.resolve(pq, state)
            result = self._execute(plan, state, query)
            if plan.note:
                result["text"] = plan.note + "\n" + result["text"]
            executed = plan.pq or pq
            result["context"] = {
                "filters": [c.label for c in state.pq.constraints] if state.pq else [],
                "sort": state.pq.sort[2] if state.pq and state.pq.sort else None,
                "focus": self.df.at[state.focus, "title"] if state.focus is not None else None,
            }
        result["trace"] = {
            **result.get("trace", {}),
            "plan": plan.kind,
            "parsed": executed.to_dict(),
            "retrieval_mode": "hybrid" if self.status["dense"] else "bm25",
            "ms": round((time.time() - t0) * 1000, 1),
        }
        return result

    def _execute(self, plan, state, query: str) -> dict:
        if plan.kind == "reset":
            state.clear()
            return builder.reset()
        if plan.kind == "smalltalk":
            return {
                "greeting": lambda: builder.greeting(self.stats),
                "thanks": builder.thanks,
                "help": lambda: builder.help_text(self.stats),
                "overview": lambda: builder.overview(self.df),
            }[plan.pq.intent]()
        if plan.kind == "page":
            state.offset += state.pq.top_k
            result = builder.search(self.df, state.pq, state.ranked, state.pq.constraints, [], offset=state.offset)
            page = state.ranked[state.offset: state.offset + state.pq.top_k]
            if page:
                state.shown = [r["id"] for r in page]
            return result
        if plan.kind == "why":
            state.focus = plan.refs[0]
            return builder.explain(self.df, self.df.loc[plan.refs[0]], state.pq, state.ranked)
        if plan.kind == "pick":
            fld, asc, phrase = plan.pick
            result = builder.pick(self.df, plan.refs, fld, asc, phrase, [r["id"] for r in state.ranked])
            state.focus = result["cards"][0]["id"]
            return result
        if plan.kind == "detail":
            target = plan.refs[0]
            ids = [r["id"] for r in state.ranked]
            pos = ids.index(target) + 1 if target in ids else None
            state.focus = target
            return builder.detail(self.df, self.df.loc[target], plan.attribute, plan.use_case, query, pos, len(ids) or None)
        if plan.kind == "compare_refs":
            if len(plan.refs) < 2:
                n = len(state.ranked)
                return {"type": "text", "cards": [], "suggestions": ["Compare the first two", "Show more"],
                        "text": f"I need two laptops from the list to compare — your current results have {n} "
                                f"laptop{'s' if n != 1 else ''}. Try e.g. **\"compare 1 and 2\"**."}
            sides = [{"kind": "model", "label": self.df.at[i, "title"], "row": self.df.loc[i], "ref": True} for i in plan.refs]
            return builder.compare(sides, [])

        # plan.kind == "run"
        pq = plan.pq
        if pq.intent == "compare":
            if pq.constraints:
                state.pq = replace(pq, intent="search")  # "how many of them…" now refers to the compared groups
            return self._compare(pq)
        if pq.intent in ("count", "aggregate", "list_values"):
            result = {"count": self._count, "aggregate": self._aggregate, "list_values": self._list_values}[pq.intent](pq)
            state.pq = pq
            return result
        result, ranked, applied = self._search(pq)
        if ranked:
            state.pq = replace(pq, constraints=list(applied))
            state.ranked, state.offset = ranked, 0
            state.shown = [r["id"] for r in ranked[: pq.top_k]]
            state.focus = None
        return result

    # ---------------------------------------------------------------- retrieval
    def _retrieve(self, text: str, ids: np.ndarray) -> tuple[dict[int, float], float, float]:
        bm25 = self.bm25.scores(text)
        dense = self.dense.scores(text) if self.status["dense"] else None
        fused = rrf_fuse(ids, dense, bm25)
        dense_max = float(dense.max()) if dense is not None else -1.0
        return fused, float(bm25.max()), dense_max

    def _search(self, pq: ParsedQuery) -> tuple[dict, list[dict], list]:
        """Returns (answer, full ranked list, constraints actually applied)."""
        if not pq.has_structure and (missing := builder.unsupported(pq.raw)):
            return missing, [], []  # "good battery life": the data has no such column - say so instead of guessing
        ids, applied, dropped = ranker.filter_with_relaxation(self.df, pq.constraints)
        fused, bm25_max, dense_max = self._retrieve(pq.raw, ids)
        if not pq.has_structure and bm25_max <= 0 and dense_max < config.OFF_TOPIC_DENSE_THRESHOLD:
            return builder.unsupported(pq.raw) or builder.off_topic(pq.raw), [], []
        ranked = ranker.rank(self.df, ids, fused, pq)
        result = builder.search(self.df, pq, ranked, applied, dropped)
        result["trace"] = {"candidates": int(len(ids)), "bm25_max": round(bm25_max, 3), "dense_max": round(dense_max, 3)}
        return result, ranked, applied

    def _filtered(self, pq: ParsedQuery) -> tuple[pd.DataFrame, list, list]:
        ids, applied, dropped = ranker.filter_with_relaxation(self.df, pq.constraints)
        return self.df.loc[ids], applied, dropped

    def _count(self, pq: ParsedQuery) -> dict:
        if not pq.constraints:
            return builder.count(self.df, self.df, pq, [], [])
        # For counts an empty answer is the truthful answer - never relax.
        mask = pd.Series(True, index=self.df.index)
        for c in pq.constraints:
            mask &= ranker.constraint_mask(self.df, c)
        return builder.count(self.df, self.df[mask], pq, pq.constraints, [])

    def _aggregate(self, pq: ParsedQuery) -> dict:
        sub, applied, dropped = self._filtered(pq)
        return builder.aggregate(sub, pq, applied, dropped)

    def _list_values(self, pq: ParsedQuery) -> dict:
        sub, applied, dropped = self._filtered(pq)
        return builder.list_values(sub, pq, applied)

    def _compare(self, pq: ParsedQuery) -> dict:
        entity_fields = {"brand", "model", "category", "use_cases"}
        shared = [c for c in pq.constraints if c.field not in entity_fields]
        sides = []
        for text in pq.compare_sides:
            side = self.parser.parse(text)
            entity = [c for c in side.constraints if c.field in entity_fields | {"cpu_family", "gpu_brand", "os"}]
            if not entity:
                continue
            ids, applied, _ = ranker.filter_with_relaxation(self.df, entity + shared)
            if len(ids) == 0:
                continue
            is_model = any(c.field == "model" for c in entity)
            if is_model:
                # pick the configuration whose name/spec text best matches this side (pure relevance)
                fused, _, _ = self._retrieve(text, ids)
                best = min(ids, key=lambda i: (-fused.get(int(i), 0.0), float(self.df.at[i, "price_eur"]), int(i)))
                sides.append({"kind": "model", "label": text, "row": self.df.loc[best]})
            else:
                label = " ".join(str(c.value[0]) if isinstance(c.value, list) else str(c.value).title() for c in entity)
                sides.append({"kind": "group", "label": label, "rows": self.df.loc[ids]})
        if len(sides) < 2:
            pq.intent = "search"
            return self._search(pq)[0]
        return builder.compare(sides, shared)


kb = KnowledgeBase()
