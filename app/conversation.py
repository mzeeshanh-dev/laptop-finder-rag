"""Multi-turn conversation without an LLM: a rule-based dialogue state tracker.

Each browser session keeps a small state (last query's filters, the ranked results, which laptops were shown,
and which laptop is "in focus"). Every new message is resolved against that state with deterministic rules:

  * reset            "start over", "new search"
  * page             "show more", "next"
  * why              "why this one?", "why the first?"
  * detail           "tell me more about the second one", "how heavy is it?", "is it good for gaming?"
  * compare_refs     "compare the first two", "compare 1 and 3", "compare them"
  * run (refined)    "what about HP?", "under 800", "cheaper ones", "only with SSD", "any brand", "similar but lighter"
  * run (fresh)      anything that reads like a brand-new request
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field, replace

import pandas as pd

from app.query.parser import Constraint, ParsedQuery

SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_SESSIONS = 1000

OPENERS = ["Sure.", "Got it.", "Okay.", "Alright."]

RESET_RE = re.compile(r"^(start over|new search|reset|restart|clear( (it|all|everything|filters))?|forget (it|everything|that)|start again|from scratch)\b")
PAGE_RE = re.compile(r"^(show |give |see |load )?(me )?(more|next|others?|other ones|more results|more options|the rest|next ones|next page)\b|^(any|are there) (more|others?)\b")
WHY_RE = re.compile(r"^why\b|\bwhy (this|that|is it|did you|the|it|#)|\bhow did you (rank|choose|pick)\b|\bexplain (the )?(ranking|choice|pick)\b")
COMPARE_RE = re.compile(r"\b(compare|comparison|vs\.?|versus|difference|differ|which (one )?is better|better one)\b")
SIMILAR_RE = re.compile(r"\b(similar|alternatives?|like (this|that|it)( one)?|something like|same kind)\b")
SHOW_RE = re.compile(r"\b(show|list|see|display|give|find)\b")
DETAIL_RE = re.compile(r"\b(tell me (more )?about|more (details|info|information) (about|on)|details|specs|specifications|describe|what about the|info on)\b")

FOLLOWUP_START_RE = re.compile(
    r"^(and|also|plus|but|what about|how about|ok(ay)?|now|then|instead|only|just|make it|what if|with|without|except|excluding|"
    r"under|below|over|above|less than|more than|between|around|at least|at most|max|min|sorted|sort|order|same|maybe|hmm|"
    r"from|by|in|preferably|ideally|rather|actually)\b"
)
FOLLOWUP_WORD_RE = re.compile(r"\b(these|those|them|they|ones|same|instead|as well|too|also|only|just|still|anymore|either|of them)\b")
SUBJECT_RE = re.compile(r"\b(laptops?|notebooks?|computers?|pcs?|machines?|ultrabooks?|chromebooks?|macbooks?|workstations?|netbooks?)\b")
ADDITIVE_RE = re.compile(r"\b(also|as well|too|or|plus|in addition|and also|both)\b")

ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3,
            "fifth": 4, "5th": 4, "sixth": 5, "6th": 5, "last": -1}
# Ordinals count as references only in reference-like phrasing: "the first one", "compare first and second",
# "about the last" - but not "my first laptop".
ORDINAL_RE = re.compile(
    r"(?:^|(?<=\bthe )|(?<=\band )|(?<=\bvs )|(?<=\bcompare )|(?<=\babout )|(?<=\bwith )|(?<=, )|(?<=\bor ))"
    r"(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|last)\b"
    r"(?=\s+(?:one|ones|option|result|pick|choice|laptop|model)\b|\s*$|\s+(?:and|vs|with|or)\b|\s*,)"
)
NUMBER_REF_RE = re.compile(r"(?:#|number |no\.? ?|option |result |rank |laptop )(\d{1,2})\b")
TOP_N_RE = re.compile(r"\b(?:first|top) (two|three|four|five|2|3|4|5)\b")
BARE_NUMBERS_RE = re.compile(r"^(?:compare )?(\d)(?:\s*(?:,|and|&|vs\.?|with)\s*(\d))+$")
PRONOUN_RE = re.compile(r"\b(it|its|this one|that one|this laptop|that laptop|this model|that model|this|that)\b")
TOP_PICK_RE = re.compile(r"\b(top one|top pick|best one|the best|winner|number one|#1)\b")
WORD_NUM = {"two": 2, "three": 3, "four": 4, "five": 5}

# "which one is lighter?" -> pick among the laptops already shown: (pattern, field, ascending, phrase)
WHICH_RE = re.compile(
    r"^(which|what)( one| laptop| model)?( of (them|these|those|the \w+))?( one)? (is|has|have|offers|gives|weighs|costs|would be)\b"
    r"|^which one\b|^which\b.*\b(them|these|those|shown)\b"
)
PICK_RULES = [
    (r"cheap|less expensive|lowest price|affordable", "price_eur", True, "the cheapest"),
    (r"expensive|pricie|costl", "price_eur", False, "the most expensive"),
    (r"light|portable|weighs less", "weight_kg", True, "the lightest"),
    (r"heav", "weight_kg", False, "the heaviest"),
    (r"\bram\b|memory", "ram_gb", False, "the one with the most RAM"),
    (r"storage|ssd|space|disk", "total_storage_gb", False, "the one with the most storage"),
    (r"bigger screen|biggest screen|largest|bigger display", "screen_inches", False, "the one with the biggest screen"),
    (r"smaller|smallest|compact", "screen_inches", True, "the most compact"),
    (r"value|worth|deal", "value_score", False, "the best value for money"),
    (r"resolution|sharp", "res_w", False, "the one with the sharpest screen"),
    (r"fast|powerful|perform|strong|better|best|gaming|recommend", "perf_score", False, "the most powerful"),
]

# (pattern, field, op, phrase) - relative refinements anchored on a laptop
RELATIVE_RULES = [
    (r"\b(cheaper|less expensive|lower price|more affordable|lower priced|less money|lower budget)\b", "price_eur", "<", "cheaper than"),
    (r"\b(more expensive|pricier|higher price|higher priced|more premium)\b", "price_eur", ">", "pricier than"),
    (r"\b(lighter|less heavy|more portable|weighs less)\b", "weight_kg", "<", "lighter than"),
    (r"\b(heavier|weighs more)\b", "weight_kg", ">", "heavier than"),
    (r"\b(more|higher|bigger|extra) (ram|memory)\b", "ram_gb", ">", "more RAM than"),
    (r"\b(less|lower|smaller) (ram|memory)\b", "ram_gb", "<", "less RAM than"),
    (r"\b(more|bigger|larger|extra) (storage|ssd|space|disk)\b", "total_storage_gb", ">", "more storage than"),
    (r"\b(bigger|larger) (screen|display|one|ones|laptop)\b", "screen_inches", ">", "bigger screen than"),
    (r"\b(smaller) (screen|display|one|ones|laptop)\b|\bmore compact\b", "screen_inches", "<", "smaller screen than"),
    (r"\b(more powerful|faster|better performance|stronger|better specs|higher performance|more power|better ones?|upgrade|beefier)\b", "perf_score", ">", "more powerful than"),
    (r"\b(less powerful|slower|weaker)\b", "perf_score", "<", "less powerful than"),
]

# Removing filters: "any brand", "forget the budget", "remove filter: price ≤ €1,000"
REMOVE_RE = re.compile(r"\b(remove|drop|forget|ignore|clear|delete|without the|no|any|don'?t care about|doesn'?t matter|not important|whatever|skip|regardless of)\b")
FIELD_GROUPS = [
    (r"price|budget|cost|money", ["price_eur"]),
    (r"brand|make|manufacturer|company", ["brand"]),
    (r"\bram\b|memory", ["ram_gb"]),
    (r"storage|ssd|hdd|disk", ["solid_gb", "hdd_gb", "total_storage_gb", "has_ssd"]),
    (r"screen|display|size|resolution|touch", ["screen_inches", "res_w", "touchscreen", "ips", "retina"]),
    (r"weight|heavy", ["weight_kg"]),
    (r"cpu|processor", ["cpu_family", "cpu_tier", "cpu_brand", "cpu_ghz"]),
    (r"gpu|graphics", ["gpu_brand", "gpu_norm", "dedicated_gpu"]),
    (r"\bos\b|operating system", ["os", "os_full"]),
    (r"category|type", ["category"]),
    (r"use case|purpose|usage", ["use_cases"]),
    (r"model|series", ["model"]),
]

FIELD_LABEL = {"price_eur": "price", "weight_kg": "weight", "ram_gb": "RAM", "total_storage_gb": "storage",
               "screen_inches": "screen", "perf_score": "performance"}


@dataclass
class ConversationState:
    pq: ParsedQuery | None = None          # last executed query (constraints actually applied)
    ranked: list[dict] = field(default_factory=list)
    shown: list[int] = field(default_factory=list)
    offset: int = 0
    focus: int | None = None               # laptop currently being talked about
    turn: int = 0
    updated: float = field(default_factory=time.time)

    def clear(self) -> None:
        self.pq, self.ranked, self.shown, self.offset, self.focus = None, [], [], 0, None


@dataclass
class Plan:
    kind: str                              # reset | smalltalk | page | why | detail | compare_refs | pick | run
    pq: ParsedQuery | None = None
    refs: list[int] = field(default_factory=list)
    note: str = ""
    attribute: str | None = None
    use_case: str | None = None
    pick: tuple | None = None              # (field, ascending, phrase)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> ConversationState:
        with self._lock:
            now = time.time()
            if len(self._sessions) > MAX_SESSIONS:
                for sid in sorted(self._sessions, key=lambda s: self._sessions[s].updated)[: len(self._sessions) // 4]:
                    self._sessions.pop(sid, None)
            state = self._sessions.get(session_id)
            if state is None or now - state.updated > SESSION_TTL_SECONDS:
                state = self._sessions[session_id] = ConversationState()
            state.updated = now
            return state

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# ============================================================================ helpers
def _copy_pq(pq: ParsedQuery, **changes) -> ParsedQuery:
    new = replace(pq, constraints=[replace(c) for c in pq.constraints], preferences=list(pq.preferences),
                  compare_sides=list(pq.compare_sides))
    for k, v in changes.items():
        setattr(new, k, v)
    return new


def _brand_constraint(op: str, brands: list[str]) -> Constraint:
    brands = sorted(set(brands))
    if op == "in":
        return Constraint("brand", "in", brands, "brand: " + " or ".join(brands), 10)
    return Constraint("brand", "not_in", brands, "not " + ", ".join(brands), 10)


def _fmt_value(fld: str, v: float) -> str:
    if fld == "price_eur":
        return f"€{v:,.0f}"
    if fld == "weight_kg":
        return f"{v:g} kg"
    if fld in ("ram_gb", "total_storage_gb"):
        return f"{v / 1024:g}TB" if v >= 1024 else f"{v:g}GB"
    if fld == "screen_inches":
        return f'{v:g}"'
    return f"{v:.2f}"


def merge(prev: ParsedQuery, new: ParsedQuery, additive: bool) -> ParsedQuery:
    """Fold a follow-up's constraints into the previous query. New info on a field replaces old info,
    except brands/use-cases when the user says 'also'/'or'."""
    merged = _copy_pq(prev)
    cons = merged.constraints
    for c in new.constraints:
        if c.field == "brand":
            inc = next((x for x in cons if x.field == "brand" and x.op == "in"), None)
            exc = next((x for x in cons if x.field == "brand" and x.op == "not_in"), None)
            if c.op == "in":
                brands = list(c.value) + (list(inc.value) if (inc and additive) else [])
                cons[:] = [x for x in cons if not (x.field == "brand" and x.op == "in")]
                if exc:
                    remaining = [b for b in exc.value if b not in brands]
                    cons.remove(exc)
                    if remaining:
                        cons.append(_brand_constraint("not_in", remaining))
                cons.append(_brand_constraint("in", brands))
            else:
                brands = list(c.value) + (list(exc.value) if exc else [])
                cons[:] = [x for x in cons if not (x.field == "brand" and x.op == "not_in")]
                if inc:
                    remaining = [b for b in inc.value if b not in brands]
                    cons.remove(inc)
                    if remaining:
                        cons.append(_brand_constraint("in", remaining))
                cons.append(_brand_constraint("not_in", brands))
            continue
        if c.field == "use_cases":
            if not additive:
                cons[:] = [x for x in cons if x.field != "use_cases"]
            if not any(x.field == "use_cases" and x.value == c.value for x in cons):
                cons.append(replace(c))
            continue
        cons[:] = [x for x in cons if x.field != c.field]
        cons.append(replace(c))
    if new.sort:
        merged.sort = new.sort
    merged.preferences = sorted(set(prev.preferences) | set(new.preferences))
    if new.explicit_k:
        merged.top_k, merged.explicit_k = new.top_k, True
    merged.raw = f"{prev.raw} {new.raw}".strip()
    merged.normalized = new.normalized
    return merged


def _change_note(prev: list[Constraint], now: list[Constraint], turn: int, extra: str = "") -> str:
    """'Got it. Updated your search: now **brand: HP**; kept RAM ≥ 16GB.' - makes the memory visible."""
    before, after = [c.label for c in prev], [c.label for c in now]
    added = [l for l in after if l not in before]
    removed = [l for l in before if l not in after]
    kept = [l for l in after if l in before]
    parts = []
    if added:
        parts.append("now **" + "**, **".join(added) + "**")
    if removed:
        parts.append("dropped " + ", ".join(f"~~{l}~~" for l in removed))
    if kept:
        parts.append("kept " + ", ".join(kept))
    if not added and not removed:
        if extra:
            return f"{OPENERS[turn % len(OPENERS)]} {extra}"
        return f"{OPENERS[turn % len(OPENERS)]} Same filters as before ({', '.join(kept) or 'none'})."
    return f"{OPENERS[turn % len(OPENERS)]} Updated your search: {'; '.join(parts)}.{(' ' + extra) if extra else ''}"


def short_title(title: str) -> str:
    """Dataset titles are sometimes truncated spec strings: 'HP 15-BS026nv (i5-7200U/8GB/...' -> 'HP 15-BS026nv'."""
    return title.split(" (")[0]


# ============================================================================ resolver
class Resolver:
    def __init__(self, df: pd.DataFrame, brand_aliases: dict[str, str]):
        self.df = df
        self.brand_aliases = brand_aliases

    # ---------------------------------------------------------------- references
    def references(self, q: str, state: ConversationState) -> list[int]:
        shown, ranked_ids = state.shown, [r["id"] for r in state.ranked]
        if not shown:
            return []
        refs: list[int] = []
        if m := TOP_N_RE.search(q):
            n = WORD_NUM.get(m.group(1)) or int(m.group(1))
            refs.extend(shown[:n])
        if m := BARE_NUMBERS_RE.match(q):
            for n in re.findall(r"\d", q):
                if 1 <= int(n) <= len(ranked_ids):
                    refs.append(ranked_ids[int(n) - 1])
        for m in ORDINAL_RE.finditer(q):
            i = ORDINALS[m.group(1)]
            if -len(shown) <= i < len(shown):
                refs.append(shown[i])
        for m in NUMBER_REF_RE.finditer(q):
            n = int(m.group(1))
            if 1 <= n <= len(ranked_ids):
                refs.append(ranked_ids[n - 1])
        if TOP_PICK_RE.search(q):
            refs.append(shown[0])
        for m in re.finditer(r"\bthe ([a-z]+) (?:one|laptop|model)\b", q):
            brand = self.brand_aliases.get(m.group(1))
            match = next((i for i in shown if brand and self.df.at[i, "brand"] == brand), None)
            if match is not None:
                refs.append(match)
        seen, out = set(), []
        for r in refs:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def pronoun_target(self, q: str, state: ConversationState) -> int | None:
        # "laptops that are good for gaming" uses "that" as a relative pronoun, not a reference
        if SUBJECT_RE.search(re.sub(r"\b(this|that) (laptop|model|one)\b", " ", q)):
            return None
        if PRONOUN_RE.search(q):
            if state.focus is not None:
                return state.focus
            if state.shown:
                return state.shown[0]
        return None

    # ---------------------------------------------------------------- follow-up detection
    @staticmethod
    def is_followup(q: str, pq: ParsedQuery, state: ConversationState) -> bool:
        if state.pq is None:
            return False
        if FOLLOWUP_START_RE.match(q) or FOLLOWUP_WORD_RE.search(q):
            return True
        if any(re.search(p, q) for p, *_ in RELATIVE_RULES):
            return True
        if REMOVE_RE.search(q) and any(re.search(p, q) for p, _ in FIELD_GROUPS) and len(q.split()) <= 8:
            return True  # "any brand", "remove the price limit"
        if q.startswith("remove filter:"):
            return True
        words = len(q.split())
        has_subject = SUBJECT_RE.search(q) or any(c.field == "use_cases" for c in pq.constraints)
        return words <= 6 and not has_subject and bool(pq.constraints or pq.sort or pq.preferences)

    @staticmethod
    def removals(q: str, pq: ParsedQuery, prev: ParsedQuery) -> list[Constraint]:
        """Constraints from `prev` that the user asked to drop."""
        if m := re.match(r"^remove filter:\s*(.+)$", q):
            label = m.group(1).strip().lower()
            return [c for c in prev.constraints if c.label.lower() == label]
        if not REMOVE_RE.search(q):
            return []
        new_fields = {c.field for c in pq.constraints}
        out = []
        for pattern, fields in FIELD_GROUPS:
            if re.search(pattern, q) and not (set(fields) & new_fields):
                out.extend(c for c in prev.constraints if c.field in fields)
        if re.search(r"\b(any|no) (filters?|constraints?|limits?)\b|\bremove (all|everything)\b", q):
            out = list(prev.constraints)
        return out

    def relative(self, q: str, pq: ParsedQuery, anchor: int | None) -> list[Constraint]:
        if anchor is None:
            return []
        row = self.df.loc[anchor]
        new_fields = {c.field for c in pq.constraints}
        out = []
        for pattern, fld, op, phrase in RELATIVE_RULES:
            if re.search(pattern, q) and fld not in new_fields:
                v = float(row[fld])
                sym = "<" if op == "<" else ">"
                label = (f"{phrase} {short_title(row['title'])}" if fld == "perf_score"
                         else f"{FIELD_LABEL[fld]} {sym} {_fmt_value(fld, v)} ({phrase} {short_title(row['title'])})")
                out.append(Constraint(fld, op, v, label, 9 if fld == "price_eur" else 6))
        return out

    def similar(self, anchor: int) -> list[Constraint]:
        row = self.df.loc[anchor]
        lo, hi = round(row["price_eur"] * 0.75), round(row["price_eur"] * 1.25)
        return [
            Constraint("price_eur", "between", [lo, hi], f"price €{lo:,}–€{hi:,} (similar to {short_title(row['title'])})", 9),
            Constraint("category", "in", [row["category"]], f"category: {row['category']}", 7),
            # the dataset lists many configurations of the same model - "similar" means a different model
            Constraint("title", "not_in", [row["title"]], f"other models than {short_title(row['title'])}", 1),
        ]

    @staticmethod
    def attribute(q: str) -> str | None:
        rules = [
            (r"how much|price|cost|expensive|cheap|afford", "price"),
            (r"weigh|heavy|weight|\blight\b|portable", "weight"),
            (r"\bram\b|memory", "ram"),
            (r"storage|ssd|hdd|space|disk|drive", "storage"),
            (r"gpu|graphics|video card|nvidia|geforce|radeon", "gpu"),
            (r"cpu|processor|chip|core i|ryzen|intel|amd", "cpu"),
            (r"screen|display|resolution|inch|touch|ips", "screen"),
            (r"\bos\b|operating system|windows|linux|mac ?os|chrome", "os"),
        ]
        for pattern, name in rules:
            if re.search(pattern, q):
                return name
        return None

    # ---------------------------------------------------------------- main
    def resolve(self, pq: ParsedQuery, state: ConversationState) -> Plan:
        q = pq.normalized
        if RESET_RE.match(q):
            return Plan("reset")
        if pq.intent in ("greeting", "thanks", "help", "overview"):
            return Plan("smalltalk", pq)
        if state.pq is None and not state.shown:
            return Plan("run", pq)

        refs = self.references(q, state)
        pronoun = self.pronoun_target(q, state)
        short = len(q.split()) <= 12
        relative_hit = any(re.search(p, q) for p, *_ in RELATIVE_RULES)

        # "show more" (but "show me more powerful ones" is a refinement)
        if state.ranked and PAGE_RE.match(q) and not pq.constraints and not relative_hit:
            return Plan("page")

        # a brand-new comparison: "compare dell vs hp"
        if pq.intent == "compare" and len(pq.compare_sides) >= 2 and not refs and not BARE_NUMBERS_RE.match(q):
            return Plan("run", pq)

        # "why this one?"
        if WHY_RE.search(q) and state.shown:
            target = refs[0] if refs else (pronoun if pronoun is not None else state.shown[0])
            return Plan("why", refs=[target])

        # "which one is the lightest?" -> answer from the laptops already shown
        if state.shown and WHICH_RE.search(q) and all(c.field == "use_cases" for c in pq.constraints):
            for pattern, fld, asc, phrase in PICK_RULES:
                if re.search(pattern, q):
                    return Plan("pick", refs=refs if len(refs) >= 2 else list(state.shown), pick=(fld, asc, phrase))

        # "compare the first two" / "compare them" / "1 and 3"
        if state.shown and (COMPARE_RE.search(q) or BARE_NUMBERS_RE.match(q)):
            if len(refs) >= 2 or BARE_NUMBERS_RE.match(q):
                return Plan("compare_refs", refs=refs[:4])
            if re.search(r"\b(them|these|those|all|both|results)\b", q) or (not pq.constraints and pq.intent != "compare"):
                return Plan("compare_refs", refs=state.shown[: 2 if "both" in q else 3])
            if len(refs) == 1 and pq.intent == "compare":
                pass  # e.g. "compare the first one with the MacBook Air" -> handled as a normal compare

        anchor = refs[0] if refs else pronoun
        # with an explicit reference ("the dell one"), the brand/model words name the reference, not a new filter
        ignorable = {"use_cases", "has_ssd", "touchscreen", "ips", "dedicated_gpu"} | ({"brand", "model"} if refs else set())
        ref_only = all(c.field in ignorable for c in pq.constraints)

        # "similar ones", "similar but cheaper"
        if SIMILAR_RE.search(q) and (anchor is not None or state.focus is not None or state.shown):
            anchor = next(a for a in (anchor, state.focus, state.shown[0] if state.shown else None) if a is not None)
            base = _copy_pq(pq, constraints=self.similar(anchor), sort=None, raw=str(self.df.at[anchor, "text"]))
            merged = merge(base, pq, additive=False)
            for c in self.relative(q, pq, anchor):
                merged.constraints = [x for x in merged.constraints if x.field != c.field] + [c]
            merged.intent = "search"
            note = f"{OPENERS[state.turn % len(OPENERS)]} Looking for laptops similar to **{short_title(self.df.at[anchor, 'title'])}**."
            return Plan("run", merged, note=note)

        # questions about a specific laptop: "tell me more about the second one", "is it good for gaming?"
        target = refs[:1] if refs else ([pronoun] if pronoun is not None else [])
        if target and ref_only and short and not relative_hit and pq.intent == "search" and not PAGE_RE.match(q):
            tag = next((c.value for c in pq.constraints if c.field == "use_cases"), None)
            return Plan("detail", refs=target, attribute=None if tag else self.attribute(q), use_case=tag)

        if not self.is_followup(q, pq, state) and not refs:
            return Plan("run", pq)

        # ------------------------------------------------ refinement of the previous query
        prev = state.pq
        additive = bool(ADDITIVE_RE.search(q))
        removed = self.removals(q, pq, prev)
        base = _copy_pq(prev, constraints=[c for c in prev.constraints if c not in removed])
        merged = merge(base, pq, additive)
        rel_anchor = anchor if anchor is not None else (state.shown[0] if state.shown else None)
        for c in self.relative(q, pq, rel_anchor):
            merged.constraints = [x for x in merged.constraints if x.field != c.field] + [c]
        if relative_hit and rel_anchor is None:
            if re.search(RELATIVE_RULES[0][0], q):
                merged.sort = ("price_eur", True, "lowest price")

        # carry the previous intent: "how many HP laptops?" -> "what about Dell?" stays a count
        if pq.intent == "search" and prev.intent in ("count", "aggregate", "list_values") and not SHOW_RE.search(q):
            merged.intent, merged.aggregate_metric, merged.list_field = prev.intent, prev.aggregate_metric, prev.list_field
        elif pq.intent in ("count", "aggregate", "list_values"):
            merged.intent, merged.aggregate_metric, merged.list_field = pq.intent, pq.aggregate_metric, pq.list_field
        else:
            merged.intent = "search"
        if SHOW_RE.search(q) and re.search(r"\b(them|those|these|all|the laptops)\b", q):
            merged.intent = "search"
        sort_note = f"Now sorted by **{merged.sort[2]}**." if merged.sort and merged.sort != prev.sort else ""
        return Plan("run", merged, note=_change_note(prev.constraints, merged.constraints, state.turn, sort_note))
