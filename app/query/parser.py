"""Query processing - rule-based structured extraction (no LLM).

Turns a free-text question into:
  * intent       - search / count / aggregate / list_values / compare / overview / greeting / help / thanks
  * constraints  - hard filters (price <= 1000, ram_gb >= 16, brand in [Lenovo], ...)
  * sort         - explicit ordering (cheapest, lightest, most powerful, ...)
  * preferences  - soft biases for the ranker (cheap, powerful)
  * top_k        - how many results the user wants
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app import config

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "a few": 3,
}

LESS = r"(?:under|below|less than|lower than|cheaper than|max(?:imum)?|up ?to|at most|within|no more than|not more than|<=?|upto|till|budget(?: of| is)?)"
MORE = r"(?:over|above|more than|greater than|higher than|at least|min(?:imum)?|>=?|starting (?:at|from)|from)"
EXACT = r"(?:exactly|only|just)"
UNIT_GUARD = r"(?!\s*(?:gb|tb|inch|in\b|\"|”|''|kg|ghz|hz|%|\.\d|-?inch))"
NUM = r"(\d+(?:\.\d+)?)\s*(k\b)?"
CUR = r"(?:eur|euros?|usd|dollars?|bucks|rs|pkr)"

SORT_RULES = [
    (r"cheapest|lowest price|least expensive|most affordable|lowest cost|lowest priced", ("price_eur", True, "lowest price")),
    (r"most expensive|priciest|highest price|costliest|highest priced", ("price_eur", False, "highest price")),
    (r"lightest|least weight|lowest weight|most portable", ("weight_kg", True, "lightest")),
    (r"heaviest", ("weight_kg", False, "heaviest")),
    (r"(most|highest|maximum|largest|biggest|max) (ram|memory)", ("ram_gb", False, "most RAM")),
    (r"(most|largest|biggest|max(imum)?) (storage|ssd|disk|drive|space)", ("total_storage_gb", False, "most storage")),
    (r"(biggest|largest) (screen|display)|biggest laptop|largest laptop", ("screen_inches", False, "largest screen")),
    (r"(smallest) (screen|display)|smallest laptop|most compact", ("screen_inches", True, "smallest screen")),
    (r"highest resolution|sharpest|best (screen|display)", ("res_w", False, "highest resolution")),
    (r"best value|value for money|bang for (the )?buck|best deal|good deal", ("value_score", False, "best value for money")),
    (r"fastest|most powerful|best performance|highest performance|strongest|top performance|best specs|beast|best laptop", ("perf_score", False, "best performance")),
]

USE_CASE_PATTERNS = [
    (r"gaming|games?\b|gamer|play(ing)? games|esports|fps", "gaming"),
    (r"machine learning|deep learning|\bml\b|\bai\b|data science|neural|cuda|tensorflow|pytorch", "machine learning"),
    (r"video edit|photo edit|editing|content creat|graphic design|designer|photoshop|premiere|after effects|youtube|streaming|creative", "video editing"),
    (r"programming|coding|developer|software|code\b|web dev|development|computer science", "programming"),
    (r"\boffice\b|business|productivity|excel|word processing|emails?|work from home|corporate|everyday", "office"),
    (r"student|school|college|university|study|studying|homework|class(es)?\b", "student"),
    (r"travel|commut|on the go|carry around", "travel"),
    (r"\bcad\b|engineering|3d|render|autocad|solidworks|architect", "workstation"),
]

GPU_MODEL_RE = re.compile(r"\b(gtx|rtx|mx|quadro|rx|firepro|radeon pro)\s*-?\s*([a-z]?\d{3,4}[a-z]{0,2})\b")


@dataclass
class Constraint:
    field: str
    op: str            # <=, >=, ==, between, in, not_in, contains, is
    value: object
    label: str
    priority: int = 5  # relaxation order: lower priority gets dropped first when nothing matches


@dataclass
class ParsedQuery:
    raw: str
    normalized: str
    intent: str = "search"
    constraints: list[Constraint] = field(default_factory=list)
    sort: tuple | None = None          # (field, ascending, label)
    preferences: list[str] = field(default_factory=list)
    top_k: int = config.DEFAULT_TOP_K
    explicit_k: bool = False
    aggregate_metric: str | None = None
    list_field: str | None = None
    compare_sides: list[str] = field(default_factory=list)

    @property
    def has_structure(self) -> bool:
        return bool(self.constraints or self.sort or self.preferences)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sort"] = {"field": self.sort[0], "ascending": self.sort[1], "label": self.sort[2]} if self.sort else None
        return d


def _num(value: str, k: str | None) -> float:
    v = float(value)
    return v * 1000 if k else v


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:g}"


class QueryParser:
    def __init__(self, brands: list[str], series: list[str], model_codes: set[str], model_names: list[str] | None = None):
        self.brands = sorted(brands, key=len, reverse=True)
        self.series = series
        self.model_names = model_names or []
        self.model_codes = model_codes
        self.brand_aliases = {b.lower(): b for b in brands}
        self.brand_aliases.update({"macbook": "Apple", "mac": "Apple", "hewlett packard": "HP", "hewlett-packard": "HP"})

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def normalize(q: str) -> str:
        q = q.lower().strip()
        q = q.replace("€", " eur ").replace("$", " usd ").replace("£", " eur ")
        q = re.sub(r"(\d),(\d{3})\b", r"\1\2", q)               # 1,000 -> 1000
        q = re.sub(r"(\d)\s*(gb|tb)\b", r"\1\2", q)              # 16 gb -> 16gb
        q = re.sub(r"(\d+(?:\.\d+)?)\s*[\"”]", r"\1 inch", q)    # 15.6" -> 15.6 inch
        q = re.sub(r"\b(2|two)[\s-]*in[\s-]*(1|one)\b", "2-in-1", q)
        q = re.sub(r"[?!]", " ", q)
        return re.sub(r"\s+", " ", q).strip()

    @staticmethod
    def _comparator(before: str, after: str, default: str) -> str:
        before = before[-28:]
        # Trailing qualifiers must directly follow the value ("16gb ram or less", "€800 max"), so that
        # "8gb ram and under 700" does not attach "under" to the RAM.
        unit = r"^\s*(?:ram |memory |ssd |storage |hdd |eur |euros? |inch(?:es)? |kg )?"
        if re.match(unit + r"(or less|or lower|or under|and below|max(imum)?\b(?!\s*(?:eur\s*)?\d)|or cheaper|at most\b(?!\s*\d))", after[:24]):
            return "<="
        if re.match(unit + r"(\+|or more|or higher|or above|and above|and up|at least|minimum|plus)", after[:24]):
            return ">="
        # Only look back to the previous clause/number: in "under 1500 with 16gb", "under" belongs to 1500, not 16gb
        before = re.split(r"\b(?:with|and|having|has|have|plus|but|or)\b|,|\d", before)[-1]
        if re.search(LESS + r"\s*(?:\w+\s*){0,2}$", before):
            return "<="
        if re.search(MORE + r"\s*(?:\w+\s*){0,2}$", before):
            return ">="
        if re.search(EXACT + r"\s*(?:\w+\s*){0,1}$", before):
            return "=="
        return default

    # ------------------------------------------------------------------ main
    def parse(self, raw: str) -> ParsedQuery:
        q = self.normalize(raw)
        pq = ParsedQuery(raw=raw, normalized=q)
        self._intent(q, pq)
        if pq.intent in ("greeting", "help", "thanks"):
            return pq
        if pq.intent == "compare":
            self._compare_sides(q, pq)
        self._price(q, pq)
        self._memory_storage(q, pq)
        self._screen(q, pq)
        self._weight(q, pq)
        self._brand(q, pq)
        self._model(q, pq)
        self._category(q, pq)
        self._cpu(q, pq)
        self._gpu(q, pq)
        self._os(q, pq)
        self._display(q, pq)
        self._use_cases(q, pq)
        self._sort(q, pq)
        self._top_k(q, pq)
        return pq

    # ------------------------------------------------------------------ intent
    def _intent(self, q: str, pq: ParsedQuery) -> None:
        words = q.split()
        if re.match(r"^(hi+|hello|hey|hiya|salam|assalam|aoa|good (morning|evening|afternoon)|yo)\b", q) and len(words) <= 4:
            pq.intent = "greeting"
        elif re.search(r"^(thanks|thank you|thx|shukriya|great|ok|okay|cool)\b", q) and len(words) <= 4:
            pq.intent = "thanks"
        elif re.search(r"^help$|what can you (do|answer)|how (do|does) (this|it|you) work|example (questions|queries)|what (should|can) i ask", q):
            pq.intent = "help"
        elif re.search(r"\b(overview|summary|statistics|stats|describe|summarize)\b.*\b(dataset|data|database|catalog|inventory|everything)\b|^(dataset|data) (overview|summary|stats)", q):
            pq.intent = "overview"
        elif re.search(r"\b(compare|comparison|vs\.?|versus|difference between|differences between)\b", q):
            pq.intent = "compare"
        elif re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal number\b", q):
            pq.intent = "count"
        elif re.search(r"\b(average|avg|mean|median|typical)\b|\bprice range\b|\brange of prices\b", q):
            pq.intent = "aggregate"
            pq.aggregate_metric = self._metric(q)
        elif m := re.search(r"\b(which|what|list|all|available|show)\b.*\b(brands?|manufacturers?|companies|makers|categories|types|operating systems|os|gpus|graphics cards|processors|cpus)\b", q):
            target = m.group(2)
            if not re.search(r"\b(laptop|laptops|notebook)s?\b.*\b(with|under|for)\b", q):
                pq.intent = "list_values"
                pq.list_field = (
                    "brand" if re.match(r"brand|manufacturer|compan|maker", target)
                    else "category" if re.match(r"categor|type", target)
                    else "os" if re.match(r"operating|os", target)
                    else "gpu" if re.match(r"gpu|graphics", target)
                    else "cpu_family"
                )

    @staticmethod
    def _metric(q: str) -> str:
        if re.search(r"weigh", q):
            return "weight_kg"
        if re.search(r"\bram\b|memory", q):
            return "ram_gb"
        if re.search(r"storage|ssd|disk", q):
            return "total_storage_gb"
        if re.search(r"screen|display|size", q):
            return "screen_inches"
        return "price_eur"

    def _compare_sides(self, q: str, pq: ParsedQuery) -> None:
        body = re.sub(r"^.*?\b(compare|comparison of|comparison between|difference between|differences between)\b", "", q).strip()
        body = re.sub(r"\b(laptops?|in terms of.*|which is better|price and specs)\b", " ", body)
        parts = re.split(r"\s+(?:vs\.?|versus|and|or|with|against|to)\s+|,", body)
        pq.compare_sides = [p.strip(" .") for p in parts if p.strip(" .")][:4]

    # ------------------------------------------------------------------ numeric
    def _price(self, q: str, pq: ParsedQuery) -> None:
        price_ctx = re.search(CUR + r"|price|cost|budget|cheap|expensive|afford|spend|pay", q) is not None
        m = re.search(r"(?:between|from)\s*(?:" + CUR + r"\s*)?" + NUM + r"\s*(?:" + CUR + r"\s*)?(?:and|to|-)\s*(?:" + CUR + r"\s*)?" + NUM + UNIT_GUARD, q) \
            or re.search(NUM + r"\s*(?:" + CUR + r")?\s*-\s*" + NUM + r"\s*" + CUR, q)
        if m:
            lo, hi = _num(m.group(1), m.group(2)), _num(m.group(3), m.group(4))
            if hi >= 100 and (price_ctx or hi >= 200):
                lo, hi = min(lo, hi), max(lo, hi)
                pq.constraints.append(Constraint("price_eur", "between", [lo, hi], f"price €{_fmt(lo)}–€{_fmt(hi)}", 9))
                return
        m = re.search(r"(around|about|approximately|approx|roughly|close to|near|~)\s*(?:" + CUR + r"\s*)?" + NUM + UNIT_GUARD, q)
        if m and (_num(m.group(2), m.group(3)) >= 100):
            v = _num(m.group(2), m.group(3))
            lo, hi = round(v * 0.85), round(v * 1.15)
            pq.constraints.append(Constraint("price_eur", "between", [lo, hi], f"price around €{_fmt(v)} (€{_fmt(lo)}–€{_fmt(hi)})", 9))
            return
        for m in re.finditer(r"(?<![\d.])(?:" + CUR + r"\s*)?" + NUM + r"\s*(" + CUR + r")?" + UNIT_GUARD, q):
            value = _num(m.group(1), m.group(2))
            has_cur = bool(m.group(3)) or bool(re.search(CUR + r"\s*$", q[: m.start(1)]))
            if value < 100 and not has_cur:
                continue
            before, after = q[: m.start()], q[m.end():]
            # "4k display" / "2k screen" are resolutions; "under 2k" is a price
            if m.group(2) and not has_cur and not re.search(r"(" + LESS + "|" + MORE + r"|around|about)\s*$", before):
                continue
            op = self._comparator(before, after, "<=" if has_cur or price_ctx else "")
            if not op:
                continue
            if op == "==":
                op = "<="
            label = f"price {'≤' if op == '<=' else '≥'} €{_fmt(value)}"
            pq.constraints.append(Constraint("price_eur", op, value, label, 9))
            if len([c for c in pq.constraints if c.field == "price_eur"]) >= 2:
                break

    def _memory_storage(self, q: str, pq: ParsedQuery) -> None:
        for m in re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)(gb|tb)\b", q):
            value = float(m.group(1)) * (1024 if m.group(2) == "tb" else 1)
            before, after = q[max(0, m.start() - 30): m.start()], q[m.end(): m.end() + 30]
            kind = None
            if re.match(r"\s*(?:\+\s*)?(?:of\s+)?(?:ddr\d\s*)?(ram|memory)\b", after):
                kind = "ram"
            elif sm := re.match(r"\s*(?:\+\s*)?(?:of\s+)?(?:nvme\s+|m\.2\s+)?(ssd|hdd|storage|flash|emmc|hard|disk|drive|space|rom)", after):
                kind = {"ssd": "ssd", "flash": "ssd", "emmc": "ssd", "hdd": "hdd", "hard": "hdd"}.get(sm.group(1), "storage")
            elif re.search(r"(ram|memory)\W+(?:\w+\W+){0,3}$", before):
                kind = "ram"
            elif sm := re.search(r"(ssd|hdd|storage)\W+(?:\w+\W+){0,3}$", before):
                kind = {"ssd": "ssd", "hdd": "hdd"}.get(sm.group(1), "storage")
            else:
                kind = "storage" if value >= 128 else "ram"
            # "how many have 16GB RAM" means exactly 16GB; "laptop with 16GB RAM" means at least 16GB
            op = self._comparator(before, after, "==" if pq.intent == "count" else ">=")
            sym = {"<=": "≤", ">=": "≥", "==": "="}[op]
            nice = f"{value / 1024:g}TB" if value >= 1024 else f"{value:g}GB"
            if kind == "ram":
                pq.constraints.append(Constraint("ram_gb", op, value, f"RAM {sym} {nice}", 8))
            elif kind == "ssd":
                pq.constraints.append(Constraint("solid_gb", op, value, f"SSD {sym} {nice}", 6))
            elif kind == "hdd":
                pq.constraints.append(Constraint("hdd_gb", op, value, f"HDD {sym} {nice}", 4))
            else:
                pq.constraints.append(Constraint("total_storage_gb", op, value, f"storage {sym} {nice}", 6))
        if not any(c.field in ("solid_gb",) for c in pq.constraints):
            if re.search(r"\b(ssd|nvme|solid state|flash storage)\b", q) and not re.search(r"\b(no|without) ssd\b", q):
                pq.constraints.append(Constraint("has_ssd", "is", True, "has SSD", 5))

    def _screen(self, q: str, pq: ParsedQuery) -> None:
        m = re.search(r"(?<![\d.])(1\d(?:\.\d)?)\s*(?:-?\s*inch(?:es)?|in\b|\s*(?:screen|display)|\")", q)
        if m:
            v = float(m.group(1))
            op = self._comparator(q[: m.start()], q[m.end():], "approx")
            if op == "approx" or op == "==":
                lo, hi = (v, v + 0.99) if v.is_integer() else (v - 0.15, v + 0.15)
                pq.constraints.append(Constraint("screen_inches", "between", [lo, hi], f'{v:g}" screen', 4))
            else:
                pq.constraints.append(Constraint("screen_inches", op, v, f'screen {"≤" if op == "<=" else "≥"} {v:g}"', 4))
        elif re.search(r"\b(large|big|bigger|larger|huge|wide)\s+(screen|display)\b", q):
            pq.constraints.append(Constraint("screen_inches", ">=", 15.6, 'large screen (≥ 15.6")', 3))
        elif re.search(r"\b(small|compact|smaller|tiny|mini)\s+(screen|display|laptop|size)\b", q):
            pq.constraints.append(Constraint("screen_inches", "<=", 13.3, 'compact (≤ 13.3")', 3))

    def _weight(self, q: str, pq: ParsedQuery) -> None:
        m = re.search(r"(\d(?:\.\d+)?)\s*(kg|kgs|kilo(?:gram)?s?)\b", q)
        if m:
            v = float(m.group(1))
            op = self._comparator(q[: m.start()], q[m.end():], "<=")
            if op == "==":
                op = "<="
            pq.constraints.append(Constraint("weight_kg", op, v, f"weight {'≤' if op == '<=' else '≥'} {v:g} kg", 4))
        elif re.search(r"\b(lightweight|light weight|ultra-?light|ultralight|thin and light|light laptop|portable|easy to carry|not heavy|slim)\b", q):
            pq.constraints.append(Constraint("weight_kg", "<=", 1.6, "lightweight (≤ 1.6 kg)", 4))

    # ------------------------------------------------------------------ categorical
    def _brand(self, q: str, pq: ParsedQuery) -> None:
        include, exclude = [], []
        for alias, brand in self.brand_aliases.items():
            for m in re.finditer(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", q):
                if alias == "mac" and re.match(r"\s*os", q[m.end():]):
                    continue  # "mac os" is an operating system, not the brand
                before = q[max(0, m.start() - 40): m.start()]
                negated = re.search(
                    r"\b(not|no|except|excluding|exclude|without|other than|besides|apart from|avoid|anything but)\s+(?:an?\s+)?"
                    r"(?:(?:[\w-]+)(?:\s*,\s*|\s+(?:or|and|nor)\s+))*$",
                    before,
                )
                (exclude if negated else include).append(brand)
        include = sorted(set(include) - set(exclude))
        exclude = sorted(set(exclude))
        if include:
            pq.constraints.append(Constraint("brand", "in", include, "brand: " + " or ".join(include), 10))
        if exclude:
            pq.constraints.append(Constraint("brand", "not_in", exclude, "not " + ", ".join(exclude), 10))

    def _model(self, q: str, pq: ParsedQuery) -> None:
        if pq.intent == "compare":
            return  # compare resolves each side separately
        tokens = set(re.findall(r"[a-z0-9]+", q))
        hits = []
        for s in self.series:
            m = re.search(r"(?<![\w])" + re.escape(s) + r"(?![\w])(?:\s+([a-z]?\d{1,4}[a-z]*))?", q)
            if not m:
                continue
            if any(s in h for h in hits):
                continue  # "macbook air" already covers "macbook"
            # "xps 13" / "yoga 520": keep the number if some model name really contains it
            full = f"{s} {m.group(1)}" if m.group(1) else None
            hits.append(full if full and any(full in name for name in self.model_names) else s)
        codes = [t for t in tokens if t in self.model_codes and not any(t in h for h in hits)]
        for term in hits + codes:
            pq.constraints.append(Constraint("model", "contains", term, f'model contains "{term}"', 7))

    def _category(self, q: str, pq: ParsedQuery) -> None:
        cats = []
        if re.search(r"\bultrabooks?\b", q):
            cats.append("Ultrabook")
        if re.search(r"\bworkstations?\b", q):
            cats.append("Workstation")
        if re.search(r"\bnetbooks?\b", q):
            cats.append("Netbook")
        if re.search(r"2-in-1|\bconvertible|\bdetachable|\btablet|\bflip\b|\bx360\b", q):
            cats.append("2 in 1 Convertible")
        if re.search(r"\b(regular|standard|normal|basic) (notebook|laptop)s?\b|\bnotebook category\b", q):
            cats.append("Notebook")
        if cats:
            pq.constraints.append(Constraint("category", "in", cats, "category: " + " or ".join(cats), 7))

    def _cpu(self, q: str, pq: ParsedQuery) -> None:
        fams = []
        for m in re.finditer(r"\b(?:core\s*)?i([3579])\b", q):
            fams.append((f"Core i{m.group(1)}", m))
        for pat, fam in [(r"\bryzen\b", "Ryzen"), (r"\bceleron\b", "Celeron"), (r"\bpentium\b", "Pentium"),
                         (r"\batom\b", "Atom"), (r"\bxeon\b", "Xeon"), (r"\bcore m\b", "Core M")]:
            if m := re.search(pat, q):
                fams.append((fam, m))
        if fams:
            from app.ingest.cleaner import CPU_TIER

            fam, m = fams[0]
            after, before = q[m.end(): m.end() + 25], q[max(0, m.start() - 25): m.start()]
            if re.match(r"\s*(\+|or (better|above|higher|newer|more)|and (above|up|better|higher))", after) or re.search(r"(at least|minimum|min|better than|or better)\s*(?:an?\s+)?(?:intel\s+)?(?:core\s+)?$", before):
                tier = CPU_TIER[fam]
                pq.constraints.append(Constraint("cpu_tier", ">=", tier, f"CPU {fam} or better", 6))
            else:
                names = sorted({f for f, _ in fams})
                pq.constraints.append(Constraint("cpu_family", "in", names, "CPU: " + " or ".join(names), 6))
        elif re.search(r"\bintel\b(?!\s*(hd|uhd|iris|graphics))", q) and not re.search(r"\bamd\b", q):
            pq.constraints.append(Constraint("cpu_brand", "in", ["Intel"], "Intel CPU", 5))
        elif re.search(r"\bamd\b(?!\s*(gpu|graphics|radeon|card))", q) and not re.search(r"\bintel\b", q):
            pq.constraints.append(Constraint("cpu_brand", "in", ["AMD"], "AMD CPU", 5))
        if m := re.search(r"(\d(?:\.\d+)?)\s*ghz", q):
            v = float(m.group(1))
            op = self._comparator(q[: m.start()], q[m.end():], ">=")
            pq.constraints.append(Constraint("cpu_ghz", op, v, f"CPU {'≥' if op != '<=' else '≤'} {v:g} GHz", 3))

    def _gpu(self, q: str, pq: ParsedQuery) -> None:
        if m := GPU_MODEL_RE.search(q):
            term = f"{m.group(1)} {m.group(2)}"
            pq.constraints.append(Constraint("gpu_norm", "contains", term.replace(" ", ""), f"GPU {term.upper()}", 6))
            return
        if re.search(r"\b(nvidia|geforce)\b", q):
            pq.constraints.append(Constraint("gpu_brand", "in", ["Nvidia"], "Nvidia GPU", 6))
        elif re.search(r"\b(radeon|amd (gpu|graphics|card))\b", q):
            pq.constraints.append(Constraint("gpu_brand", "in", ["AMD"], "AMD GPU", 6))
        if re.search(r"\b(integrated graphics|no (dedicated|discrete) (gpu|graphics)|without (a )?(dedicated )?gpu)\b", q):
            pq.constraints.append(Constraint("dedicated_gpu", "is", False, "integrated graphics", 5))
        elif re.search(r"\b(dedicated|discrete|graphics card|good graphics|gpu)\b", q):
            pq.constraints.append(Constraint("dedicated_gpu", "is", True, "dedicated GPU", 5))

    def _os(self, q: str, pq: ParsedQuery) -> None:
        if re.search(r"\b(no os|without (an? )?(os|operating system|windows)|freedos|no operating system)\b", q):
            pq.constraints.append(Constraint("os", "in", ["No OS"], "no OS preinstalled", 5))
        elif re.search(r"\bwindows 10 s\b", q):
            pq.constraints.append(Constraint("os_full", "in", ["Windows 10 S"], "Windows 10 S", 5))
        elif re.search(r"\bwindows 7\b", q):
            pq.constraints.append(Constraint("os_full", "in", ["Windows 7"], "Windows 7", 5))
        elif re.search(r"\b(windows|win ?10)\b", q):
            pq.constraints.append(Constraint("os", "in", ["Windows"], "Windows", 5))
        elif re.search(r"\b(linux|ubuntu)\b", q):
            pq.constraints.append(Constraint("os", "in", ["Linux"], "Linux", 5))
        elif re.search(r"\b(mac ?os|osx|os x)\b", q):
            pq.constraints.append(Constraint("os", "in", ["macOS"], "macOS", 5))
        elif re.search(r"\b(chrome ?os|chromebooks?)\b", q):
            pq.constraints.append(Constraint("os", "in", ["Chrome OS"], "Chrome OS", 5))
        elif re.search(r"\bandroid\b", q):
            pq.constraints.append(Constraint("os", "in", ["Android"], "Android", 5))

    def _display(self, q: str, pq: ParsedQuery) -> None:
        # "under 2k" is a price, not a 2K display - remove price-style "Nk" before matching resolutions
        q = re.sub(r"(" + LESS + "|" + MORE + r"|around|about)\s*\d+(?:\.\d+)?\s*k\b", " ", q)
        if re.search(r"\btouch ?(screen)?\b", q) and not re.search(r"\b(no|without|non)[ -]touch", q):
            pq.constraints.append(Constraint("touchscreen", "is", True, "touchscreen", 4))
        if re.search(r"\bips\b", q):
            pq.constraints.append(Constraint("ips", "is", True, "IPS panel", 3))
        if re.search(r"\bretina\b", q):
            pq.constraints.append(Constraint("retina", "is", True, "Retina display", 3))
        if re.search(r"\b(4k|uhd|ultra hd|2160p|3840)\b", q):
            pq.constraints.append(Constraint("res_w", ">=", 3840, "4K display", 4))
        elif re.search(r"\b(qhd|quad hd|1440p|2k|2560|high[- ]res(olution)?)\b", q):
            pq.constraints.append(Constraint("res_w", ">=", 2560, "QHD+ display", 4))
        elif re.search(r"\b(full ?hd|fhd|1080p?|1920)\b", q):
            pq.constraints.append(Constraint("res_w", ">=", 1920, "Full HD or better", 4))

    def _use_cases(self, q: str, pq: ParsedQuery) -> None:
        for pattern, tag in USE_CASE_PATTERNS:
            if re.search(pattern, q):
                if tag == "workstation" and any(c.field == "category" and "Workstation" in c.value for c in pq.constraints):
                    continue
                pq.constraints.append(Constraint("use_cases", "contains", tag, f"suitable for {tag}", 7))
        if re.search(r"\b(cheap|budget|affordable|low[- ]cost|inexpensive|economical|low price)\b", q) and not pq.sort:
            pq.preferences.append("cheap")
        if re.search(r"\b(powerful|high[- ]performance|fast|high[- ]end|premium|beefy|top[- ]end|flagship)\b", q):
            pq.preferences.append("powerful")

    def _sort(self, q: str, pq: ParsedQuery) -> None:
        for pattern, spec in SORT_RULES:
            if re.search(r"\b(" + pattern + r")\b", q):
                pq.sort = spec
                if "cheap" in pq.preferences and spec[0] == "price_eur":
                    pq.preferences.remove("cheap")
                return

    def _top_k(self, q: str, pq: ParsedQuery) -> None:
        q2 = q.replace("2-in-1", " ")
        m = re.search(r"\btop\s*(\d+|" + "|".join(NUMBER_WORDS) + r")\b", q2) or re.search(
            r"(?<![\d.])\b(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\s+(?!(?:inch|in|gb|tb|kg|eur|ghz|hz|k)\b)(?:[\w-]+\s+){0,3}?"
            r"(laptops|options|models|results|suggestions|recommendations|choices|machines|notebooks|picks|ones|devices)\b",
            q2,
        )
        if m:
            raw = m.group(1)
            k = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
            if 1 <= k <= config.MAX_TOP_K:
                pq.top_k, pq.explicit_k = k, True
                return
        if pq.sort and re.search(r"\bthe (cheapest|lightest|heaviest|fastest|most|best|biggest|largest|smallest|priciest)\b.*\blaptop\b(?!s)", q):
            pq.top_k = 3
