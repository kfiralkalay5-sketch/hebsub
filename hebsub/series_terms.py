"""Names and franchise terms the way the series' own professional Hebrew subtitles render them.

The library has human Hebrew for many episodes of the series we translate (the aligned
English/Hebrew pairs the training pipeline built in ~/hebsub/train/work). For every
capitalised English term that recurs in those episodes (names, places, races, titles such as
"Middle-earth", "Halbrand", "Queen Regent") we look for the Hebrew word or phrase that appears
in most of the lines containing it and rarely elsewhere. The translator is then told
"Middle-earth = הארץ התיכונה" instead of inventing its own rendering, and names get the
spelling Israeli viewers of the series already know.

The episode being translated is always excluded, so a test episode never sees its own
reference.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from .paths import HOME

TRAIN_WORK = HOME / "train" / "work"
CACHE = HOME / "series_terms"
_PRE = ("וכש", "ושה", "וה", "וב", "ול", "ומ", "וש", "שה", "שב", "של", "כש", "מה", "לה", "בה",
        "ו", "ה", "ב", "ל", "מ", "ש", "כ")
_SKIP = {"I", "I'm", "I'll", "I've", "I'd", "OK", "Okay", "Mr", "Mrs", "Ms", "Dr", "God", "Oh", "Hey",
         "Mom", "Dad", "Mum", "Sir", "Ma'am", "TV", "Yes", "No"}
_L = "A-Za-zÀ-ÖØ-öø-ÿ"
_EN = re.compile(rf"[{_L}][{_L}'\-]*[{_L}]|[{_L}]|[.!?:;\"]")
# Hebrew words that cannot open or close a term's rendering ("אני אסטריד", "שר המתנות של")
_FUNC = {"אני", "אתה", "את", "הוא", "היא", "זה", "זאת", "הזה", "הזאת", "של", "מה", "לגבי", "תודה", "בשם",
         "אמר", "אמרה", "עם", "על", "אל", "כן", "לא", "גם", "רק", "כל", "יש", "אין", "היה", "הייתה", "שלי",
         "שלך", "שלו", "שלה", "שלנו", "אותו", "אותה", "הנה", "איפה", "בית", "זו", "או", "אבל", "כי", "אם"}
_HE = re.compile(r"[א-ת][א-ת'\"׳״\-]*")


def _series_of(video: str, roots=("TV Shows",)) -> str | None:
    parts = Path(video).parts
    for i, part in enumerate(parts[:-2]):
        if part in roots:
            return parts[i + 1]
    return None


def _source_stems(series: str) -> set[str]:
    """Video stems of the series' aligned human subtitles (the sources of the table)."""
    out = set()
    for st in TRAIN_WORK.glob("*/stat.json"):
        try:
            video = json.loads(st.read_text(encoding="utf-8")).get("video", "")
        except (OSError, ValueError):
            continue
        if _series_of(video) == series and (st.parent / "target.json").is_file():
            out.add(Path(video).stem)
    return out


def _pairs(series: str, exclude_stem: str) -> list[tuple[str, str]]:
    out = []
    for st in TRAIN_WORK.glob("*/stat.json"):
        try:
            video = json.loads(st.read_text(encoding="utf-8")).get("video", "")
        except (OSError, ValueError):
            continue
        d = st.parent
        if _series_of(video) != series or Path(video).stem == exclude_stem or not (d / "target.json").is_file():
            continue
        try:
            he = json.loads((d / "target.json").read_text(encoding="utf-8"))
            cues = json.loads((d / "prep/manifest.json").read_text(encoding="utf-8-sig"))["cues"]
        except (OSError, ValueError, KeyError):
            continue
        for c in cues:
            h = he.get(str(c["index"]))
            if h and c.get("clean_text"):
                out.append((re.sub(r"<[^>]+>|\{[^}]*\}", "", c["clean_text"]).replace("\n", " "),
                            re.sub(r"<[^>]+>|\{[^}]*\}", "", h).replace("\n", " ")))
    return out


def _cands(he: str) -> set[str]:
    toks = _HE.findall(he)
    out = set()
    for n in (1, 2, 3):
        for k in range(len(toks) - n + 1):
            gram = toks[k:k + n]
            out.add(" ".join(gram))
            for p in _PRE:  # the same phrase with a prefix letter (ו/ה/ב/ל...) removed
                if gram[0].startswith(p) and len(gram[0]) - len(p) >= 2:
                    out.add(" ".join([gram[0][len(p):]] + gram[1:]))
    return out


_COMMON: dict | None = None


def _common_hebrew() -> dict:
    global _COMMON
    if _COMMON is None:
        f = HOME / "tm" / "vocab_freq.json"
        _COMMON = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    return _COMMON


def candidates(series: str, exclude_stem: str = "", min_count: int = 3) -> dict[str, dict]:
    """{term: {"hebrew": [top phrases], "examples": [(en, he), ...]}} from the series' human subtitles."""
    pairs = _pairs(series, exclude_stem)
    if len(pairs) < 300:
        return {}
    lower = Counter(w for en, _ in pairs for w in re.findall(r"\b[a-z][a-z'\-]+\b", en))
    seen: Counter = Counter()
    for en, _ in pairs:
        toks = _EN.findall(en)
        for k, t in enumerate(toks):
            # a term is capitalised in mid-sentence (sentence-initial words prove nothing)
            if not t[0].isupper() or t in _SKIP or k == 0 or toks[k - 1] in ".!?:;\"":
                continue
            seen[t] += 1
            phrase, j = [t], k + 1
            while j < len(toks) and toks[j][:1].isupper() and toks[j] not in _SKIP and len(phrase) < 3:
                phrase.append(toks[j])
                seen[" ".join(phrase)] += 1
                j += 1
    terms = [t for t, n in seen.items() if n >= min_count and not t.endswith("'s") and len(t) >= 3
             and not (" " not in t and lower.get(t.lower(), 0) >= 0.3 * n)]  # "King", "Captain"
    he_sets = [_cands(he) for _, he in pairs]
    df = Counter(w for st in he_sets for w in st)
    out = {}
    for term in terms:
        rx = re.compile(r"(?<![^\W\d_])" + re.escape(term) + r"(?![^\W\d_])")
        ls = [i for i, (en, _) in enumerate(pairs) if rx.search(en)]
        if len(ls) < min_count:
            continue
        words = term.split()
        n_en = sum(len(w.split("-")) for w in words)
        c = Counter(w for i in ls for w in he_sets[i])
        scored = []
        for w, n in c.most_common(80):
            hw = w.split()
            cov, spec = n / len(ls), n / df[w]
            if (cov >= 0.4 and spec >= 0.3 and len(hw) <= max(n_en, len(words)) and len(w.replace(" ", "")) >= 2
                    and hw[0] not in _FUNC and hw[-1] not in _FUNC and not re.search(r"[\"״]", w)):
                scored.append((round(cov * spec, 3), w))
        if scored:
            top = [w for _, w in sorted(scored, reverse=True)[:6]]
            ex = [pairs[i] for i in ls[:: max(1, len(ls) // 2)][:2]]
            out[term] = {"hebrew": top, "examples": ex, "count": len(ls)}
    return out


NORMALIZE_PROMPT = """Below are recurring English names and terms from a TV series. For each one you get Hebrew
phrases that appear in the series' professional Hebrew subtitles on the lines that contain it,
and two example line pairs. Give the Hebrew rendering of the term itself, the way it would fill
the blank in "זה ___": drop a prefix letter (ו ב ל מ ש כ, or a ה that only belongs to that one
sentence), but keep the definite article when the name itself carries it (הארץ התיכונה,
המלכה השליטה, הנאמנים). No extra words. Build it from the given phrases. Return "" when none of
the phrases renders the term (a different word, a sentence fragment).
Return JSON {"items":[{"term":"<English>","hebrew":"<Hebrew or empty>"}]} for every term.

"""


def normalize(engine, cands: dict[str, dict], batch: int = 25) -> dict[str, str]:
    from .engines import _extract_json
    table = {}
    terms = sorted(cands)
    schema = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
        "type": "object", "required": ["term", "hebrew"],
        "properties": {"term": {"type": "string"}, "hebrew": {"type": "string"}}}}}}
    for s in range(0, len(terms), batch):
        chunk = [{"term": t, "hebrew_phrases": cands[t]["hebrew"],
                  "examples": [{"en": e, "he": h} for e, h in cands[t]["examples"]]} for t in terms[s:s + batch]]
        try:
            data = _extract_json(engine.complete(NORMALIZE_PROMPT + json.dumps(chunk, ensure_ascii=False, indent=1),
                                                 schema=schema))
        except Exception:  # noqa: BLE001 - a failed chunk only loses its terms
            continue
        for x in (data.get("items", []) if isinstance(data, dict) else []):
            t, h = x.get("term"), (x.get("hebrew") or "").strip()
            if t not in cands or not h or re.search(r"[A-Za-z]", h):
                continue
            # it must really come from the human phrases (no invented rendering)
            pool = " ".join(cands[t]["hebrew"]) + " " + " ".join(e[1] for e in cands[t]["examples"])
            if all(w in pool or any(w == p[len(q):] for p in pool.split() for q in _PRE if p.startswith(q))
                   for w in h.split()):
                table[t] = h
    # a single word that mostly occurs inside a longer listed name ("Tower" in "Iago Tower")
    for term in [t for t in table if " " not in t]:
        longer = sum(cands[t]["count"] for t in table if " " in t and term in t.split())
        if longer >= 0.6 * cands[term]["count"]:
            del table[term]
    return dict(sorted(table.items()))


def for_video(video: Path, engine=None, exclude_self: bool = True, log=None) -> dict[str, str]:
    """Cached per series (and per excluded episode)."""
    series = _series_of(str(video))
    if not series:
        return {}
    stem = Path(video).stem if exclude_self else ""
    if stem and stem not in _source_stems(series):
        # הפרק עצמו לא נמצא בין המקורות (אין לו עברית אנושית), אז הטבלה זהה לכל פרקי
        # הסדרה: מטמון אחד לסדרה במקום כריה מחדש של 3 דקות בכל פרק
        stem = ""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^\w\-]+", "_", f"{series}__{stem}")[:180]
    path = CACHE / f"{key}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    cands = candidates(series, stem)
    if not cands or engine is None:
        return {}
    table = normalize(engine, cands)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    if log:
        log(f"series terms: {len(table)} name(s)/term(s) from the series' own Hebrew subtitles")
    return table


def terms_for(table: dict[str, str], text: str) -> list[str]:
    """'Term = Hebrew' lines for table terms that occur in this text (longest first)."""
    out, used = [], []
    for term in sorted(table, key=len, reverse=True):
        if re.search(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])", text) and \
                not any(term in u for u in used):
            out.append(f"{term} = {table[term]}")
            used.append(term)
    return out
