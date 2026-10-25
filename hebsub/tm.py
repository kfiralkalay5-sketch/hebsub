"""Translation memory: how professional Israeli subtitlers translated similar English lines.

Built from the human Hebrew subtitles aligned to English in the library (train/work/*),
holdout and evaluation videos excluded. For each batch, the most similar English lines
(BM25 over words and word pairs) and their human Hebrew are shown to the model as style
and terminology examples ("natural selection" -> "ברירה טבעית", "Let's see" -> "בואי נראה"),
without any training.
"""
from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from .paths import HOME

INDEX = HOME / "tm" / "index.pkl"
_WORD = re.compile(r"[a-z']+")
_STOP = set("a an the to of and or in on at is are was were be been i you he she it we they me him her us them "
            "my your his its our their this that these those do does did not no yes so but if for with as "
            "what who how why when where there here just oh ok okay uh um hey".split())


def _terms(text: str) -> list[str]:
    ws = [w for w in _WORD.findall(re.sub(r"<[^>]+>", " ", text.lower()))]
    content = [w for w in ws if w not in _STOP]
    return content + [f"{a} {b}" for a, b in zip(ws, ws[1:])]


def build(pairs: list[tuple[str, str]]) -> dict:
    docs = [(_terms(en), en, he) for en, he in pairs]
    df = Counter(t for terms, _, _ in docs for t in set(terms))
    post = defaultdict(list)
    for i, (terms, _, _) in enumerate(docs):
        for t, c in Counter(terms).items():
            post[t].append((i, c))
    avg = sum(len(t) for t, _, _ in docs) / max(len(docs), 1)
    return {"n": len(docs), "df": dict(df), "post": dict(post), "len": [len(t) for t, _, _ in docs],
            "avg": avg, "en": [e for _, e, _ in docs], "he": [h for _, _, h in docs]}


_cache: dict | None = None


def load() -> dict | None:
    global _cache
    if _cache is None and INDEX.is_file():
        _cache = pickle.loads(INDEX.read_bytes())
    return _cache


def search(index: dict, text: str, k: int = 3, k1: float = 1.2, b: float = 0.75) -> list[tuple[float, int]]:
    scores: Counter = Counter()
    for t in set(_terms(text)):
        plist = index["post"].get(t)
        if not plist or len(plist) > 20000:  # too common to tell anything
            continue
        idf = math.log(1 + (index["n"] - len(plist) + 0.5) / (len(plist) + 0.5))
        for i, c in plist:
            denom = c + k1 * (1 - b + b * index["len"][i] / index["avg"])
            scores[i] += idf * c * (k1 + 1) / denom
    return [(s, i) for i, s in scores.most_common(k)]


def examples_for(items: list[dict], per_line: int = 1, limit: int = 10, min_score: float = 6.0) -> list[dict]:
    """[{en, he}] - the closest human-translated lines for a batch's cues."""
    index = load()
    if not index:
        return []
    seen, out = set(), []
    scored = []
    for it in items:
        if not it.get("translate", True):
            continue
        for s, i in search(index, it.get("source", ""), k=per_line):
            if s >= min_score and i not in seen:
                seen.add(i)
                scored.append((s, i))
    for s, i in sorted(scored, reverse=True)[:limit]:
        out.append({"en": index["en"][i], "he": index["he"][i]})
    return out
