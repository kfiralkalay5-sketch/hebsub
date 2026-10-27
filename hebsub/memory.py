"""Per-series character memory.

Each TV series gets one JSON file under ``memory.dir`` (default ~/hebsub/memory).
It holds every character seen so far with their grammatical gender, confidence,
aliases and the episodes they appeared in, plus how characters address each other.

Before an episode is translated, the characters from memory whose name or alias
appears in that episode seed its bible, so the model does not re-decide a gender
that earlier episodes already settled. Afterwards the episode's findings are merged
back. Entries with ``"locked": true`` (set by hand) are never changed.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from .paths import HOME

RANK = {"unresolved": 0, "low": 1, "medium": 2, "high": 3}
# Bare titles are never treated as an alias of a longer name.
TITLES = {"sir", "general", "dr", "doctor", "mr", "mrs", "ms", "miss", "captain",
          "lord", "lady", "king", "queen", "professor", "father", "mother", "kid"}


_SEASON = re.compile(r"^(season|series|s)[\s._-]*\d+|^specials?$|^עונה", re.I)
_EPISODE = re.compile(r"(?<![a-z\d])s\d{1,2}[\s._-]*e\d{1,3}|(?<!\d)\d{1,2}x\d{2,3}(?!\d)", re.I)


def series_key(video: Path, root_names: list[str]) -> str | None:
    """Series folder name: the path component right after a TV root such as 'TV Shows'.
    Any other layout: the folder above 'Season N', else the show name before the episode
    code of the file name (Show.Name.S01E02.1080p.mkv -> 'Show Name')."""
    parts = Path(video).parts
    for i, part in enumerate(parts[:-2]):
        if part in root_names:
            return parts[i + 1]
    if len(parts) >= 3 and _SEASON.search(parts[-2]):
        return parts[-3]
    m = _EPISODE.search(parts[-1]) if parts else None
    if m:
        name = re.sub(r"[._]+", " ", parts[-1][:m.start()]).strip(" -")
        return name or None
    return None


def _path(cfg, key: str) -> Path:
    base = Path(cfg.get("memory.dir", str(HOME / "memory"))).expanduser()
    safe = re.sub(r"[^\w\-\[\]() .]+", "_", key).strip() or "unknown"
    return base / f"{safe}.json"


def load(cfg, key: str) -> dict:
    p = _path(cfg, key)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"series": key, "characters": [], "relationships": []}


def save(cfg, key: str, mem: dict) -> Path:
    p = _path(cfg, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    mem["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def _names(char: dict) -> list[str]:
    return [n for n in [char.get("name", "")] + list(char.get("aliases", [])) if n]


def _mentioned(name: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE) is not None


def relevant(mem: dict, episode_text: str) -> list[dict]:
    """Memory characters named (by name or alias) somewhere in this episode."""
    return [c for c in mem.get("characters", [])
            if any(_mentioned(n, episode_text) for n in _names(c))]


def _alias_target(name: str, chars: list[dict]) -> dict | None:
    """A longer, existing character whose name contains every word of ``name``."""
    words = name.lower().split()
    if not words or (len(words) == 1 and words[0] in TITLES):
        return None
    hits = [c for c in chars
            if c.get("name", "").lower() != name.lower()
            and set(words) < set(c.get("name", "").lower().split())]
    return hits[0] if len(hits) == 1 else None


def merge(mem: dict, bible: dict, episode: str, log) -> dict:
    """Fold one episode's bible into memory. Returns the updated memory."""
    chars = mem.setdefault("characters", [])
    by_name = {n.lower(): c for c in chars for n in _names(c)}
    for new in bible.get("characters", []) or []:
        name = (new.get("name") or "").strip()
        if not name:
            continue
        cur = by_name.get(name.lower()) or _alias_target(name, chars)
        if cur is None:
            cur = {"name": name, "aliases": [],
                   "grammatical_gender": new.get("grammatical_gender", "unknown"),
                   "confidence": new.get("confidence", "unresolved"), "episodes": []}
            chars.append(cur)
        elif name.lower() not in (n.lower() for n in _names(cur)):
            cur.setdefault("aliases", []).append(name)
        by_name[name.lower()] = cur

        g_new, g_cur = new.get("grammatical_gender"), cur.get("grammatical_gender")
        c_new = RANK.get(new.get("confidence", "unresolved"), 0)
        c_cur = RANK.get(cur.get("confidence", "unresolved"), 0)
        known_new = g_new not in (None, "", "unknown")
        if cur.get("locked") or not known_new:
            pass
        elif g_cur in (None, "", "unknown") or (g_new == g_cur and c_new > c_cur):
            cur["grammatical_gender"] = g_new
            cur["confidence"] = new.get("confidence")
        elif g_new != g_cur and c_new > c_cur:
            # Only stronger evidence may flip a settled gender.
            log(f"memory: {cur['name']} changed {g_cur}/{cur.get('confidence')} -> "
                f"{g_new}/{new.get('confidence')}")
            cur["grammatical_gender"] = g_new
            cur["confidence"] = new.get("confidence")
        elif g_new != g_cur:
            log(f"memory: gender conflict for {cur['name']}: memory={g_cur}/"
                f"{cur.get('confidence')} episode={g_new}/{new.get('confidence')}, keeping memory")
        eps = cur.setdefault("episodes", [])
        if episode not in eps:
            eps.append(episode)

    rels = mem.setdefault("relationships", [])
    seen = {(r.get("from"), r.get("to")) for r in rels}
    for rel in bible.get("relationships", []) or []:
        if (rel.get("from"), rel.get("to")) not in seen:
            rels.append(rel)
            seen.add((rel.get("from"), rel.get("to")))
    return mem
