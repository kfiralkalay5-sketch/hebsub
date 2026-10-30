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
# A rank or title before a name (Hunter Sung, Chairman Go, King) and its Hebrew, with or without
# ה. The names step transliterated some titles (האנטר סונג, קינג, צ'יף וו) or lost one to a raw
# quote (Chairman Go -> יו), and terms_pass then forced that on every line (Solo Leveling, 160
# lines). Such an entry stays out of the name table and the votes, so the translator renders the
# title itself, in the person's gender; a translated one (Coach Chilton = המאמנת צ'ילטון) stays.
RANK_TITLES = {
    "hunter": "צייד ציידת", "chairman": "יו\"ר יו״ר יושב יושבת", "chairwoman": "יו\"ר יו״ר יושבת",
    "chief": "ראש מפקד מפקדת מפקח מפקחת נגד", "president": "נשיא נשיאה", "director": "מנהל מנהלת במאי במאית",
    "commander": "מפקד מפקדת", "general": "גנרל גנרלית", "colonel": "קולונל", "major": "מייג'ור",
    "admiral": "אדמירל", "marshal": "מרשל", "king": "מלך", "queen": "מלכה", "prince": "נסיך",
    "princess": "נסיכה", "emperor": "קיסר", "empress": "קיסרית", "lord": "לורד אדון", "lady": "ליידי גברת",
    "duke": "דוכס", "duchess": "דוכסית", "sir": "סר", "master": "מאסטר אדון", "judge": "שופט שופטת",
    "coach": "מאמן מאמנת", "principal": "מנהל מנהלת", "professor": "פרופסור פרופ'",
    "senator": "סנטור סנטורית סנאטור סנאטורית", "governor": "מושל מושלת", "mayor": "ראש", "sheriff": "שריף",
    "deputy": "סגן סגנית", "chancellor": "קנצלר קנצלרית", "minister": "שר שרה",
    "inspector": "מפקח מפקחת פקח", "officer": "שוטר שוטרת קצין קצינה סמל נגד סוכן סוכנת", "sergeant": "סמל סמלת סרג'נט",
    "detective": "בלש בלשית", "captain": "קפטן סרן", "lieutenant": "סגן סגנית סרן", "agent": "סוכן סוכנת",
    "nurse": "אחות אח", "doctor": "ד\"ר ד״ר דוקטור רופא רופאה", "uncle": "דוד", "aunt": "דודה",
    "auntie": "דודה", "grandma": "סבתא", "granny": "סבתא", "grandpa": "סבא",
}


def bad_title(en: str, he: str) -> bool:
    """'Hunter Sung' = 'האנטר סונג': the title was transliterated, not translated."""
    words, hw = (en or "").split(), (he or "").split()
    if not words or words[0].rstrip(".").lower() not in RANK_TITLES:
        return False
    ok = RANK_TITLES[words[0].rstrip(".").lower()].split()
    return not hw or (hw[0] not in ok and re.sub(r"^ה", "", hw[0]) not in ok)


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


# ---------------------------------------------------------------- Hebrew spelling of names

_UNFINAL = str.maketrans("ךםןףץ", "כמנפצ")
_HE_NAME = re.compile(r"[א-ת'\"׳״\- .]+")


def clean_hebrew_name(he: str) -> str:
    """Without vowel points; '' when it holds anything but Hebrew letters (a model once
    answered with Bengali letters inside a name)."""
    he = re.sub(r"[֑-ֽֿ-ׇ]", "", he or "").strip()
    he = re.sub(r"(?<=[א-ת])[ךםןףץ](?=[אבגדהוזחטיכלמנסעפצקרשת])", lambda m: m.group(0).translate(_UNFINAL), he)
    return he if _HE_NAME.fullmatch(he) and re.search(r"[א-ת]", he) else ""


def settled_names(mem: dict) -> dict[str, str]:
    """{English: Hebrew} as earlier episodes spelled each name most often (a tie settles nothing).
    A compound ("Matsumoto Rangiku", "Matsumoto-san") takes the settled spelling of each of its
    parts, so it never disagrees with the name alone."""
    out = {}
    for en, votes in (mem.get("names_he") or {}).items():
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]) and not bad_title(en, ranked[0][0]):
            out[en] = ranked[0][0]
    for en, he in list(out.items()):
        en_parts, he_parts = re.split(r"[ -]", en), re.split(r"([ -])", he)
        if len(en_parts) > 1 and len(en_parts) == len(he_parts[::2]):
            he_parts[::2] = [out.get(e, h) if e in out and " " not in out[e] and "-" not in out[e] else h
                             for e, h in zip(en_parts, he_parts[::2])]
            out[en] = "".join(he_parts)
    return out


def vote_names(mem: dict, names: dict[str, str], episode: str) -> bool:
    """Count this episode's spellings once (a re-translation does not vote again)."""
    done = mem.setdefault("names_he_episodes", [])
    if episode in done:
        return False
    table = mem.setdefault("names_he", {})
    for en, he in names.items():
        he = clean_hebrew_name(he)
        if he and not bad_title(en, he):
            votes = table.setdefault(en, {})
            votes[he] = votes.get(he, 0) + 1
    done.append(episode)
    return True
