"""Self-learning (`hebsub learn`): everything learned stays in the data folder.

From the library:
- Hebrew vocabulary for the spelling pass (tm/vocab.txt, tm/vocab_freq.json), counted over every
  human Hebrew subtitle next to the videos.
- English/Hebrew line pairs (pairs/<id>.json) for videos that have both, aligned by time after a
  search for the best offset and frame-rate scale. series_terms.py reads them: names and terms
  the way the series' own Hebrew subtitles write them.

From the user's corrections:
- A copy of every subtitle hebsub writes is kept in learn/outputs. When the user edits the file
  next to the video, the changed lines go to learn/corrections.jsonl, and a word replaced the
  same way at least twice in a series becomes a rule (learn/user_terms.json) that every later
  translation of that series gets, including the series' name table.

Hebrew written by hebsub is never learned from: it carries the RTL mark at the start of its
lines, and every file hebsub wrote is listed in learn/outputs even after the user edited it.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .paths import HOME

PAIRS = HOME / "pairs"
LEARN = HOME / "learn"
OUTPUTS = LEARN / "outputs"
CORRECTIONS = LEARN / "corrections.jsonl"
USER_TERMS = LEARN / "user_terms.json"
VOCAB, VOCAB_FREQ = HOME / "tm" / "vocab.txt", HOME / "tm" / "vocab_freq.json"
VIDEO = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".wmv"}
HEB_TAGS = (".he.", ".heb.", ".hebrew.", ".iw.")
BIDI = re.compile("[‎‏‪-‮⁦-⁩]")
HEB = re.compile("[א-ת]")
HEB_WORD = re.compile("[א-ת][א-ת׳״'\"]*[א-ת׳']|[א-ת]")
CREDIT = re.compile(r"(תורגם|סונכרן|תרגום:|תרגום ע|הובא|הורד מ|www\.|https?://|subs?center|wizdom|ktuvit|"
                    r"opensubtitles)", re.I)
TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
PREFIX = "והבלמשכ"
MAX_EMBEDDED_GB = 6  # pulling an embedded English track reads the whole file


# ---------------------------------------------------------------- subtitle files

def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if exc.start >= len(raw) - 3:  # a head cut inside the last character is still UTF-8
            return raw[:exc.start].decode("utf-8-sig")
        he = raw.decode("cp1255", "replace")
        return he if len(HEB.findall(he)) > 50 else raw.decode("cp1252", "replace")


def _secs(t: str) -> float:
    h, m, s, ms = map(int, TIME.search(t).groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_srt(path: Path) -> list[dict]:
    """[{start, end, text}] for any encoding Israeli subtitle files use."""
    cues = []
    for block in re.split(r"\r?\n\s*\r?\n", _decode(Path(path).read_bytes()).strip()):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines[:2]):
            if "-->" in line and TIME.search(line):
                a, b = line.split("-->")
                body = "\n".join(lines[i + 1:]).strip()
                if body:
                    cues.append({"start": _secs(a), "end": _secs(b), "text": body})
                break
    return cues


def ours(path: Path) -> bool:
    """A Hebrew file hebsub wrote: RTL embedding mark at the start of its lines, or one listed
    in learn/outputs (the user may have edited it, and the editor may have dropped the marks)."""
    if _record_path(Path(path)).is_file():
        return True
    try:
        head = Path(path).read_bytes()[:4000].decode("utf-8-sig", "replace")
    except OSError:
        return True
    lines = [l for l in head.splitlines() if l and not l[0].isdigit()]
    return bool(lines) and sum(l.startswith("‫") for l in lines) > 0.8 * len(lines)


def clean_en(text: str) -> str:
    t = re.sub(r"<[^>]+>|\{[^}]*\}", "", text)
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", t)              # sound descriptions
    t = re.sub(r"(?m)^\s*-?\s*[A-Z][A-Z .'\-]{1,30}:\s*", "", t)  # SDH speaker labels
    return re.sub(r"\s+", " ", t).strip(" -")


_LEAD_PUNCT = re.compile(r'^(\s*(?:-\s*)?)("?[.,!?…:;]+"?)(.*[א-ת].*?)(\s*)$')
_TRAIL_DASH = re.compile(r"^(\s*)(?!\s*-)(.*[א-ת])\s*-([.,!?…:;\"]*)(\s*)$")


def _logical(line: str) -> str:
    """Many Israeli subtitle files keep the old 'visual order': the final '.', '?' or '...'
    first and the dialogue dash last ('.שלום', 'כן-'), for players without bidi support."""
    m = _LEAD_PUNCT.match(line)
    if m and not re.search(r"[.,!?…:;]$", m.group(3).strip()):
        core, lead = m.group(3).rstrip(), ""
        if core.endswith('"') and core.count('"') % 2:
            core, lead = core[:-1], '"'
        line = f"{m.group(1)}{lead}{core}{m.group(2)[::-1]}{m.group(4)}"
    d = _TRAIL_DASH.match(line)
    return f"{d.group(1)}-{d.group(2)}{d.group(3)}{d.group(4)}" if d else line


def clean_he(text: str) -> str:
    t = BIDI.sub("", re.sub(r"<[^>]+>|\{[^}]*\}", "", text))
    return re.sub(r"\s+", " ", " ".join(_logical(l) for l in t.splitlines())).strip()


# ---------------------------------------------------------------- library scan

def videos(roots) -> list[Path]:
    out = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in VIDEO and not any(part.startswith(("@", ".")) for part in p.parts[len(root.parts):]):
                out.append(p)
    return sorted(out)


def human_hebrew(video: Path) -> Path | None:
    """A Hebrew subtitle next to the video that a person made (not hebsub)."""
    stem = video.with_suffix("").name
    found = []
    for f in video.parent.glob("*.srt"):
        if not f.name.startswith(stem):
            continue
        tag = f.name[len(stem):].lower()
        tagged = any(t in tag + "." for t in HEB_TAGS) or tag in (".he.srt", ".heb.srt")
        if tagged or tag == ".srt":
            try:
                if len(HEB.findall(_decode(f.read_bytes()[:20000]))) > 200 and not ours(f):
                    found.append(f)
            except OSError:
                continue
    return max(found, key=lambda f: f.stat().st_size) if found else None


def english_for(video: Path, cfg, work_dir: Path, log=lambda m: None) -> Path | None:
    from . import subsource
    found = subsource.find_english(video, cfg)
    if found:
        return found
    try:
        if video.stat().st_size > MAX_EMBEDDED_GB * 1e9:
            return None
    except OSError:
        return None
    return subsource.extract_embedded_english(video, cfg, work_dir, log)


# ---------------------------------------------------------------- alignment

def _iou_pairs(en: list[dict], he: list[dict], off: float, scale: float) -> list[tuple[int, int]]:
    return _matches(en, he, off, scale)[1]


def _matches(en: list[dict], he: list[dict], off: float, scale: float) -> tuple[tuple[int, float], list]:
    """((matched lines, total overlap), [(en index, he index)]): the overlap breaks ties."""
    pairs, j, total = [], 0, 0.0
    hs = [(h["start"] * scale + off, h["end"] * scale + off, k) for k, h in enumerate(he)]
    for i, e in enumerate(en):
        best, bk = 0.0, None
        while j < len(hs) and hs[j][1] < e["start"]:
            j += 1
        k = j
        while k < len(hs) and hs[k][0] < e["end"]:
            inter = min(e["end"], hs[k][1]) - max(e["start"], hs[k][0])
            union = max(e["end"], hs[k][1]) - min(e["start"], hs[k][0])
            iou = inter / union if union > 0 else 0
            if iou > best:
                best, bk = iou, hs[k][2]
            k += 1
        if best >= 0.5:
            pairs.append((i, bk))
            total += best
    return (len(pairs), round(total, 3)), pairs


def _one_to_one(pairs):
    used = defaultdict(list)
    for i, k in pairs:
        used[k].append(i)
    return [(i, k) for i, k in pairs if len(used[k]) == 1]


def align(en: list[dict], he: list[dict]) -> tuple[list[tuple[int, int]], float, float]:
    """One-to-one (en index, he index) matches after searching the offset (+-15 s) and the
    frame-rate scale (PAL/NTSC releases); windows that fit a different local offset (a
    different cut: ads, a longer intro) are re-aligned on their own."""
    best = ((0, 0.0), 0.0, 1.0, [])
    for scale in (1.0, 25 / 23.976, 23.976 / 25):
        for step in range(-150, 151, 2):
            score, p = _matches(en, he, step / 10, scale)
            if score > best[0]:
                best = (score, step / 10, scale, p)
    _, off, scale, _ = best
    for fine in range(-10, 11):
        score, p = _matches(en, he, off + fine / 100, scale)
        if score > best[0]:
            best = (score, off + fine / 100, scale, p)
    _, off, scale, pairs = best
    if len(pairs) < 0.9 * len(en):
        windowed = []
        for w in range(0, len(en), 60):
            part = en[w:w + 60]
            got = []
            for step in range(-80, 81, 2):
                p = [(i + w, k) for i, k in _iou_pairs(part, he, off + step / 10, scale)]
                if len(p) > len(got):
                    got = p
            if len(got) >= 0.6 * len(part):
                windowed += got
        if len(windowed) > len(pairs):
            pairs = windowed
    return _one_to_one(pairs), off, scale


def pair_video(video: Path, he_path: Path, en_path: Path) -> dict:
    en = [dict(c, text=clean_en(c["text"])) for c in parse_srt(en_path)]
    en = [c for c in en if len(c["text"]) >= 2]
    he = [dict(c, text=clean_he(c["text"])) for c in parse_srt(he_path)
          if HEB.search(c["text"]) and not CREDIT.search(c["text"])]
    if len(en) < 20 or len(he) < 20:
        return {"error": "too few lines"}
    pairs, off, scale = align(en, he)
    return {"rate": round(len(pairs) / len(en), 3), "offset": off, "scale": round(scale, 4),
            "pairs": [[en[i]["text"], he[k]["text"]] for i, k in pairs]}


# ---------------------------------------------------------------- vocabulary

def words(text: str) -> list[str]:
    return [w.strip("'\"") for w in HEB_WORD.findall(BIDI.sub("", text))]


def build_vocab(hebrew_files: list[Path], log=print, min_count: int = 2) -> int:
    freq: Counter = Counter()
    for f in hebrew_files:
        try:
            for c in parse_srt(f):
                if not CREDIT.search(c["text"]):
                    freq.update(words(c["text"]))
        except OSError:
            continue
    if not freq:
        return 0
    VOCAB.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_FREQ.write_text(json.dumps(dict(freq), ensure_ascii=False), encoding="utf-8")
    VOCAB.write_text("\n".join(sorted(w for w, n in freq.items() if n >= min_count)) + "\n", encoding="utf-8")
    log(f"vocabulary: {sum(n >= min_count for n in freq.values())} Hebrew words from {len(hebrew_files)} subtitle(s)")
    return len(freq)


# ---------------------------------------------------------------- mining

def _key(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def mine(roots, cfg, log=print, limit: int | None = None) -> dict:
    """Vocabulary from every human Hebrew subtitle; aligned pairs for videos with English too."""
    from . import memory
    PAIRS.mkdir(parents=True, exist_ok=True)
    work = HOME / "work" / "learn"
    found, new, pairs, learned = [], 0, 0, set()
    for video in videos(roots):
        he = human_hebrew(video)
        if not he:
            continue
        found.append(he)
        out = PAIRS / f"{_key(video)}.json"
        mtime = he.stat().st_mtime
        if out.is_file():
            try:
                if json.loads(out.read_text(encoding="utf-8")).get("he_mtime") == mtime:
                    continue
            except ValueError:
                pass
        if limit is not None and new >= limit:
            continue
        new += 1
        rec = {"video": str(video), "he": str(he), "he_mtime": mtime,
               "series": memory.series_key(video, cfg.get("memory.series_roots", ["TV Shows"]))}
        en = english_for(video, cfg, work / _key(video), log)
        if not en:
            rec["error"] = "no English"
        else:
            try:
                rec.update(pair_video(video, he, en))
            except Exception as exc:  # noqa: BLE001 - one broken file never stops the scan
                rec["error"] = str(exc)[:200]
        if rec.get("rate", 0) < 0.5:
            rec.pop("pairs", None)
            rec.setdefault("error", f"alignment {rec.get('rate')}")
        else:
            pairs += len(rec["pairs"])
            learned.add(rec["series"])
        out.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        log(f"learn: {video.name}: " + (f"{len(rec['pairs'])} line pairs" if "pairs" in rec else rec["error"]))
    build_vocab(found, log)
    if learned:
        from . import series_terms
        series_terms.clear_cache(learned)
    return {"hebrew_files": len(found), "new_videos": new, "new_pairs": pairs}


# ---------------------------------------------------------------- user corrections

def _record_path(output: Path) -> Path:
    return OUTPUTS / f"{_key(Path(output))}.json"


def _cues(path: Path) -> list[list]:
    return [[round(c["start"], 3), round(c["end"], 3), c["text"]] for c in parse_srt(path)]


def remember_output(cfg, video: Path, english: Path, output: Path) -> None:
    """Keep what hebsub wrote, to tell the user's later edits apart (called after every write)."""
    from . import memory
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rec = {"video": str(video), "output": str(output), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "series": memory.series_key(Path(video), cfg.get("memory.series_roots", ["TV Shows"])),
           "mtime": Path(output).stat().st_mtime, "english": _cues(english), "ours": _cues(output)}
    _record_path(output).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def refresh_baseline(output: Path) -> None:
    """hebsub's own later rewrite of a file (an editorial pass) is not a user correction."""
    p = _record_path(Path(output))
    if p.is_file() and Path(output).is_file():
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec.update(ours=_cues(Path(output)), mtime=Path(output).stat().st_mtime)
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", BIDI.sub("", re.sub(r"<[^>]+>", "", text))).strip()


def _english_at(english: list, start: float, end: float) -> str:
    best, text = 0.0, ""
    for a, b, t in english:
        ov = min(end, b) - max(start, a)
        if ov > best:
            best, text = ov, t
    return clean_en(text)


def collect_corrections(log=print) -> int:
    """Lines the user changed in subtitles hebsub wrote, since the last look."""
    if not OUTPUTS.is_dir():
        return 0
    added = 0
    for p in sorted(OUTPUTS.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        out = Path(rec["output"])
        try:
            mtime = out.stat().st_mtime
        except OSError:
            continue  # deleted, or the drive is not mounted
        if mtime == rec.get("mtime"):
            continue
        now = _cues(out)
        by_start = {round(a, 1): t for a, _, t in now}
        changed, matched = [], 0
        for a, b, t in rec["ours"]:
            theirs = by_start.get(round(a, 1))
            if theirs is None:
                continue
            matched += 1
            if _norm(theirs) != _norm(t):
                changed.append({"series": rec.get("series"), "video": rec["video"], "start": a,
                                "en": _english_at(rec["english"], a, b), "ours": _norm(t), "theirs": _norm(theirs)})
        if matched < 0.5 * max(len(rec["ours"]), 1):
            rec["replaced"] = True  # a different subtitle now: not an edit of ours
        elif changed:
            CORRECTIONS.parent.mkdir(parents=True, exist_ok=True)
            with CORRECTIONS.open("a", encoding="utf-8") as fh:
                for c in changed:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            added += len(changed)
            log(f"learn: {len(changed)} corrected line(s) in {out.name}")
        rec.update(ours=now, mtime=mtime)
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    if added:
        build_rules(log)
    return added


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\s.,!?…:;\"()\-]+", text)


# pronouns and other words whose right form depends on who speaks to whom: a correction of
# them belongs to that line, never to a series-wide rule
_CONTEXT = {"את", "אתה", "אתם", "אתן", "הוא", "היא", "הם", "הן", "אני", "אנחנו", "לך", "לכם", "לכן",
            "אותך", "אותכם", "שלך", "שלכם", "בך", "ממך", "אליך", "עליך", "איתך", "אתך"}
_ENDINGS = {"", "ת", "ה", "י", "ו", "ן", "ך", "ים", "ות", "תי", "נו", "תם", "תן", "כם", "כן", "ני", "תה", "ית"}
_common: dict | None = None


def _rule_ok(old: str, new: str) -> bool:
    """Spellings, names and word choices; not inflections (gender, number) or very common words."""
    global _common
    if old in _CONTEXT or new in _CONTEXT or len(old) < 3:
        return False
    stem = 0
    while stem < min(len(old), len(new)) and old[stem] == new[stem]:
        stem += 1
    if old[stem:] in _ENDINGS and new[stem:] in _ENDINGS:
        return False  # only an inflection ending changed: יודע/יודעת, תעשה/תעשי, ילד/ילדים
    if _common is None:
        try:
            _common = json.loads(VOCAB_FREQ.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _common = {}
    return " " in old or _common.get(old, 0) < 2000


def build_rules(log=print, min_count: int = 2) -> dict:
    """{series: {old: new}} for words (up to 3) the user replaced the same way at least twice."""
    if not CORRECTIONS.is_file():
        return {}
    counts: dict[str, Counter] = defaultdict(Counter)
    for line in CORRECTIONS.read_text(encoding="utf-8").splitlines():
        try:
            c = json.loads(line)
        except ValueError:
            continue
        a, b = _tokens(c["ours"]), _tokens(c["theirs"])
        for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "replace" and i2 - i1 <= 3 and j2 - j1 <= 3:
                old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
                if HEB.search(old) and HEB.search(new):
                    counts[c.get("series") or ""][(old, new)] += 1
    rules: dict[str, dict] = {}
    for series, cnt in counts.items():
        best: dict[str, tuple[int, str]] = {}
        for (old, new), n in cnt.items():
            if n >= min_count and n > best.get(old, (0, ""))[0] and _rule_ok(old, new):
                best[old] = (n, new)
        if best:
            rules[series] = {old: new for old, (_, new) in best.items()}
    USER_TERMS.parent.mkdir(parents=True, exist_ok=True)
    USER_TERMS.write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(r) for r in rules.values())
    if n:
        log(f"learn: {n} replacement rule(s) from the user's corrections")
    return rules


def rules_for(series: str | None) -> dict[str, str]:
    if not USER_TERMS.is_file():
        return {}
    try:
        rules = json.loads(USER_TERMS.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return {**rules.get("", {}), **rules.get(series or "", {})}


def apply_rules(rules: dict[str, str], text: str) -> str:
    """Whole-word replacement that keeps Hebrew prefix letters (ו, ה, ב, ל, מ, ש, כ)."""
    for old in sorted(rules, key=len, reverse=True):
        pat = rf"(?<![א-ת\w])([{PREFIX}]{{0,3}}){re.escape(old)}(?![א-ת\w])"
        text = re.sub(pat, lambda m: m.group(1) + rules[old], text)
    return text


def learn_genders(cfg, log=print, min_count: int = 2) -> dict:
    """A character the user keeps addressing in the other gender ("אתה יודע, מנדי" corrected to
    "את יודעת, מנדי") gets that gender in the series memory, at high confidence, so later
    episodes start from it. Only lines that name whom they speak to count; anything else
    stays a correction of that one line."""
    from . import editorial, memory
    if not CORRECTIONS.is_file():
        return {}
    votes: dict[tuple, Counter] = defaultdict(Counter)
    for line in CORRECTIONS.read_text(encoding="utf-8").splitlines():
        try:
            c = json.loads(line)
        except ValueError:
            continue
        was, now = editorial._labels(c["ours"]).get("you"), editorial._labels(c["theirs"]).get("you")
        names = editorial.vocative_names(c.get("en", ""))
        if not was or not now or was == now or not names or len(set(names)) != 1:
            continue
        votes[(c.get("series") or "", names[0])][now] += 1
    learned = {}
    for (series, name), cnt in votes.items():
        (g, n), = cnt.most_common(1)
        if n < min_count or len(cnt) > 1 or not series:
            continue
        gender = {"m": "masculine", "f": "feminine"}[g]
        mem = memory.load(cfg, series)
        chars = mem.setdefault("characters", [])
        char = next((ch for ch in chars if name.lower() in (x.lower() for x in memory._names(ch))), None)
        if char is None:
            char = {"name": name, "aliases": [], "episodes": []}
            chars.append(char)
        if char.get("locked") or (char.get("grammatical_gender") == gender and char.get("source") == "user"):
            continue
        if char.get("grammatical_gender") != gender:
            log(f"learn: {series}: {name} is {gender} (the user corrected the 'you' form {n} times)")
        char.update(grammatical_gender=gender, confidence="high", source="user")
        memory.save(cfg, series, mem)
        learned.setdefault(series, {})[name] = gender
    return learned


# ---------------------------------------------------------------- entry point

def run(roots, cfg, log=print, mine_library: bool = True, limit: int | None = None) -> dict:
    res = {"corrections": collect_corrections(log)}
    res["genders"] = learn_genders(cfg, log)
    if mine_library and roots:
        res.update(mine(roots, cfg, log, limit=limit))
    return res
