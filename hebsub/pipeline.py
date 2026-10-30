"""Orchestrate the skill's srt_pipeline.py around a translation engine."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from . import cast as cast_mod, editorial, learn, memory
from .engines import Engine, EngineError
from .subsource import acquire_english, find_hebrew
from .paths import HOME, pid_alive

Log = Callable[[str], None]

CUE_RE = re.compile(r"^\s*(\d+)\s*$")


class PipelineError(RuntimeError):
    pass


def _run_script(cfg, args: list[str], log: Log, script: str = "srt_pipeline.py") -> dict:
    script = cfg.skill_dir / "scripts" / script
    if not script.is_file():
        raise PipelineError(f"srt_pipeline.py not found at {script}")
    cmd = [sys.executable, str(script), *args]
    # the skill prints Hebrew JSON: force UTF-8 in the child too (Windows defaults to cp1252)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        raise PipelineError(
            f"srt_pipeline {args[0]} failed (exit {proc.returncode}):\n"
            f"{(proc.stderr.strip() or proc.stdout.strip())[:1500]}"
        )
    out = proc.stdout.strip()
    try:
        return json.loads(out) if out.startswith("{") else {"stdout": out}
    except json.JSONDecodeError:
        return {"stdout": out}


def _dialogue_digest(srt_path: Path, max_chars_per_chunk: int = 5000) -> list[list[str]]:
    """Return chunks of 'cue_id: text' lines for character-bible analysis."""
    raw = srt_path.read_bytes().decode("utf-8-sig", "replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for block in raw.split("\n\n"):
        parts = [p for p in block.split("\n") if p.strip()]
        if len(parts) < 3:
            continue
        m = CUE_RE.match(parts[0])
        if not m or "-->" not in parts[1]:
            continue
        text = " ".join(parts[2:]).strip()
        if text:
            lines.append(f"{m.group(1)}: {text}")

    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) > max_chars_per_chunk and current:
            chunks.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append(current)
    return chunks


def _merge_bibles(base: dict, extra: dict) -> dict:
    by_name = {c.get("name", "").strip().lower(): c
               for c in base.get("characters", []) if c.get("name")}
    for char in extra.get("characters", []) or []:
        key = (char.get("name") or "").strip().lower()
        if not key:
            continue
        if key not in by_name:
            by_name[key] = char
            continue
        cur = by_name[key]
        rank = {"unresolved": 0, "low": 1, "medium": 2, "high": 3}
        if rank.get(char.get("confidence", "unresolved"), 0) > rank.get(cur.get("confidence", "unresolved"), 0):
            cur.update({k: v for k, v in char.items() if k != "evidence"})
        ev = (cur.get("evidence") or []) + (char.get("evidence") or [])
        cur["evidence"] = ev[:8]
    base["characters"] = list(by_name.values())
    seen = {(r.get("from"), r.get("to")) for r in base.get("relationships", [])}
    for rel in extra.get("relationships", []) or []:
        if (rel.get("from"), rel.get("to")) not in seen:
            base.setdefault("relationships", []).append(rel)
    notes = base.setdefault("global_notes", [])
    for note in extra.get("global_notes", []) or []:
        if note not in notes:
            notes.append(note)
    return base


def build_character_bible(cfg, engine: Engine, english_srt: Path,
                          work_dir: Path, log: Log, video: Path | None = None) -> Path:
    bible_path = work_dir / "character-bible.json"
    _run_script(cfg, ["character-template", str(english_srt),
                      "--output", str(bible_path)], log)
    bible = json.loads(bible_path.read_text(encoding="utf-8-sig"))
    bible.setdefault("schema_version", 1)
    bible.setdefault("characters", [])
    bible.setdefault("relationships", [])
    bible.setdefault("global_notes", [])

    # Series memory: characters settled in earlier episodes seed this one.
    series = None
    if video is not None and cfg.get("memory.enabled", True):
        series = memory.series_key(video, cfg.get("memory.series_roots", ["TV Shows"]))
    mem, known = None, []
    episode_text = english_srt.read_text(encoding="utf-8-sig", errors="replace")
    if series:
        mem = memory.load(cfg, series)
        known = memory.relevant(mem, episode_text)
        bible = _merge_bibles(bible, {"characters": [dict(c) for c in known],
                                      "relationships": mem.get("relationships", [])})
        log(f"memory: {series}: {len(known)} known character(s) named in this episode "
            f"(of {len(mem.get('characters', []))} in memory)")

    if not cfg.get("bible.enabled", True):
        log("bible: disabled in config, using speaker labels and series memory only")
        bible_path.write_text(json.dumps(bible, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        return bible_path

    chunks = _dialogue_digest(
        english_srt, int(cfg.get("bible.max_chars_per_chunk", 5000)))
    max_chunks = int(cfg.get("bible.max_chunks", 6))
    if len(chunks) > max_chunks:
        # Sample evenly across the episode rather than only analysing the start.
        step = len(chunks) / max_chunks
        chunks = [chunks[int(i * step)] for i in range(max_chunks)]
        log(f"bible: sampling {max_chunks} chunks across the episode")
    log(f"bible: analysing {len(chunks)} dialogue chunk(s)")
    for i, chunk in enumerate(chunks, 1):
        started = time.time()
        try:
            extra = engine.build_bible_chunk(chunk, known=known)
            bible = _merge_bibles(bible, extra)
            log(f"bible: chunk {i}/{len(chunks)} done in {time.time()-started:.0f}s "
                f"-> {len(bible['characters'])} characters")
        except EngineError as exc:
            log(f"bible: chunk {i} failed after {time.time()-started:.0f}s, "
                f"continuing without it ({exc})")

    # Who plays whom (TMDB): a listed actor's gender beats a guess from dialogue.
    cast = cast_mod.in_text(cast_mod.cast_for(video, cfg), episode_text) if video is not None else []
    if cast:
        n = _apply_cast(bible, cast)
        log(f"cast: {len(cast)} named character(s) from TMDB, {n} bible gender(s) set from it")

    if series:
        mem = memory.merge(mem, bible, video.stem, log)
        saved = memory.save(cfg, series, mem)
        # Rebuild from memory so aliases are folded together and settled genders win.
        in_episode = [c for c in mem["characters"]
                      if video.stem in c.get("episodes", []) or c in known]
        bible["characters"] = in_episode
        bible["relationships"] = mem.get("relationships", [])
        log(f"memory: saved {len(mem['characters'])} character(s) -> {saved}")
        if cast:
            _apply_cast(bible, cast)  # settled memory must not undo the cast list

    bible_path.write_text(json.dumps(bible, ensure_ascii=False, indent=2), encoding="utf-8")
    return bible_path


def _apply_cast(bible: dict, cast: list[dict]) -> int:
    """Set bible genders from the cast list and add cast members the bible missed."""
    idx = cast_mod.gender_index(cast)
    changed = 0
    for c in bible["characters"]:
        names = [c.get("name", "")] + list(c.get("aliases", []))
        g = next((x for x in (cast_mod.lookup(idx, n) for n in names) if x), None)
        if g and (c.get("grammatical_gender") != g or c.get("confidence") != "high"):
            c["grammatical_gender"], c["confidence"] = g, "high"
            changed += 1
    have = {k for c in bible["characters"] for n in [c.get("name", "")] + list(c.get("aliases", []))
            for k in cast_mod._keys(n)}
    for m in cast:
        if not cast_mod._keys(m["name"]) & have:
            bible["characters"].append({"name": m["name"], "grammatical_gender": m["gender"],
                                        "confidence": "high"})
            have |= cast_mod._keys(m["name"])
    return changed


_FONT = re.compile(r"</?font\b[^>]*>", re.IGNORECASE)
_BR = re.compile(r"\s*<br\s*/?>\s*", re.IGNORECASE)
_BIDI = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_OPEN_ONLY = re.compile(r"^([ \t]*)([\[(])([a-z][a-z '\-]*?)([ \t]*)$", re.MULTILINE)
_CLOSE_ONLY = re.compile(r"^([ \t]*)([a-z][a-z '\-]*?)([\])])([ \t]*)$", re.MULTILINE)


_CAPTION_CREDIT = re.compile(r"\b(?:captioned\s+by|captioning\s+(?:by|provided|made|sponsored)|closed\s+captioning|"
                             r"media\s+access\s+group|access\.wgbh|captionmax|vitac|subtitles?\s+by|"
                             r"(?:synced|corrected|ripped|re-?encoded|encoded)\s+(?:and\s+corrected\s+)?by|opensubtitles|"
                             r"addic7ed|\|\s*PSA\b)", re.I)
# קרדיטים של קבוצות תרגום חובבים מתוך כתוביות ASS ("Translation\nVelorien", "{\an6}Typesetting\nJecht",
# "#sgkk@irc.rizon.net", ציורי ASS "m 0 0 l 561 4 ..."): כל פריים של אנימציה הוא כתובית של 40ms,
# ובפרקי Bleach 317-333 תורגמו עשרות עותקים ל"תרגום / Velorien"
_FANSUB_ROLE = (r"(?:translat(?:ion|ors?)(?:\s+check(?:ing)?)?|tl(?:\s+check)?|edit(?:ing|ors?)?|"
                r"typeset(?:ting|ters?|s)?|tim(?:ing|ers?)|encod(?:e|ing|ers?)|karaoke|logo|qc|"
                r"quality\s+check(?:ing)?|special\s+thanks(?:\s+to)?|raws?(?:\s+provider)?|fansub|romaji|kanji)")
# "Editing Staff", "Typesetting and Distro"
_ROLES = (_FANSUB_ROLE + r"(?:\s+staff)?(?:\s*(?:and|&|/|,)\s*(?:" + _FANSUB_ROLE + r"|distro|distribution))*")
_ROLE_ONLY = re.compile(_ROLES + r"\s*:?", re.I)
# "Translation: Velorien", "Translation         Velorien" (רווחים רבים בין התפקיד לשם), "Translated by X"
_ROLE_ONE_LINE = re.compile(_ROLES + r"(?:\s*:\s*|\s{2,})[^\s.?!].{0,40}|(?:translated|edited|typeset|timed|encoded|"
                            r"subbed|fansubbed|karaoke)\s+by\s+[^\s.?!].{0,40}", re.I)
# ג'יבריש של קידוד שגוי (ערבית שנקראה כ-cp1252: "ÊÑÌãÉ = Anastashia")
_MOJIBAKE = re.compile(r"[\u00c0-\u00ff]")
_IRC = re.compile(r"#\w+@irc\.|\birc\.[\w-]+\.(?:net|org|com)\b|\bfansubs?\b", re.I)  # גם "This is a free fansub!"
_ASS_DRAW = re.compile(r"m\s+-?[\d.]+\s+-?[\d.]+(?:\s+(?:[mnlbspc]|-?[\d.]+))*")
_OVERRIDE = re.compile(r"\{[^}]*\}|<[^>]+>")


def _ass_plain(line: str) -> str:
    """בלי תגיות, ו-\\h (רווח קשיח של ASS, "Edit\\n\\h\\h\\h...Shadow8820") הופך לרווח."""
    return _OVERRIDE.sub("", line).replace("\\h", " ").strip()


def _fansub_credit(body_lines: list[str]) -> str | None:
    """'credit' לכתובית קרדיט, 'role' לכתובית שיש בה רק תפקיד (השמות בכתובית נפרדת באותו זמן)."""
    ls = [l for l in (_ass_plain(x) for x in body_lines) if l]
    if not ls:
        return None
    text = " ".join(ls)
    if _IRC.search(text) or _ASS_DRAW.fullmatch(text) or re.fullmatch(r"(?:https?://|www\.)\S+", text, re.I):
        return "credit"
    letters = re.findall(r"[^\W\d_]", text)
    if len(letters) >= 6 and len(_MOJIBAKE.findall(text)) >= 0.4 * len(letters):
        return "credit"
    handle = lambda l: len(l) <= 40 and len(l.split()) <= 5 and not re.search(r"[.?!,]$", l)
    if _ROLE_ONLY.fullmatch(ls[0]):
        if len(ls) == 1:
            return "role"
        return "credit" if len(ls) <= 4 and all(handle(l) for l in ls[1:]) else None
    if len(ls) == 1 and _ROLE_ONE_LINE.fullmatch(ls[0]) and handle(re.split(r":|\s{2,}", ls[0], 1)[-1].strip()):
        return "credit"
    return None


_SDH_TO = {"chuckling": "laughing", "chuckle": "laughs", "laughter": "laughing", "giggling": "laughing",
           "giggles": "laughs", "sighing": "sighs", "gasping": "gasps", "sobbing": "sobs", "screaming": "screaming",
           "cheers": "cheering", "applauding": "applause", "chuckles softly": "chuckles", "laughs softly": "laughs"}
_SDH_ALIAS = re.compile(r"\[(" + "|".join(sorted(map(re.escape, _SDH_TO), key=len, reverse=True)) + r")\]", re.I)


def _repair_sound_tags(src: Path, work_dir: Path, log: Log) -> Path:
    """Close sound tags that lost a bracket, e.g. "growls]" -> "[growls]".

    srt_pipeline only recognises complete [..]/(..) tags as SDH; a broken one is
    sent for translation, the model rightly returns nothing, and the cue fails.
    Works on a copy so a subtitle file next to the video is never modified.
    """
    raw = src.read_bytes()
    try:
        text, recoded = raw.decode("utf-8-sig"), False
    except UnicodeDecodeError:
        # The skill rejects non-UTF-8 input; English legacy files are almost always cp1252.
        text, recoded = raw.decode("cp1252", errors="replace"), True
    # <font ...> wrappers (colour/size from rippers) hide sound descriptions such as
    # '<font size="71">-[eerie music playing]' from the skill's SDH detection.
    # שורות שבירה מעורבות (Windows בתוך כתובית, Bleach 299) מסתירות שורה ריקה באמצע כתובית
    crlf = "\r" in text
    if crlf:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, n0 = _FONT.subn("", text)
    pair = {"[": "]", "(": ")", "]": "[", ")": "("}
    fixed, n1 = _OPEN_ONLY.subn(lambda m: f"{m[1]}{m[2]}{m[3]}{pair[m[2]]}{m[4]}", text)
    fixed, n2 = _CLOSE_ONLY.subn(lambda m: f"{m[1]}{pair[m[3]]}{m[2]}{m[3]}{m[4]}", fixed)
    # caption-house credits the skill does not flag ("Captioned by Media Access Group at
    # WGBH access.wgbh.org"): the whole cue becomes a known sound tag, which the skill drops;
    # "[chuckling]:" -> "[laughing]" (the skill's known tags, no stray speaker colon)
    blocks = fixed.split("\n\n")
    # כתובית בלי טקסט או שזמן הסיום שלה לא אחרי ההתחלה: הכלי דוחה בגללה את כל הקובץ
    # (Bleach 272: "Malformed SRT block", Bleach 300: invalid_duration) - מסירים אותה
    def _secs(t):
        h, m_, s = t.replace(",", ".").split(":")
        return int(h) * 3600 + int(m_) * 60 + float(s)
    # שורה ריקה באמצע כתובית (Bleach 299: שם המתקפה בפסקה נפרדת) יוצרת "כתובית" בלי זמן,
    # והכלי דוחה את הקובץ: הפסקה היתומה מצטרפת לכתובית שלפניה
    merged = []
    for b in blocks:
        if not b.strip():
            continue
        if merged and "-->" not in b and not re.match(r"^\s*\d+\s*$", b.strip().split("\n")[0]):
            merged[-1] = merged[-1].rstrip("\n") + "\n" + "\n".join(l for l in b.split("\n") if l.strip())
            continue
        merged.append(b)
    blocks = merged
    kept, n7, n8, n9 = [], 0, 0, 0
    role_times = set()  # התפקיד ("Karaoke") והשמות לפעמים בשתי כתוביות נפרדות עם אותו זמן בדיוק
    for b in blocks:
        tm = re.search(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", b)
        lines_b = b.strip("\n").split("\n")
        k = next((j for j, l in enumerate(lines_b) if "-->" in l), None)
        if tm and k is not None and _fansub_credit(lines_b[k + 1:]) == "role":
            role_times.add((tm[1], tm[2]))
    last_by_body: dict = {}
    for b in blocks:
        tm = re.search(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", b)
        lines_b = b.strip("\n").split("\n")
        k = next((j for j, l in enumerate(lines_b) if "-->" in l), None)
        if tm and k is not None and (not " ".join(lines_b[k + 1:]).strip() or _secs(tm[2]) <= _secs(tm[1])):
            n7 += 1
            continue
        if tm and k is not None:
            body_l = lines_b[k + 1:]
            kind = _fansub_credit(body_l)
            plain = [l for l in (_ass_plain(x) for x in body_l) if l]
            names_of_role = ((tm[1], tm[2]) in role_times and len(plain) <= 3
                             and all(len(l) <= 40 and not re.search(r"[.?!,]$", l) for l in plain))
            if kind or names_of_role:
                n8 += 1
                continue
            # אותו טקסט בפריים הבא של אנימציה (כתוביות של עשיריות שנייה, רווח של פחות מעשירית):
            # מאריכים את הכתובית הקודמת. משפט שנאמר פעמיים ("It's okay." / "It's okay.") לא מתאחד
            key = " ".join(plain)
            prev = last_by_body.get(key)
            short = _secs(tm[2]) - _secs(tm[1]) <= 0.25
            if key and prev is not None and (short or prev[2]) and _secs(tm[1]) <= prev[1] + 0.1:
                if _secs(tm[2]) > prev[1]:
                    pb = kept[prev[0]]
                    pt = re.search(r"-->\s*(\d+:\d+:\d+[,.]\d+)", pb)
                    kept[prev[0]] = pb[:pt.start(1)] + tm[2] + pb[pt.end(1):]
                last_by_body[key] = (prev[0], max(prev[1], _secs(tm[2])), True)
                n9 += 1
                continue
            if key:
                last_by_body[key] = (len(kept), _secs(tm[2]), short)
        kept.append(b)
    blocks = kept
    n3 = n5 = 0
    after_credit = 0
    # תוויות דובר באותיות רגילות ("Jackson: There he is." / "Lopez: 7-Adam-07"): רק שמות שחוזרים
    # 3 פעמים לפחות בתחילת שורה, ורק כשיש לפחות שני שמות כאלה בקובץ (Rookie, Hobbit)
    _lab = re.compile(r"(?m)^(\s*-?\s*(?:<i>)?)([A-Z][a-z]+(?: [A-Z][a-z]+)?):\s+(?=\S)")
    from collections import Counter as _C
    _cnt = _C(m.group(2) for m in _lab.finditer("\n\n".join(blocks)))
    title_labels = {k for k, v in _cnt.items() if v >= 3} if sum(1 for v in _cnt.values() if v >= 3) >= 2 else set()
    for i, b in enumerate(blocks):
        lines = b.split("\n")
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        body = " ".join(lines[2:])
        # the sponsor lines after "Captioning sponsored by CBS" ("WARNER BROS. TELEVISION",
        # "and TOYOTA.") are short all-caps cues
        if _CAPTION_CREDIT.search(body) or (after_credit and re.fullmatch(r"(?:and\s+)?[A-Z0-9 .,&'!-]{2,40}", body.strip())):
            blocks[i] = "\n".join(lines[:2] + ["[silence]"])
            n3 += 1
            after_credit = 3 if _CAPTION_CREDIT.search(body) else after_credit - 1
            continue
        after_credit = 0
        # closed-caption speaker labels ("JIM: I'm having fun!", "ALL:") are not dialogue;
        # also "[Zi] Ah, a mother's set." / "-[Pavi] What is this place?" (Avatar FENiX SDH)
        new = [re.sub(r"^(\s*-?\s*(?:<i>)?)[A-Z][A-Z .'&-]{1,24}:\s*", r"\1", l) for l in lines[2:]]
        if title_labels:
            new = [_lab.sub(lambda m: m.group(1) if m.group(2) in title_labels else m.group(0), l) for l in new]
        new = [re.sub(r"^(\s*-?\s*)\[(?:[A-Z][\w'.-]*)(?: [A-Z][\w'.-]*){0,2}\]\s*(?=\S)", r"\1", l) for l in new]
        # a sound tag in front of dialogue ("[sharp inhale] The bandits") was translated with it
        new = [re.sub(r"^(\s*-?\s*(?:<i>)?)\[[a-z][^\]\n]{0,40}\]\s*(?=[^\s\[])", r"\1", l) for l in new]
        if new != lines[2:]:
            n5 += 1
            new = [l for l in new if l.strip()] or ["[silence]"]
            blocks[i] = "\n".join(lines[:2] + new)
    fixed = "\n\n".join(blocks)
    fixed, n4 = _SDH_ALIAS.subn(lambda m: "[" + _SDH_TO[m[1].lower()] + "]", fixed)
    fixed = re.sub(r"(\[[^\]\n]+\])\s*:", r"\1", fixed)
    # תיאור מוזיקה בתוך סימני תווים ("♪ ominous music plays ♪") הוא תיאור צליל, לא מילות שיר:
    # בלי סוגריים הוא נשלח לתרגום ("מוזיקה מאיימת מתנגנת"). הופך אותו לסוגריים כדי שיוסר
    # גם תיאור שנשבר על שתי שורות או שאין לו ♪ סוגר ("♪ ominous music plays" / "...music\nplays ♪")
    # some SDH tracks mark music with asterisks ("* foreboding music plays *", Avatar FENiX)
    fixed = re.sub(r"(?m)^([ \t]*-?[ \t]*(?:<i>)?)\*[ \t]*([^*\n]{1,90}?)[ \t]*\*", r"\1♪ \2 ♪", fixed)
    fixed, n6 = re.subn(r"♪[ \t]*((?:(?!\n\s*\n)[^♪]){0,90}?\b(?:music|song|melody|tune|score|humming|whistling|"
                        r"fanfare|plays|playing|continues|ends|swells|fades|stops|resumes|intensifies|builds)\b"
                        r"(?:(?!\n\s*\n)[^♪]){0,50}?)[ \t]*(?:♪|(?=\n\s*\n)|\Z)",
                        lambda m: (f"[{' '.join(m.group(1).split())}]" if len(m.group(1).split()) <= 10
                                   and re.search(r"\bmusic\b|\b(?:plays|playing|continues|ends|swells|fades|stops|"
                                                 r"resumes|intensifies|builds)\s*$", m.group(1).strip(), re.I)
                                   else m.group(0)), fixed, flags=re.I)
    n4 += n6
    if n3:
        log(f"source: dropped {n3} caption credit cue(s)")
    if n5:
        log(f"source: removed speaker labels from {n5} cue(s)")
    if n7:
        log(f"source: dropped {n7} empty or zero-length cue(s)")
    if n8:
        log(f"source: dropped {n8} fansub credit cue(s)")
    if n9:
        log(f"source: merged {n9} repeated animation frame(s)")
    if crlf:
        log("source: normalised Windows line endings")
    if not (n0 + n1 + n2 + n3 + n4 + n5 + n7 + n8 + n9 or recoded or crlf):
        return src
    out = work_dir / "source.repaired.en.srt"
    out.write_text(fixed, encoding="utf-8")
    if recoded:
        log("source: converted from cp1252 to UTF-8")
    if n0:
        log(f"source: removed {n0} <font> tag(s)")
    if n1 + n2:
        log(f"source: closed {n1 + n2} broken sound tag(s)")
    return out


def add_scene_map(engine: Engine, english: Path, bible_path: Path, log: Log,
                  window: int = 400, overlap: int = 40) -> None:
    """Store who-is-in-each-scene in the bible so every batch knows who "you" is."""
    lines = []
    for block in english.read_text(encoding="utf-8-sig", errors="replace").strip().split("\n\n"):
        parts = block.strip().split("\n")
        if len(parts) >= 3 and parts[0].strip().isdigit():
            text = " / ".join(re.sub(r"<[^>]+>", "", l) for l in parts[2:]).strip()
            if text:
                lines.append(f"{parts[0].strip()}: {text}")
    bible = json.loads(bible_path.read_text(encoding="utf-8-sig"))
    scenes: list[dict] = []
    started = time.time()
    for start in range(0, len(lines), window - overlap):
        chunk = lines[start:start + window]
        try:
            found = engine.scene_map(chunk, bible.get("characters", []))
        except EngineError as exc:
            log(f"scenes: window at cue line {start} failed ({exc})")
            continue
        last = scenes[-1]["to"] if scenes else 0
        scenes += [sc for sc in found if sc["to"] > last]  # later window wins the overlap
        if start + window >= len(lines):
            break
    bible["scenes"] = scenes
    bible_path.write_text(json.dumps(bible, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"scenes: {len(scenes)} scene(s) mapped in {time.time() - started:.0f}s")


VOICE_PYTHON = HOME / "train/venv/bin/python"
VOICE_SCRIPT = HOME / "voice/hints.py"


def voice_hints(cfg, engine: Engine, video: Path, english: Path, work_dir: Path, log: Log) -> dict:
    """{"cue": "f>m"} from the soundtrack (voice/hints.py, CPU, ~2-5 min); {} when unavailable.

    Used by the gender pass. Local engines only by default: Gemini already gets gender
    nearly right, and the voice tools live in the training venv on this Mac.
    """
    if not cfg.get(f"{engine.name}.gender_pass", engine.name == "ollama"):
        return {}
    if not (VOICE_PYTHON.is_file() and VOICE_SCRIPT.is_file()):
        # any OS: the optional voicekit (sherpa-onnx, CPU) when its models are installed
        from . import voicekit
        if not voicekit.available():
            return {}
        started = time.time()
        try:
            hints = voicekit.voice_hints(video, english, work_dir, ffmpeg=cfg.get("ffmpeg.binary", "ffmpeg"))
        except Exception as exc:  # noqa: BLE001
            log(f"voices: analysis failed, continuing without ({str(exc)[:200]})")
            return {}
        log(f"voices: {len(hints)} cues with voice hints in {(time.time() - started) / 60:.1f} min")
        return hints
    out = work_dir / "voices.json"
    if not out.is_file():
        started = time.time()
        res = subprocess.run([str(VOICE_PYTHON), str(VOICE_SCRIPT), str(video), str(english), str(out)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
        if res.returncode != 0 or not out.is_file():
            log(f"voices: analysis failed, continuing without ({(res.stderr or res.stdout)[-200:]})")
            return {}
        log(f"voices: {res.stdout.strip()} in {(time.time() - started) / 60:.1f} min")
    return json.loads(out.read_text(encoding="utf-8"))


ADDR_SCRIPT = HOME / "voice/addr_classify.py"
ADDR_ADAPTER = HOME / "voice/addr_adapter"
SPK_SCRIPT = HOME / "voice/spk_classify.py"
SPK_ADAPTER = HOME / "voice/spk_adapter"


def speaker_classifier(cfg, engine: Engine, video: Path, english: Path, bible_path: Path,
                       work_dir: Path, log: Log, extra: list[int] | None = None) -> dict[str, float]:
    """{cue: p(speaker is a woman)} from the trained speaker classifier, which reads the
    dialogue around the line as well as the voice (a mid voice saying "Man, I hate..." is a
    man). MLX, so Ollama's copy of the model is unloaded first."""
    if not cfg.get(f"{engine.name}.speaker_classifier", False) or not (SPK_ADAPTER / "adapters.safetensors").is_file():
        return {}
    out = work_dir / "speaker.json"
    if not out.is_file():
        model = getattr(engine, "model", "")
        from .engines import MLX_LOCK
        MLX_LOCK.write_text(str(os.getpid()))
        if model:
            subprocess.run(["ollama", "stop", model], capture_output=True)
        npz = HOME / "voice/work" / video.stem / "cues.npz"
        started = time.time()
        extra_file = work_dir / "spk_extra.json"
        extra_file.write_text(json.dumps(sorted(extra or [])))
        try:
            res = subprocess.run([str(VOICE_PYTHON), str(SPK_SCRIPT), str(english), str(npz) if npz.is_file() else "-",
                                  str(bible_path), str(out), "-", str(extra_file)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
        finally:
            MLX_LOCK.unlink(missing_ok=True)
        if res.returncode != 0 or not out.is_file():
            log(f"speaker: classifier failed, continuing without ({(res.stderr or res.stdout)[-200:]})")
            return {}
        log(f"speaker: {res.stdout.strip()[-80:]} in {(time.time() - started) / 60:.1f} min")
    return json.loads(out.read_text(encoding="utf-8"))


def addressee_classifier(cfg, engine: Engine, video: Path, english: Path, bible_path: Path,
                         work_dir: Path, log: Log, extra: list[int] | None = None) -> dict[int, str]:
    """{cue: 'f'|'m'} for "you" lines where the trained addressee classifier is sure.

    Held-out precision: woman 94.6% at p>=0.98 (12% of female addressees), man 99.1% at
    p<=0.05. Runs in MLX, so Ollama's copy of the model is unloaded first (32 GB Mac)."""
    if not cfg.get(f"{engine.name}.addressee_classifier", False):
        return {}
    if not ADDR_ADAPTER.is_dir():
        # any OS: the portable ONNX classifier (voicekit), with its own calibrated thresholds
        from . import voicekit
        if not voicekit.addressee_available():
            return {}
        started = time.time()
        try:
            bible = json.loads(bible_path.read_text(encoding="utf-8-sig"))
            p = voicekit.addressee(english, bible, work_dir, set(extra or []))
        except Exception as exc:  # noqa: BLE001
            log(f"addressee: portable classifier failed, continuing without ({str(exc)[:200]})")
            return {}
        hi = float(cfg.get("addressee.onnx_woman_threshold", 0.965))
        lo = float(cfg.get("addressee.onnx_man_threshold", 0.061))
        skip = set(extra or [])  # imperatives were not in the classifier's training: veto use only
        out = {int(k): ("f" if v >= hi else "m") for k, v in p.items() if (v >= hi or v <= lo) and int(k) not in skip}
        log(f"addressee: {len(p)} lines classified in {(time.time() - started) / 60:.1f} min (portable)")
        log(f"addressee: {sum(1 for g in out.values() if g == 'f')} woman / "
            f"{sum(1 for g in out.values() if g == 'm')} man addressees from the classifier")
        return out
    out = work_dir / "addressee.json"
    if not out.is_file():
        model = getattr(engine, "model", "")
        from .engines import MLX_LOCK
        MLX_LOCK.write_text(str(os.getpid()))  # עבודות אחרות לא יטענו את מודל התרגום בינתיים
        if model:
            subprocess.run(["ollama", "stop", model], capture_output=True)
        npz = HOME / "voice/work" / video.stem / "cues.npz"
        started = time.time()
        extra_file = work_dir / "addr_extra.json"
        extra_file.write_text(json.dumps(sorted(extra or [])))
        try:
            res = subprocess.run([str(VOICE_PYTHON), str(ADDR_SCRIPT), str(english), str(npz) if npz.is_file() else "-",
                                  str(bible_path), str(out), "-", str(extra_file)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
        finally:
            MLX_LOCK.unlink(missing_ok=True)
        if res.returncode != 0 or not out.is_file():
            log(f"addressee: classifier failed, continuing without ({(res.stderr or res.stdout)[-200:]})")
            return {}
        log(f"addressee: {res.stdout.strip()[-80:]} in {(time.time() - started) / 60:.1f} min")
    p = json.loads(out.read_text(encoding="utf-8"))
    hi, lo = float(cfg.get("addressee.woman_threshold", 0.98)), float(cfg.get("addressee.man_threshold", 0.05))
    skip = set(extra or [])  # imperatives were not in the classifier's training: veto use only
    got = {int(k): ("f" if v >= hi else "m") for k, v in p.items() if (v >= hi or v <= lo) and int(k) not in skip}
    log(f"addressee: {sum(g == 'f' for g in got.values())} woman / {sum(g == 'm' for g in got.values())} man "
        f"addressees from the classifier")
    return got


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def _degenerate(got: dict[int, str], items: list[dict]) -> set[int]:
    """Cues whose Hebrew is clearly broken output, not translation.

    Two signatures seen from local models: the same short text repeated across many
    different source cues (a decoding loop), and Hebrew that is a tiny fraction of a
    long English line.
    """
    src = {it["cue_id"]: _plain(it.get("source", "")) for it in items if it.get("translate", True)}
    bad: set[int] = set()
    groups: dict[str, list[int]] = {}
    for cid, text in got.items():
        if cid not in src:
            continue
        t = _plain(text)
        groups.setdefault(t, []).append(cid)
        if len(src[cid]) > 20 and len(t) < 0.2 * len(src[cid]):
            bad.add(cid)
    for t, cids in groups.items():
        if len(cids) >= 3 and len({src[c].casefold() for c in cids}) >= 3:
            bad.update(cids)
    return bad


_HE_LETTER = "\u05d0-\u05ea"


def _quote_cut(got: dict[int, str], order: list[int]) -> int | None:
    """First cue whose text a raw " cut off inside a Hebrew acronym, or None.

    Under the JSON grammar the quote in התנ"ך / ל"ט closes the string: the cue ends on
    the acronym's stem ("...התנ", "פסוק ל") and the next cue opens with its last letter
    ("ך כן מדבר..."), so every later text sits one cue late. Gershayim always stands
    before an acronym's last letter, and Hebrew has no one-letter words.
    """
    ids = [c for c in order if c in got]
    for k, cid in enumerate(ids):
        t = re.sub(r"<[^>]+>", "", got[cid]).strip()
        if not re.search(f"[{_HE_LETTER}]$", t):
            continue
        if editorial.CUT_END.search(t):
            return cid
        nxt = re.sub(r"<[^>]+>", "", got[ids[k + 1]]).strip() if k + 1 < len(ids) else ""
        if re.match(f"[{_HE_LETTER}](?:[\\s.,!?…:;]|$)", nxt):
            return cid
    return None


def _translate_one_batch(engine: Engine, batch: dict, bible: dict,
                         sdh_mode: str, log: Log, attempts: int = 3,
                         allow_partial: bool = False) -> list[dict]:
    """Translate one batch; keep good cues across attempts and retry only the rest.

    A few cues that still fail are then tried one at a time. With allow_partial
    (evaluation only) cues that never succeed come back empty instead of failing
    the whole batch.
    """
    items = batch.get("items", [])
    wanted = [it["cue_id"] for it in items]
    needs_text = {it["cue_id"] for it in items if it.get("translate", True)}
    # Omitted cues are always empty: any text there is the model ignoring the contract.
    done: dict[int, str] = {c: "" for c in wanted if c not in needs_text}
    pending = batch
    last_error = ""
    free = False  # after a quote cut: translate without the JSON grammar
    for attempt in range(1, attempts + 1):
        try:
            if free:
                result = engine.translate_batch(pending, bible, sdh_mode, constrained=False)
            else:
                result = engine.translate_batch(pending, bible, sdh_mode)
        except EngineError as exc:
            last_error = str(exc)
            log(f"  batch {batch.get('batch_number')} attempt {attempt}: {exc}")
            free = False  # free output unusable: back to the grammar
            time.sleep(3 * attempt)
            continue
        asked = {it["cue_id"] for it in pending["items"]}
        got: dict[int, str] = {}
        for rec in result:
            if isinstance(rec, dict) and "cue_id" in rec:
                try:
                    cid, text = int(rec["cue_id"]), str(rec.get("text", ""))
                except (TypeError, ValueError):
                    continue
                # Some models write HTML line breaks; SRT needs a real newline. Bidi
                # controls are stripped: the skill adds its own tested RTL profile.
                text = _BIDI.sub("", _BR.sub("\n", text))
                # A blank translation fails `validate --strict`, so treat it as missing; so is
                # JSON debris from a garbled free answer ('לכת". }, { "cue_id": 51, ...').
                if re.search(r'"cue_id"|"text"\s*:|\}\s*,\s*\{', text):
                    continue
                if cid in asked and cid in needs_text and text.strip():
                    got[cid] = text
        broken = _degenerate(got, pending["items"])
        for cid in broken:
            got.pop(cid, None)
        # Free output cannot be cut by a quote: a hit there is a stutter ("מ... מ"). On the
        # last attempt a cut text beats failing the batch (truncation_pass re-does that line).
        cut = None if free or attempt == attempts else _quote_cut(got, [it["cue_id"] for it in pending["items"]])
        if cut is not None:
            order = [it["cue_id"] for it in pending["items"]]
            for cid in order[order.index(cut):]:
                got.pop(cid, None)
            log(f"  batch {batch.get('batch_number')}: a quote cut cue {cut} (acronym), "
                f"redoing it and the cues after it without the JSON grammar")
            free = True
        done.update(got)
        missing = [c for c in wanted if c not in done]
        if not missing:
            return [{"cue_id": c, "text": done[c]} for c in wanted]
        last_error = (f"missing cue_ids {missing[:10]}"
                      + (f" ({len(broken)} broken output)" if broken else ""))
        log(f"  batch {batch.get('batch_number')} attempt {attempt}: {last_error}")
        pending = dict(batch, items=[it for it in items if it["cue_id"] in missing])
    missing = [c for c in wanted if c not in done]
    if len(missing) > 5:
        # המודל נתקע בלולאה על מנה שלמה (Bleach 293: 29 שורות שבורות, 3 פעמים): מנות של 5
        # בדרך כלל עוברות. רק מה שעדיין חסר אחר כך עובר לשלב של שורה אחת בכל פעם
        rest = [it for it in items if it["cue_id"] in missing]
        for s in range(0, len(rest), 5):
            part = rest[s:s + 5]
            try:
                rec = engine.translate_batch(dict(batch, items=part), bible, sdh_mode, constrained=not free)
            except EngineError:
                continue
            got = {}
            for r in rec:
                if isinstance(r, dict) and r.get("cue_id") in {it["cue_id"] for it in part}:
                    text = _BIDI.sub("", _BR.sub("\n", str(r.get("text", ""))))
                    if text.strip() and not re.search(r'"cue_id"|"text"\s*:|\}\s*,\s*\{', text):
                        got[int(r["cue_id"])] = text
            for cid in _degenerate(got, part):
                got.pop(cid, None)
            done.update(got)
        missing = [c for c in wanted if c not in done]
        log(f"  batch {batch.get('batch_number')}: retried in chunks of 5, {len(missing)} cue(s) still missing")
    if len(missing) <= 5:  # last resort: each stubborn cue on its own
        for it in [it for it in items if it["cue_id"] in missing]:
            try:
                rec = engine.translate_batch(dict(batch, items=[it]), bible, sdh_mode, constrained=not free)
            except EngineError:
                continue
            for r in rec:
                text = _BR.sub("\n", str(r.get("text", ""))) if isinstance(r, dict) else ""
                if isinstance(r, dict) and r.get("cue_id") == it["cue_id"] and text.strip():
                    done[it["cue_id"]] = text
        missing = [c for c in wanted if c not in done]
        if not missing:
            log(f"  batch {batch.get('batch_number')}: recovered the last cue(s) one by one")
            return [{"cue_id": c, "text": done[c]} for c in wanted]
    if allow_partial:
        return [{"cue_id": c, "text": done.get(c, "")} for c in wanted]
    raise PipelineError(f"Batch {batch.get('batch_number')} failed: {last_error}")


def _text_len(t: str) -> int:
    return max(len(re.sub(r"<[^>]+>|\s+", "", t or "")), 1)


def misaligned(items: list[dict], records: list[dict]) -> str:
    """Why a batch's texts look slid against their cue ids ('' when they don't).

    Seen on an episode opening with a title card and song lyrics: correct cue_ids, but
    every text two or three cues late. Signs: the same Hebrew line returned for different
    English lines, or Hebrew lengths fitting a neighbouring cue's English much better.
    """
    import math
    he = {r["cue_id"]: r["text"] for r in records}
    rows = [(it.get("source", ""), he.get(it["cue_id"], "")) for it in items if it.get("translate", True)]
    if len(rows) < 6:
        return ""
    seen, dup = {}, 0
    for en, h in rows:
        key = re.sub(r"\W", "", h)
        if len(key) >= 10:
            if key in seen and re.sub(r"\W", "", seen[key]) != re.sub(r"\W", "", en):
                dup += 1
            seen[key] = en

    def cost(k):
        pairs = [(h, rows[i + k][0]) for i, (_, h) in enumerate(rows) if 0 <= i + k < len(rows)]
        return sum(abs(math.log(_text_len(h) / _text_len(e))) for h, e in pairs) / max(len(pairs), 1)
    c0 = cost(0)
    best = min((cost(k), k) for k in (-3, -2, -1, 1, 2, 3))
    if dup >= 2:
        return f"{dup} repeated lines"
    if best[0] < c0 - 0.25:
        return f"lengths fit cue offset {best[1]}"
    return ""


def translate_checked(engine: Engine, batch: dict, bible: dict, sdh_mode: str, log: Log) -> list[dict]:
    """_translate_one_batch, re-done in halves when the result looks slid."""
    records = _translate_one_batch(engine, batch, bible, sdh_mode, log)
    why = misaligned(batch.get("items", []), records)
    if not why or len(batch.get("items", [])) < 8:
        return records
    log(f"  batch {batch.get('batch_number')}: texts look shifted ({why}), translating in halves")
    items = batch["items"]
    half = len(items) // 2
    redone = []
    for part in (items[:half], items[half:]):
        redone += _translate_one_batch(engine, dict(batch, items=part), bible, sdh_mode, log)
    if not misaligned(items, redone):
        return redone
    # still slid (seen when a batch opens with a title card and song lyrics): small chunks
    log(f"  batch {batch.get('batch_number')}: still suspicious after halving, translating 5 cues at a time")
    small = []
    for s in range(0, len(items), 5):
        small += _translate_one_batch(engine, dict(batch, items=items[s:s + 5]), bible, sdh_mode, log)
    return small


TRANSLATING_LOCK = HOME / "translating.lock"
TRAINER_PID = HOME / "train" / "loop" / "train.pid"


MLX_TRANSLATE = HOME / "train/mlx_translate.py"


def mlx_translate(cfg, engine: Engine, work_dir: Path, sdh_mode: str, log: Log) -> None:
    """Translation batches by the trained model (MLX + LoRA adapter) in a separate process.

    Off unless cfg mlx_translate.adapter (or env HEBSUB_MLX_ADAPTER) points to an adapter.
    Ollama's copy of the model is unloaded first and MLX_LOCK keeps it unloaded: the two
    26B models never share the 32 GB. Batches that fail here are left for Ollama."""
    adapter = os.environ.get("HEBSUB_MLX_ADAPTER") or cfg.get("mlx_translate.adapter")
    if not adapter or engine.name != "ollama" or not (VOICE_PYTHON.is_file() and MLX_TRANSLATE.is_file()):
        return
    from .engines import MLX_LOCK
    MLX_LOCK.write_text(str(os.getpid()))
    try:
        model = getattr(engine, "model", "")
        if model:
            subprocess.run(["ollama", "stop", model], capture_output=True)
        started = time.time()
        res = subprocess.run([str(VOICE_PYTHON), str(MLX_TRANSLATE), str(work_dir), str(adapter), sdh_mode],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=4 * 3600)
        tail = (res.stdout.strip().splitlines() or [""])[-1]
        log(f"translate (trained model): {tail if res.returncode == 0 else 'failed: ' + (res.stderr or '')[-200:]}"
            f" in {(time.time() - started) / 60:.1f} min")
    finally:
        MLX_LOCK.unlink(missing_ok=True)


def _claim_gpu(engine: Engine, log: Log) -> bool:
    """Local engines need the GPU memory the trainer uses: pause training first."""
    if engine.name not in ("mlx", "ollama"):
        return False
    TRANSLATING_LOCK.write_text(str(os.getpid()))
    try:
        pid = int(TRAINER_PID.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        log(f"training paused for this translation (stopped pid {pid})")
        for _ in range(60):
            if not pid_alive(pid):
                break
            time.sleep(1)
    except (FileNotFoundError, ValueError, ProcessLookupError):
        pass
    return True


# screams, effort and laughs too: human Hebrew keeps Aaah 4%, Ugh 2%, Hyah 5%, Hahaha 5% (1,208 aligned
# videos); not Wow (36%), Whoa / Ouch / Argh (15%)
_INTERJECTION = re.compile(r"(?:(?:hm+|m+|mm+-?hm+|uh-?huh|ah+|oh+|uh+|um+|huh|er+|eh|a{2,}h*|a+h{2,}|u+g+h+"
                           r"|hy+a+h+|hi+ya+h+|g+r{2,}|ga+h+|(?:ha){2,}h?|tsk+)[.!?…,\s-]*)+", re.I)


def _drop_interjections(srt: Path, english: dict[int, str]) -> int:
    """Remove cues whose English is only a hesitation sound: professional Hebrew subtitles
    omit 94-98% of them (measured on 380k aligned lines); "Wow", "Yeah", "Okay" stay."""
    raw = srt.read_text(encoding="utf-8-sig")
    blocks = [b for b in re.split(r"\n{2,}", raw.strip("\n")) if b.strip()]
    keep, dropped = [], 0
    for b in blocks:
        head = b.split("\n", 1)[0].strip()
        en = re.sub(r"<[^>]+>|\{[^}]*\}|\[[^\]]*\]|\([^)]*\)", "", english.get(int(head), "") if head.isdigit() else "")
        if head.isdigit() and en.strip() and _INTERJECTION.fullmatch(en.strip()) \
                and not re.search(r"uh-?oh", en, re.I):  # "Uh-oh" means something (אוי-אוי)
            dropped += 1
            continue
        keep.append(b)
    if dropped:
        srt.write_text("\n\n".join(keep) + "\n", encoding="utf-8-sig")
    return dropped


def _drop_untranslated(srt: Path, transcript: bool = False) -> int:
    """Remove cues with nothing for a Hebrew reader: a whole-cue sound tag ("(מקהלה שרה)",
    "[בערבית]"; not a top / bold sign like Bleach's "{\\an8}[ לוטוס לבנה ]"), and, with no Hebrew letter, only symbols ("¶¶", "* *"), an ASS drawing
    ("m 0 0 l 561 4 ...") or an IRC / web credit. When the English is a Whisper or OCR
    transcript, also untranslatable gibberish ("Ithnmc.enre.", "usc разг Gearablığım"): no
    Hebrew but lowercase Latin or another script (acronyms and numbers stay). Not for real
    subtitles, where a romaji attack name in bold is kept on purpose (Bleach)."""
    raw = srt.read_text(encoding="utf-8-sig")
    blocks = [b for b in re.split(r"\n{2,}", raw.strip("\n")) if b.strip()]
    keep, dropped = [], 0
    for b in blocks:
        lines = b.split("\n")
        t = re.sub(r"[\u200e\u200f\u202a-\u202e]|<[^<>]*>|\{[^}]*\}", "", " ".join(lines[2:])).strip()
        bare = not re.search(r"[א-ת]", t) and (
            not re.search(r"\w", t) or re.match(r"m -?\d", t) or re.search(r"@|\birc\.|www\.", t)
            or (transcript and re.search(r"[a-z]|[^\W\d_A-Za-z]", t)))
        # a bracketed cue placed at the top or in bold is an on-screen sign / attack name, not SDH
        tag = re.fullmatch(r"[(\[][^)\]]*[)\]]", t) and not re.search(r"\{\\an\d|<b>", b)
        if "-->" in (lines[1] if len(lines) > 1 else "") and (bare or tag):
            dropped += 1
            continue
        keep.append(b)
    if dropped:
        srt.write_text("\n\n".join(keep) + "\n", encoding="utf-8-sig")
    return dropped


def finish_translation(cfg, engine: Engine, video: Path, english: Path, work_dir: Path, bible: dict,
                       bible_path: Path, voices: dict, sdh_mode: str, decisions, log: Log,
                       output_dir: Path | None = None, reviewed: Path | None = None,
                       transcript: bool = False) -> Path:
    """Everything after the batches are translated: merge, repairs, review, gender, spelling,
    build and audit. Separate so post-processing changes can be re-tested on saved batches
    (bench/post_only.py) without translating again."""
    out_dir = work_dir / "translated_batches"
    manifest_path = work_dir / "manifest.json"
    _run_script(cfg, ["merge", "--input-dir", str(out_dir),
                      "--output", str(work_dir / "translations.json")], log)
    if engine.name in ("ollama", "mlx"):
        # repair before validating: English left in lines, names in Latin, 3-line cues
        tpath = work_dir / "translations.json"
        recs = json.loads(tpath.read_text(encoding="utf-8"))
        man = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        src0 = {c["index"]: c.get("clean_text") or c.get("source_text", "") for c in man["cues"]}
        recs, _ = editorial.leftover_pass(engine, recs, src0, log, names=bible.get("names_he"))
        recs, _ = editorial.truncation_pass(engine, recs, src0, log, names=bible.get("names_he"))
        recs, _ = editorial.terms_pass(engine, recs, src0, bible.get("names_he"), log)
        for r in recs:
            r["text"] = editorial.reflow(r.get("text") or "")
        tpath.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        _run_script(cfg, ["validate",
                          "--manifest", str(work_dir / "manifest.json"),
                          "--translations", str(work_dir / "translations.json"),
                          "--strict"], log)
    except PipelineError as exc:
        # strict mode also fails on warnings (a name-only line, a long line); hard
        # errors (missing/extra cues, placeholders...) still stop the run below
        log(f"validate: strict check reported warnings, continuing ({str(exc)[:160]})")
        plain = ["validate", "--manifest", str(work_dir / "manifest.json"),
                 "--translations", str(work_dir / "translations.json")]
        try:
            _run_script(cfg, plain, log)
        except PipelineError as exc2:
            # a sound description the source cleanup missed ("[chuckling]:" -> "(צוחקת):") fails
            # the whole episode; drop the bracketed part from just those cues and check again
            msg = str(exc2)
            bad = {int(x) for key in ("parenthetical_violations", "bracket_violations")
                   for m in re.finditer(rf'"{key}":\s*\[([^\]]*)\]', msg) for x in re.findall(r"\d+", m.group(1))}
            others = [m.group(1) for m in re.finditer(r'"(?!parenthetical_violations|bracket_violations)(\w+)":\s*\[\s*[^\]\s]', msg)
                      if m.group(1) in ("missing_ids", "extra_ids", "blank_required", "music_violations",
                                        "placeholders", "output_artifacts", "invalid_duration")]
            # כתובית שיצאה ריקה (Bleach 304: blank_required) מתורגמת מחדש לבד, במקום להכשיל את הפרק
            blank = {int(x) for m in re.finditer(r'"blank_required":\s*\[([^\]]*)\]', msg)
                     for x in re.findall(r"\d+", m.group(1))}
            others = [o for o in others if not (o == "blank_required" and blank)]
            if (not bad and not blank) or others:
                raise
            tpath = work_dir / "translations.json"
            recs = json.loads(tpath.read_text(encoding="utf-8"))
            if blank:
                man = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                src_b = {c["index"]: c.get("clean_text") or c.get("source_text", "") for c in man["cues"]}
                for r in recs:
                    if r["cue_id"] in blank and src_b.get(r["cue_id"], "").strip():
                        try:
                            got = engine.translate_batch({"batch_number": 0, "items": [
                                {"cue_id": r["cue_id"], "source": src_b[r["cue_id"]], "translate": True}]},
                                bible, "remove")
                            text = next((str(x.get("text", "")) for x in got
                                         if isinstance(x, dict) and x.get("cue_id") == r["cue_id"]), "")
                            if text.strip():
                                r["text"] = _BR.sub("\n", text)
                        except EngineError:
                            pass
                log(f"validate: re-translated empty cue(s) {sorted(blank)}")
            for r in recs:
                if r["cue_id"] in bad:
                    t = re.sub(r"\s*[\[(][^\])\n]*[\])]:?\s*", " ", r.get("text") or "")
                    t = "\n".join(l.strip() for l in t.split("\n") if l.strip())
                    if t:
                        r["text"] = t
            tpath.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
            if bad:
                log(f"validate: removed sound descriptions left in cue(s) {sorted(bad)}")
            _run_script(cfg, plain, log)

    translations = work_dir / "translations.json"
    rounds = int(cfg.get(f"{engine.name}.review_rounds", cfg.get("review.rounds", 0)) or 0)
    if reviewed is not None:  # bench: reuse an earlier run's reviewed + checked lines
        rounds = 0
        translations = work_dir / "translations.reused.json"
        shutil.copy2(reviewed, translations)
    run = lambda a, script="srt_pipeline.py": _run_script(cfg, a, log, script=script)
    for r in range(1, rounds + 1):
        started = time.time()
        res = editorial.review_round(engine, run, manifest_path, translations, bible,
                                     work_dir / f"review_r{r}", log,
                                     compact=bool(cfg.get(f"{engine.name}.review_compact", False)),
                                     parallel=max(1, int(cfg.get(f"{engine.name}.parallel", 1) or 1)))
        if res is None:
            break
        translations, changed = res
        log(f"review: round {r}/{rounds} changed {changed} cue(s) in "
            f"{(time.time() - started) / 60:.1f} min")
        if changed == 0:
            break
    if reviewed is None and translations != work_dir / "translations.json" and cfg.get(f"{engine.name}.review_verify", True):
        # the review fixes ~3x more lines than it breaks; a blind two-order re-check undoes
        # part of the breakage
        man = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        src0 = {c["index"]: c.get("clean_text") or c.get("source_text", "") for c in man["cues"]}
        after, undone = editorial.verify_corrections(
            engine, json.loads((work_dir / "translations.json").read_text(encoding="utf-8")),
            json.loads(translations.read_text(encoding="utf-8")), src0, log)
        if undone:
            translations = work_dir / "translations.checked.json"
            translations.write_text(json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")

    if cfg.get(f"{engine.name}.gender_pass", engine.name == "ollama"):
        # last word on gender: speaker from the voice (96%), addressee only when named
        # ("Abby, you...", 99%), plural masculine unless the scene is all women
        records = json.loads(translations.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        src = {c["index"]: c.get("clean_text") or c.get("source_text", "") for c in manifest["cues"]}
        listener = bool(cfg.get("gender.listener", False))  # voice-only listener: ~75% in live action
        hints = voices
        if listener:
            hints, filled = editorial.fill_listeners(voices, bible)
            log(f"gender: {filled} listener(s) completed from the scene map")
        named = editorial.vocative_listeners(src, bible)
        log(f"gender: {len(named)} line(s) address a known character by name")
        # human subtitles in Polish / Russian / Arabic / Spanish inside the file already decided
        # "I" and "you" for some lines (98-100% right on 30 library videos): they win
        xl = {}
        if cfg.get("xling.enabled", True):
            from . import xling
            tracks = xling.extract(video, cfg, work_dir, log)
            if tracks:
                xl = xling.hints(tracks, manifest)
                for cid, g in xl.items():
                    if g.get("you"):
                        named[cid] = g["you"]
                log(f"xling: {sum('speaker' in g for g in xl.values())} speaker and "
                    f"{sum('you' in g for g in xl.values())} addressee gender(s) from other languages")
        # lines without "you" that the translation renders as addressing someone (imperatives)
        extra = [r["cue_id"] for r in records if editorial._labels(r.get("text") or "").get("you")
                 and not re.search(r"\byou", src.get(r["cue_id"], ""), re.I)]
        for cid, g in addressee_classifier(cfg, engine, video, english, bible_path, work_dir, log,
                                           extra=extra).items():
            named.setdefault(cid, g)  # a name said aloud beats the classifier
        # vetoes: the translator's own gender choice against clear audio / classifier evidence
        load = lambda f: json.loads((work_dir / f).read_text(encoding="utf-8")) if (work_dir / f).is_file() else {}
        vp = load("voice_p.json")
        mid = [r["cue_id"] for r in records if 0.3 <= vp.get(str(r["cue_id"]), -1) <= 0.7
               and editorial._labels(r.get("text") or "").get("speaker")]
        spk_p = speaker_classifier(cfg, engine, video, english, bible_path, work_dir, log, extra=mid) if mid else {}
        hints, vs = editorial.veto_speakers(records, src, hints, vp, spk_p=spk_p,
                                            spk_lo=float(cfg.get("speaker.man_threshold", 0.05)),
                                            spk_hi=float(cfg.get("speaker.woman_threshold", 0.95)))
        for cid, g in xl.items():
            if g.get("speaker"):
                p = vp.get(str(cid))
                if p is not None and (p < 0.2 or p > 0.8) and ("f" if p > 0.8 else "m") != g["speaker"]:
                    continue  # a sure voice beats another language's subtitle (2/2 on the library)
                _, _, to = hints.get(str(cid), "?>?").partition(">")
                hints[str(cid)] = f"{g['speaker']}>{to or '?'}"
        vl = editorial.veto_listeners(records, src, load("addressee.json"), named, voices=voices, voice_p=vp)
        log(f"gender: vetoed {vs} first-person and {vl} feminine 'you' choice(s) of the translator")
        records, changed = editorial.gender_pass(engine, records, hints, src, log, listener=listener,
                                                 listeners=named)
        mixed = editorial.fix_mixed_you(records)
        if mixed:
            log(f"gender: {mixed} line(s) with a 'you' pronoun and a participle of the other gender made consistent")
        records, left = editorial.leftover_pass(engine, records, src, log, names=bible.get("names_he"))
        records, spelt = editorial.spell_pass(engine, records, src, log, names=bible.get("names_he"))
        changed += spelt
        # review / gender rewrites run under a JSON grammar too: a raw " can cut their lines
        records, recut = editorial.truncation_pass(engine, records, src, log, names=bible.get("names_he"))
        changed += recut
        for r in records:
            en = src.get(r["cue_id"], "")
            r["text"] = editorial.speaker_dashes(editorial.reflow(editorial.human_habits(editorial.fix_final_letters(
                editorial.fix_name_variants(r.get("text") or "", en, bible.get("names_he"))), en)), en)
        changed += left
        rules = learn.rules_for(memory.series_key(video, cfg.get("memory.series_roots", ["TV Shows"])))
        if rules:
            fixed = 0
            for r in records:
                new = learn.apply_rules(rules, r.get("text") or "")
                if new != r.get("text"):
                    r["text"], fixed = new, fixed + 1
            if fixed:
                log(f"learn: {fixed} line(s) follow the user's own corrections of this series")
        before = {r["cue_id"]: r.get("text") for r in json.loads(translations.read_text(encoding="utf-8"))}
        if changed or any(before.get(r["cue_id"]) != r.get("text") for r in records):
            # also when only the habits / reflow / dashes changed something
            translations = work_dir / "translations.gender.json"
            translations.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    output = video.with_suffix("").with_name(
        video.with_suffix("").name + cfg.get("subtitles.output_suffix", ".he.srt"))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output = Path(output_dir) / output.name
    staged = work_dir / "output.he.srt"
    build = ["build", "--manifest", str(work_dir / "manifest.json"),
             "--translations", str(translations), "--output", str(staged)]
    try:
        _run_script(cfg, build + ["--strict"], log)
    except PipelineError as exc:
        log(f"build: strict check reported warnings, building anyway ({str(exc)[:160]})")
        _run_script(cfg, build, log)
    audit = _run_script(cfg, ["audit", "--source", str(english),
                              "--target", str(staged),
                              "--sdh-mode", sdh_mode,
                              "--report", str(work_dir / "audit.json")]
                        + (["--cleanup-decisions", str(decisions)] if decisions else []), log)
    log(f"audit: {json.dumps(audit, ensure_ascii=False)[:400]}")
    if cfg.get("subtitles.drop_interjections", True):
        # after the audit: the skill's audit wants every source cue in the target
        man = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        dropped = _drop_interjections(staged, {c["index"]: c.get("clean_text") or "" for c in man["cues"]})
        if dropped:
            log(f"build: dropped {dropped} interjection-only cue(s) (Hmm., Oh., Uh-huh.) as Israeli subtitles do")
    junk = _drop_untranslated(staged, transcript)
    if junk:
        log(f"build: dropped {junk} cue(s) with no Hebrew (untranslatable ASR gibberish) or only a sound tag")

    shutil.copy2(staged, output)
    log(f"written: {output}")
    if output_dir is None:
        try:  # a copy of what we wrote, to learn from the user's later edits of it
            learn.remember_output(cfg, video, english, output)
        except Exception as exc:  # noqa: BLE001 - learning never fails a translation
            log(f"learn: could not keep a copy of the output ({exc})")
    return output


def _named_in(name: str, english: str) -> bool:
    """A name from the series memory is a name here too: capitalised in mid-sentence at least
    once and never a lower-case word ("Will you..." / "I will" is not Will)."""
    if re.search(r"(?<![A-Za-z])" + re.escape(name.lower()) + r"(?![A-Za-z])", english):
        return False
    return re.search(r"(?<=[a-z,] )" + re.escape(name) + r"(?![A-Za-z])", english) is not None


def translate_video(video: Path, cfg, engine: Engine, log: Log, *,
                    series_id=None, episode_id=None, movie_id=None,
                    force: bool = False, output_dir: Path | None = None,
                    resume_dir: Path | None = None) -> Path:
    """Translate one video. output_dir: write the .he.srt there instead of next to the
    video (tests on library titles must never replace a human subtitle)."""
    video = Path(video)
    if not video.is_file():
        raise PipelineError(f"Video not found: {video}")

    existing = find_hebrew(video, cfg)
    if existing and not force:
        log(f"skip: Hebrew subtitle already present ({existing.name})")
        return existing

    # resume_dir: reuse an interrupted run's work dir; batches already translated are kept
    work_dir = Path(resume_dir) if resume_dir else cfg.work_root / f"{video.stem[:80]}-{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {work_dir}")
    claimed = _claim_gpu(engine, log)

    try:
        english = acquire_english(video, cfg, work_dir, log,
                                  series_id=series_id, episode_id=episode_id,
                                  movie_id=movie_id)

        transcript = english.parent.name in ("asr", "ocr")  # Whisper / OCR: gibberish possible
        english = _repair_sound_tags(english, work_dir, log)
        sdh_mode = cfg.get("subtitles.sdh_mode", "remove")
        bible_path = build_character_bible(cfg, engine, english, work_dir, log,
                                           video=video)
        if cfg.get("scenes.enabled", True):
            add_scene_map(engine, english, bible_path, log)
        voices = voice_hints(cfg, engine, video, english, work_dir, log)
        if engine.name in ("ollama", "mlx"):
            # one Hebrew spelling per name for the whole episode (local models otherwise
            # leave names in Latin letters or spell them differently cue to cue)
            b = json.loads(bible_path.read_text(encoding="utf-8-sig"))
            en_text = english.read_text(encoding="utf-8-sig", errors="replace")
            names = editorial.hebrew_names(engine, b, en_text, log)
            b["names_he"] = {k: v for k, v in ((k, memory.clean_hebrew_name(v)) for k, v in names.items()) if v}
            series = memory.series_key(video, cfg.get("memory.series_roots", ["TV Shows"])) \
                if cfg.get("memory.enabled", True) else None
            mem = memory.load(cfg, series) if series else None
            if mem:
                # one spelling per name for the whole series: the one earlier episodes used most
                # (Bleach had מצומוטו/מצומטו 32/28 and three spellings of Soul Society)
                settled = {en: he for en, he in memory.settled_names(mem).items()
                           if en in b["names_he"] or _named_in(en, en_text)}
                changed = sum(b["names_he"].get(en) != he for en, he in settled.items())
                b["names_he"] = {**b["names_he"], **settled}
                if changed:
                    log(f"names: {changed} spelling(s) as earlier episodes of the series had them")
            if cfg.get("series_terms.enabled", True):
                # names and franchise terms as the series' own human Hebrew subtitles have them
                # (Middle-earth = הארץ התיכונה); they win over the model's transliterations
                from . import series_terms
                b["names_he"] = {**b["names_he"], **series_terms.for_video(video, engine, log=log)}
            rules = learn.rules_for(memory.series_key(video, cfg.get("memory.series_roots", ["TV Shows"])))
            if rules:  # the user's own spelling of a name wins over every table
                b["names_he"] = {k: learn.apply_rules(rules, v) for k, v in b["names_he"].items()}
            if mem and memory.vote_names(mem, b["names_he"], video.stem):
                memory.save(cfg, series, mem)
            bible_path.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")

        prepare_args = [
            "prepare", str(english),
            "--work-dir", str(work_dir),
            # Engines differ: a local 12B needs small batches, Gemini is more accurate
            # with 30 because a whole scene lands in one request.
            "--batch-size", str(cfg.get(f"{engine.name}.batch_size")
                                or cfg.get("subtitles.batch_size", 30)),
            "--context-window", str(cfg.get("subtitles.context_window", 4)),
            "--character-bible", str(bible_path),
            "--sdh-mode", sdh_mode,
        ]
        prep = _run_script(cfg, prepare_args, log)
        manifest_path = work_dir / "manifest.json"
        decisions = None
        if cfg.get("cleanup.auto", True):
            # The skill only strips SDH it recognises; the engine decides the rest.
            decisions = editorial.cleanup_decisions(
                engine, english, json.loads(manifest_path.read_text(encoding="utf-8-sig")),
                work_dir, log)
            if decisions:
                shutil.rmtree(work_dir / "batches", ignore_errors=True)
                prep = _run_script(cfg, prepare_args + ["--cleanup-decisions", str(decisions)], log)
        cue_count = prep.get("cue_count") or prep.get("cues") or "?"
        log(f"prepare: {cue_count} cues, {prep.get('batch_count', '?')} batches")

        max_cues = int(cfg.get("limits.max_cues", 3000))
        if isinstance(cue_count, int) and cue_count > max_cues:
            raise PipelineError(f"Refusing: {cue_count} cues exceeds limit {max_cues}")

        bible = json.loads(bible_path.read_text(encoding="utf-8-sig"))
        batch_files = sorted((work_dir / "batches").glob("batch_*.json"))
        if not batch_files:
            raise PipelineError("prepare produced no batches")

        out_dir = work_dir / "translated_batches"
        out_dir.mkdir(exist_ok=True)
        mlx_translate(cfg, engine, work_dir, sdh_mode, log)  # המודל המאומן, אם הוגדר (mlx_translate.adapter)
        todo = [(i, bf) for i, bf in enumerate(batch_files, 1) if not (out_dir / bf.name).is_file()]
        t0 = time.time()

        def one(job):
            i, bf = job
            batch = json.loads(bf.read_text(encoding="utf-8-sig"))
            started = time.time()
            records = translate_checked(engine, batch, bible, sdh_mode, log)
            (out_dir / bf.name).write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
            return i, time.time() - started

        # שתי בקשות במקביל ל-Ollama (OLLAMA_NUM_PARALLEL=2) כמעט לא מאטות כל אחת מהן
        workers = max(1, int(cfg.get(f"{engine.name}.parallel", 1) or 1))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            for k, (i, took) in enumerate(pool.map(one, todo), 1):
                eta = (time.time() - t0) / k * (len(todo) - k) / 60
                log(f"translate: batch {i}/{len(batch_files)} in {took:.0f}s (eta {eta:.0f} min)")

        return finish_translation(cfg, engine, video, english, work_dir, bible, bible_path, voices,
                                  sdh_mode, decisions, log, output_dir, transcript=transcript)
    finally:
        if claimed:
            TRANSLATING_LOCK.unlink(missing_ok=True)  # training resumes on its own
        if not cfg.get("limits.keep_work_dirs", False):
            shutil.rmtree(work_dir, ignore_errors=True)


def notify_rescan(cfg, log: Log, *, series_id=None, movie_id=None) -> None:
    import urllib.error
    import urllib.request

    jobs = []
    if series_id is not None and cfg.get("sonarr.rescan_after", True):
        jobs.append(("sonarr", {"name": "RescanSeries", "seriesId": series_id}))
    if movie_id is not None and cfg.get("radarr.rescan_after", True):
        jobs.append(("radarr", {"name": "RescanMovie", "movieIds": [movie_id]}))
    for service, body in jobs:
        base = cfg.get(f"{service}.url", "").rstrip("/")
        key = cfg.secret(f"{service}.api_key_env")
        if not base or not key:
            continue
        try:
            req = urllib.request.Request(
                f"{base}/api/v3/command",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Api-Key": key},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=30).read()
            log(f"{service}: rescan requested")
        except Exception as exc:  # noqa: BLE001
            log(f"{service}: rescan failed ({exc})")
