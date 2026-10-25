"""Gender hints from the human subtitles of OTHER languages inside the same video.

Polish, Russian and Arabic mark the gender of "I" and "you" in ordinary dialogue (Polish
zrobiłam / zrobiłeś, Russian ты уверена, Arabic تعرفين / اذهبي), and Spanish in a few
adjectives (estás lista). A professional translator already decided those - so when such a
track sits in the file, its decision is copied for the time-aligned English line.

Measured against the human Hebrew of 30 library videos (xling/interim.py): speaker 125 right
/ 0 wrong, addressee 231 / 3 (98.7%), covering 14% of the lines that need a gender. The
translator alone gets the addressee right ~76% of the time.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

LANGS = ("pol", "rus", "ara", "spa")
TEXT_CODECS = ("subrip", "srt", "ass", "ssa", "mov_text", "webvtt")


def _ts(t: str) -> float:
    h, m, s = t.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace")):
        lines = block.strip().splitlines()
        for k, line in enumerate(lines):
            m = re.match(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", line)
            if m:
                text = re.sub(r"<[^>]+>|\{[^}]*\}", "", " ".join(lines[k + 1:]))
                out.append((_ts(m.group(1)), _ts(m.group(2)), text))
                break
    return out


def extract(video: Path, cfg, work_dir: Path, log) -> dict[str, Path]:
    """Pull the Polish / Russian / Arabic / Spanish text tracks out of the video in one pass."""
    have = {l: work_dir / f"xling.{l}.srt" for l in LANGS if (work_dir / f"xling.{l}.srt").is_file()}
    if have:
        return have
    ffprobe = cfg.get("ffmpeg.probe_binary", "ffprobe")
    ffmpeg = cfg.get("ffmpeg.binary", "ffmpeg")
    try:
        probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "s", "-show_entries",
                                "stream=index,codec_name:stream_tags=language,title:stream_disposition=forced",
                                "-of", "json", str(video)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=True)
        streams = json.loads(probe.stdout).get("streams", [])
    except Exception as exc:  # noqa: BLE001
        log(f"xling: ffprobe failed ({exc})")
        return {}
    pick = {}
    for st in streams:
        tags = {k.lower(): str(v).lower() for k, v in (st.get("tags") or {}).items()}
        lang = tags.get("language", "")
        forced = (st.get("disposition") or {}).get("forced") or "forced" in tags.get("title", "")
        if lang in LANGS and st.get("codec_name") in TEXT_CODECS and not forced and lang not in pick:
            pick[lang] = st["index"]
    if not pick:
        return {}
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(video)]
    for lang, idx in pick.items():
        cmd += ["-map", f"0:{idx}", "-c:s", "srt", str(work_dir / f"xling.{lang}.srt")]
    try:
        subprocess.run(cmd, capture_output=True, timeout=3600)
    except Exception as exc:  # noqa: BLE001
        log(f"xling: extraction failed ({exc})")
    got = {l: work_dir / f"xling.{l}.srt" for l in pick if (work_dir / f"xling.{l}.srt").is_file()}
    log(f"xling: {', '.join(sorted(got)) or 'no'} track(s) for gender hints")
    return got


# ---------------------------------------------------------------- per-language rules
def _put(r: dict, slot: str, g: str) -> None:
    r[slot] = None if r.get(slot) not in (None, g) else g


_POL_F = {"sama", "pewna", "gotowa", "zła", "chora", "głodna", "zmęczona", "szczęśliwa", "wolna", "zajęta",
          "pijana", "przerażona", "zakochana"}
_POL_M = {"sam", "pewien", "pewny", "gotowy", "zły", "chory", "głodny", "zmęczony", "szczęśliwy", "wolny",
          "zajęty", "pijany", "przerażony", "zakochany"}


def g_pol(t: str) -> dict:
    t, r = t.lower(), {}
    if re.search(r"\b\w+łam\b|\b\w+łabym\b", t): _put(r, "speaker", "f")
    if re.search(r"\b\w+łem\b|\b\w+łbym\b", t): _put(r, "speaker", "m")
    if re.search(r"\b\w+łaś\b|\b\w+łabyś\b", t): _put(r, "you", "f")
    if re.search(r"\b\w+łeś\b|\b\w+łbyś\b", t): _put(r, "you", "m")
    for verb, slot in (("jestem", "speaker"), ("jesteś", "you")):
        for m in re.finditer(verb + r"\s+(?:taka\s+|taki\s+|bardzo\s+|naprawdę\s+|już\s+|nie\s+)?(\w+)", t):
            w = m.group(1)
            if w in _POL_F or re.fullmatch(r"\w+(?:na|ta|wa|ła|ska|cka)", w):
                _put(r, slot, "f")
            elif w in _POL_M or re.fullmatch(r"\w+(?:ny|ty|wy|ły|ski|cki)", w):
                _put(r, slot, "m")
    return {k: v for k, v in r.items() if v}


_RUS_F = {"должна", "рада", "права", "готова", "уверена", "сама", "одна", "согласна", "беременна", "занята", "свободна"}
_RUS_M = {"должен", "рад", "прав", "готов", "уверен", "сам", "один", "согласен", "занят", "свободен"}


def g_rus(t: str) -> dict:
    t, r = t.lower(), {}
    for pron, slot in (("я", "speaker"), ("ты", "you")):
        for m in re.finditer(r"\b" + pron + r"\s+(?:не\s+|уже\s+|так\s+|очень\s+|просто\s+|тоже\s+)?(\w+)", t):
            w = m.group(1)
            if re.fullmatch(r"\w+ла(?:сь)?", w) or w in _RUS_F:
                _put(r, slot, "f")
            elif re.fullmatch(r"\w+л(?:ся)?", w) or w in _RUS_M:
                _put(r, slot, "m")
    return {k: v for k, v in r.items() if v}


_AR_PAIRS = [("آسف", "آسفة"), ("متأكد", "متأكدة"), ("سعيد", "سعيدة"), ("مستعد", "مستعدة"), ("خائف", "خائفة"),
             ("جاهز", "جاهزة"), ("متعب", "متعبة"), ("مشغول", "مشغولة"), ("واثق", "واثقة"), ("قلق", "قلقة"),
             ("غاضب", "غاضبة"), ("مسرور", "مسرورة"), ("ممتن", "ممتنة")]
_AR_IMP_F = {"اذهبي", "انظري", "اسمعي", "توقفي", "تعالي", "اجلسي", "انتظري", "قولي", "اهدئي", "ابقي", "اخرجي",
             "ادخلي", "خذي", "اعطيني", "ساعديني", "صدقيني", "اتركيني", "افعلي", "اركضي", "اصمتي", "كفي", "ارجعي",
             "عودي", "اسرعي", "انهضي", "قفي", "تذكري", "اخبريني", "دعيني", "احذري", "ثقي", "استمعي", "ابتعدي"}
_AR_NOUNS_TIN = {"تمرين", "تخمين", "تأمين", "تدوين", "تعيين", "تكوين", "تلقين", "تمكين", "تحسين", "تزيين", "تموين",
                 "تخزين", "تلوين", "تبيين", "تسكين", "تطمين", "تهوين"}


def g_ara(t: str) -> dict:
    r = {}
    ws = re.findall(r"[ء-ي]+", re.sub(r"[ً-ْ]", "", t))
    for i, w in enumerate(ws):
        nxt = [x for x in ws[i + 1:i + 3] if x not in ("لست", "لا", "جدا", "حقا", "فقط")]
        n = nxt[0] if nxt else ""
        if w in ("أنا", "انا", "إنني", "انني"):
            if any(n == f for _, f in _AR_PAIRS) or (n.endswith("ة") and len(n) > 3 and not n.startswith("ال")):
                _put(r, "speaker", "f")
            elif any(n == m for m, _ in _AR_PAIRS):
                _put(r, "speaker", "m")
        if w in ("أنت", "انت") and any(n == f for _, f in _AR_PAIRS):
            _put(r, "you", "f")
        bare = re.sub(r"^(?:و|ف|س|ل)", "", w)
        if (re.fullmatch(r"(?:و|ف|س|ل)?ت[ء-ي]{2,}ين", w) and bare not in _AR_NOUNS_TIN) or \
                re.sub(r"^(?:و|ف)", "", w) in _AR_IMP_F:
            _put(r, "you", "f")
    return {k: v for k, v in r.items() if v}


def g_spa(t: str) -> dict:
    t, r = t.lower(), {}
    f_adj = r"(?:lista|cansada|segura|loca|enfadada|preocupada|casada|embarazada|equivocada|asustada|emocionada)"
    m_adj = r"(?:listo|cansado|seguro|loco|enfadado|preocupado|casado|equivocado|asustado|emocionado)"
    if re.search(r"\best[áa]s\s+(?:muy\s+)?" + f_adj + r"\b", t): _put(r, "you", "f")
    if re.search(r"\best[áa]s\s+(?:muy\s+)?" + m_adj + r"\b", t): _put(r, "you", "m")
    if re.search(r"\bestoy\s+(?:muy\s+)?" + f_adj + r"\b", t): _put(r, "speaker", "f")
    if re.search(r"\bestoy\s+(?:muy\s+)?" + m_adj + r"\b", t): _put(r, "speaker", "m")
    return {k: v for k, v in r.items() if v}


RULES = {"pol": g_pol, "rus": g_rus, "ara": g_ara, "spa": g_spa}


def hints(tracks: dict[str, Path], manifest: dict) -> dict[int, dict]:
    """{cue_id: {"speaker": m|f, "you": m|f}} for one-speaker English lines; languages that
    disagree cancel each other."""
    subs = {l: _parse(p) for l, p in tracks.items()}
    out = {}
    for c in manifest.get("cues", []):
        en = c.get("clean_text") or ""
        if not en.strip() or re.match(r"\s*-", en) or "\n-" in en:
            continue
        a, b = _ts(c["start"]), _ts(c["end"])
        votes: dict[str, set] = {}
        for lang, cues in subs.items():
            txt = " ".join(t for s, e, t in cues if min(b, e) - max(a, s) > 0.3 * min(b - a, e - s))
            if not txt or re.search(r"(?:^|\s)[-–]\s*\S.*\s[-–]\s*\S", txt):  # two speakers there
                continue
            for slot, g in RULES[lang](txt).items():
                votes.setdefault(slot, set()).add(g)
        got = {slot: next(iter(gs)) for slot, gs in votes.items() if len(gs) == 1}
        if got:
            out[c["index"]] = got
    return out
