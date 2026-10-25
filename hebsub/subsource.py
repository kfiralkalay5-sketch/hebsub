"""Acquire an English .srt for a video file.

Strategy, in order:
  1. An English subtitle file already sitting next to the video.
  2. Ask Bazarr to fetch one, then poll the folder for it to appear.
  3. Extract an embedded English subtitle track with ffmpeg.

Bazarr writes subtitles next to the media file, so step 2 only has to *trigger*
Bazarr and then watch the filesystem. That keeps us independent of the exact shape
of Bazarr's API responses, which differ between versions.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HEBREW_HINTS = ("he", "heb", "hebrew", "iw")


class SubtitleNotFound(RuntimeError):
    pass


def _candidates(video: Path, suffixes: list[str]) -> list[Path]:
    stem = video.with_suffix("").name
    out: list[Path] = []
    for suffix in suffixes:
        direct = video.parent / f"{stem}{suffix}"
        if direct.is_file() and direct.stat().st_size > 200:
            out.append(direct)
    return out


def find_hebrew(video: Path, cfg) -> Path | None:
    """Return an existing Hebrew subtitle for this video, if any - also an untagged '.srt'
    whose text is Hebrew (thousands of library titles have one, many in cp1255)."""
    suffixes = cfg.get("subtitles.hebrew_suffixes", [".he.srt"])
    found = _candidates(video, suffixes)
    if found:
        return found[0]
    bare = _candidates(video, [".srt"])
    if bare and _hebrew_text(bare[0]):
        return bare[0]
    if "hebsub" in video.name.lower().replace(".", "").replace(" ", ""):
        return video  # "HebSub" releases have the Hebrew burned into the picture
    if cfg.get("subtitles.check_embedded_hebrew", True) and embedded_hebrew(video, cfg):
        return video  # a full (not forced) Hebrew track inside the file; Plex shows it
    return None


def embedded_hebrew(video: Path, cfg) -> bool:
    """True when the video carries a full Hebrew subtitle track (HebSub releases and many
    remuxes do): translating those would duplicate what the viewer already has."""
    ffprobe = cfg.get("ffmpeg.probe_binary", "ffprobe")
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "s",
             "-show_entries", "stream=index:stream_tags=language,title:stream_disposition=forced",
             "-of", "json", str(video)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=True)
        streams = json.loads(probe.stdout).get("streams", [])
    except Exception:  # noqa: BLE001 - unreadable file: let the pipeline decide later
        return False
    for st in streams:
        tags = {k.lower(): str(v).lower() for k, v in (st.get("tags") or {}).items()}
        if tags.get("language") in ("heb", "he", "iw") or "hebrew" in tags.get("title", "") \
                or "עברית" in tags.get("title", ""):
            if not (st.get("disposition") or {}).get("forced") and "forced" not in tags.get("title", ""):
                return True
    return False


def _hebrew_text(path: Path) -> bool:
    """Hebrew letters outnumber Latin ones, reading the file as UTF-8 or else as cp1255."""
    try:
        raw = path.read_bytes()[:20000]
    except OSError:
        return False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1255", "replace")
    hebrew = sum(1 for ch in text if "\u05d0" <= ch <= "\u05ea")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return hebrew > latin


def find_english(video: Path, cfg) -> Path | None:
    """Return an existing English subtitle, skipping anything that looks Hebrew."""
    suffixes = cfg.get("subtitles.english_suffixes", [".en.srt"])
    for path in _candidates(video, suffixes):
        lowered = path.name.lower()
        # ".srt" with no language tag is ambiguous - accept it only if it is not Hebrew
        if any(f".{h}." in lowered for h in HEBREW_HINTS):
            continue
        if _looks_hebrew(path) or not _looks_english(path):
            continue
        return path
    return None


def _looks_english(path: Path) -> bool:
    """An untagged .srt may be Hebrew in a legacy code page (cp1255 read as cp1252 gives
    'àúä ìà'): require mostly ASCII letters and common English words."""
    import re
    try:
        raw = path.read_bytes()[:40000]
    except OSError:
        return False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1255", "replace")
    text = re.sub(r"<[^>]+>|\{[^}]*\}|\d+:\d+:\d+[,.]\d+", " ", text)
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 200:
        return False
    ascii_share = sum(ch.isascii() for ch in letters) / len(letters)
    common = len(re.findall(r"\b(?:the|you|and|to|is|it|what|that|I)\b", text, re.I))
    return ascii_share >= 0.85 and common >= 20


def _looks_hebrew(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:20000].decode("utf-8", "replace")
    except OSError:
        return False
    hebrew = sum(1 for ch in sample if "֐" <= ch <= "׿")
    latin = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    return hebrew > latin


# ---------------------------------------------------------------------------
# Bazarr
# ---------------------------------------------------------------------------

def _bazarr_request(cfg, method: str, path: str, params: dict | None = None) -> bytes:
    base = cfg.get("bazarr.url", "").rstrip("/")
    key = cfg.secret("bazarr.api_key_env")
    if not base or not key:
        raise RuntimeError("Bazarr URL or API key not configured")
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, method=method, headers={"X-API-KEY": key})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def bazarr_reachable(cfg) -> bool:
    try:
        _bazarr_request(cfg, "GET", "/api/system/status")
        return True
    except Exception:
        return False


def trigger_bazarr(cfg, *, series_id: int | None, episode_id: int | None,
                   movie_id: int | None, log) -> None:
    """Best-effort: ask Bazarr to search English subtitles for this item.

    Any failure here is non-fatal - we still poll the folder and fall back to ffmpeg.
    """
    attempts: list[tuple[str, str, dict]] = []
    if episode_id is not None:
        attempts.append(("PATCH", "/api/episodes/subtitles",
                         {"seriesid": series_id, "episodeid": episode_id,
                          "language": "en", "forced": "False", "hi": "False"}))
        attempts.append(("PATCH", "/api/episodes", {"seriesid": series_id,
                                                    "episodeid": episode_id,
                                                    "action": "search-missing"}))
    if movie_id is not None:
        attempts.append(("PATCH", "/api/movies/subtitles",
                         {"radarrid": movie_id, "language": "en",
                          "forced": "False", "hi": "False"}))
        attempts.append(("PATCH", "/api/movies", {"radarrid": movie_id,
                                                  "action": "search-missing"}))
    for method, path, params in attempts:
        try:
            _bazarr_request(cfg, method, path, params)
            log(f"bazarr: triggered {method} {path}")
            return
        except Exception as exc:  # noqa: BLE001 - deliberately tolerant
            log(f"bazarr: {method} {path} failed ({exc}); trying next form")
    log("bazarr: could not trigger a search, will poll and fall back to ffmpeg")


def resolve_bazarr_ids(cfg, video: Path) -> dict:
    """Find Bazarr's ids for a file by its path (Bazarr sees the same NAS paths)."""
    want = str(video)
    try:
        for m in json.loads(_bazarr_request(cfg, "GET", "/api/movies", {"start": 0, "length": -1})).get("data", []):
            if cfg.map_path(m.get("path") or "") == want:
                return {"movie_id": m["radarrId"]}
        for sr in json.loads(_bazarr_request(cfg, "GET", "/api/series", {"start": 0, "length": -1})).get("data", []):
            root = cfg.map_path(sr.get("path") or "").rstrip("/") + "/"
            if root != "/" and want.startswith(root):
                eps = json.loads(_bazarr_request(cfg, "GET", "/api/episodes",
                                                 {"seriesid[]": sr["sonarrSeriesId"]})).get("data", [])
                for e in eps:
                    if cfg.map_path(e.get("path") or "") == want:
                        return {"series_id": e["sonarrSeriesId"], "episode_id": e["sonarrEpisodeId"]}
    except Exception:  # noqa: BLE001 - lookup is best-effort
        pass
    return {}


def wait_for_english(video: Path, cfg, log) -> Path | None:
    deadline = time.time() + int(cfg.get("bazarr.wait_seconds", 45))
    while time.time() < deadline:
        found = find_english(video, cfg)
        if found:
            log(f"bazarr: picked up {found.name}")
            return found
        time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# ffmpeg embedded track
# ---------------------------------------------------------------------------

def extract_embedded_english(video: Path, cfg, out_dir: Path, log) -> Path | None:
    if not cfg.get("ffmpeg.enabled", True):
        return None
    ffprobe = cfg.get("ffmpeg.probe_binary", "ffprobe")
    ffmpeg = cfg.get("ffmpeg.binary", "ffmpeg")
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "s",
             "-show_entries", "stream=index,codec_name:stream_tags=language,title"
             ":stream_disposition=forced,hearing_impaired",
             "-of", "json", str(video)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
    except Exception as exc:  # noqa: BLE001
        log(f"ffprobe failed: {exc}")
        return None

    text_codecs = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt"}

    def rank(stream):
        """Full dialogue track first, then SDH; 'forced' / signs-and-songs tracks last
        (a forced track holds only the foreign-language lines: 3 cues on Ted Lasso)."""
        tags = {k.lower(): str(v).lower() for k, v in (stream.get("tags") or {}).items()}
        disp = stream.get("disposition") or {}
        title = tags.get("title", "")
        if disp.get("forced") or any(w in title for w in ("forced", "sign", "song", "foreign")):
            return 0
        if disp.get("hearing_impaired") or "sdh" in title or "cc" in title.split():
            return 1
        return 2
    candidates = sorted(
        (s for s in streams if s.get("codec_name") in text_codecs and
         {k.lower(): str(v).lower() for k, v in (s.get("tags") or {}).items()}.get("language", "")
         in ("eng", "en", "english")),
        key=rank, reverse=True)
    if not candidates:
        log("ffmpeg: no embedded English text subtitle track found")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "embedded.en.srt"
    # ffmpeg reads the whole file to pull a subtitle track: a 91 GB 4K remux over the network
    # needs ~20 min, so the time limit grows with the size (~40 MB/s), 10 min at least
    try:
        limit = max(600, int(video.stat().st_size / 40e6))
    except OSError:
        limit = 600
    for chosen in candidates:
        try:
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(video),
                 "-map", f"0:{chosen['index']}", "-c:s", "srt", str(out)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=limit, check=True,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"ffmpeg extraction of track #{chosen['index']} failed: {exc}")
            continue
        cues = out.read_text(encoding="utf-8", errors="replace").count("-->") if out.is_file() else 0
        if cues >= 50 or (cues > 0 and chosen is candidates[-1]):
            log(f"ffmpeg: extracted embedded English track #{chosen['index']} ({cues} cues)")
            return out
        log(f"ffmpeg: English track #{chosen['index']} has only {cues} cues, trying the next one")
    return None


def acquire_english(video: Path, cfg, work_dir: Path, log, *,
                    series_id=None, episode_id=None, movie_id=None) -> Path:
    """Return a path to an English .srt or raise SubtitleNotFound."""
    existing = find_english(video, cfg)
    if existing:
        log(f"english: using existing {existing.name}")
        return existing

    # An embedded track belongs to this exact release, so it is always in sync.
    embedded = extract_embedded_english(video, cfg, work_dir, log)
    if embedded:
        return embedded

    if cfg.get("bazarr.enabled", True):
        if episode_id is None and movie_id is None:
            ids = resolve_bazarr_ids(cfg, video)
            series_id = ids.get("series_id", series_id)
            episode_id, movie_id = ids.get("episode_id"), ids.get("movie_id")
            if not ids:
                log("bazarr: this file is not managed by Sonarr/Radarr")
        if episode_id is not None or movie_id is not None:
            trigger_bazarr(cfg, series_id=series_id, episode_id=episode_id,
                           movie_id=movie_id, log=log)
            found = wait_for_english(video, cfg, log)
            if found:
                return found

    raise SubtitleNotFound(
        f"No English subtitle available for {video.name} "
        "(checked sibling files, embedded tracks, Bazarr)"
    )
