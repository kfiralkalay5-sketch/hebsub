"""`hebsub watch`: a background translator for any OS.

Every few minutes: new downloads from Sonarr / Radarr (import history) are translated
first; when there are none, the rest of the library (videos without a Hebrew subtitle)
is translated one title at a time. A title without English subtitles is retried every
6 hours for a week (Bazarr usually finds them a day or two after an episode airs).
State is kept in watch_state.json in the data folder, so a restart continues where it
stopped. Run it as a service: see the README (Task Scheduler / launchd / systemd).
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .paths import HOME

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".ts"}
STATE = HOME / "watch_state.json"


def _log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    try:
        with (HOME / "watch.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"done": [], "failed": {}, "no_english": {}, "history": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def new_downloads(cfg: Config, st: dict) -> list[str]:
    """Imported files since the last poll (first run: the last 2 days), as local paths."""
    out = []
    since = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    for name in ("sonarr", "radarr"):
        url, key = cfg.get(f"{name}.url"), cfg.secret(f"{name}.api_key_env")
        if not url or not key:
            continue
        try:
            req = urllib.request.Request(url.rstrip("/") + "/api/v3/history?page=1&pageSize=100&sortKey=date"
                                         "&sortDirection=descending&eventType=3", headers={"X-Api-Key": key})
            with urllib.request.urlopen(req, timeout=60) as r:
                recs = json.loads(r.read().decode("utf-8")).get("records", [])
        except Exception as exc:  # noqa: BLE001 - try again next round
            _log(f"{name} unreachable: {str(exc)[:80]}")
            continue
        last = st["history"].get(name, 0)
        for r in sorted(recs, key=lambda r: r["id"]):
            if r["id"] <= last or (not last and r.get("date", "") < since):
                continue
            path = cfg.map_path((r.get("data") or {}).get("importedPath") or "")
            if path:
                out.append(path)
        if recs:
            st["history"][name] = max([last] + [r["id"] for r in recs])
    return out


def library_roots(cfg: Config, extra: list[str]) -> list[Path]:
    roots = [Path(p) for p in extra] + [Path(p) for p in (cfg.get("watch.roots") or [])]
    roots += [Path(pair[1]) for pair in (cfg.get("path_map") or []) if len(pair) == 2]
    seen, out = set(), []
    for r in roots:
        if str(r) not in seen:
            seen.add(str(r))
            out.append(r)
    return out


def backlog(cfg: Config, roots: list[Path], st: dict, limit: int = 200) -> list[str]:
    from .subsource import find_hebrew
    skip = set(st["done"]) | {p for p, n in st["failed"].items() if n >= 2}
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in VIDEO_EXT and p.is_file() and "@" not in str(p) and str(p) not in skip:
                if not find_hebrew(p, cfg):
                    out.append(str(p))
                    if len(out) >= limit:
                        return out
    return out


def _due(st: dict, path: str) -> bool:
    """No English yet: retry every 6 h, give up after 7 days."""
    first, last = st["no_english"].get(path, [0, 0])
    now = time.time()
    return not first or (now - last > 6 * 3600 and now - first < 7 * 86400)


def translate_one(cfg: Config, engine, path: str, st: dict) -> None:
    from .pipeline import translate_video
    from .subsource import SubtitleNotFound
    _log(f"translating {Path(path).name}")
    try:
        out = translate_video(Path(path), cfg, engine, _log)
        st["done"].append(path)
        st["no_english"].pop(path, None)
        _log(f"OK -> {out}")
    except SubtitleNotFound:
        first = st["no_english"].get(path, [time.time(), 0])[0]
        st["no_english"][path] = [first, time.time()]
        _log("no English subtitles yet, will retry later")
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the service
        st["failed"][path] = st["failed"].get(path, 0) + 1
        _log(f"FAILED ({st['failed'][path]}x): {str(exc)[:200]}")
    save_state(st)


def run(config_path: str | None = None, folders: list[str] | None = None, interval: int = 300) -> int:
    from .engines import build_engine
    cfg = Config.load(config_path)
    engine = build_engine(cfg)
    st = load_state()
    roots = library_roots(cfg, folders or [])
    _log(f"watch started: engine {engine.name}, {len(roots)} library folder(s)")
    pending: list[str] = []
    last_scan = 0.0
    while True:
        fresh = [p for p in new_downloads(cfg, st) if p not in st["done"]]
        save_state(st)
        if fresh:
            _log(f"{len(fresh)} new download(s)")
        # new downloads first, then the library; English-less titles only when due
        if time.time() - last_scan > 6 * 3600:
            pending = backlog(cfg, roots, st)
            last_scan = time.time()
            _log(f"library: {len(pending)} title(s) without Hebrew")
            if cfg.get("learn.enabled", True):
                try:  # your edits of our subtitles, new human Hebrew in the library (vocabulary, names)
                    from .learn import run as learn
                    learn(roots, cfg, _log, limit=int(cfg.get("learn.videos_per_round", 50)))
                except Exception as exc:  # noqa: BLE001 - learning never stops the service
                    _log(f"learn failed: {exc}")
        queue = [p for p in fresh + pending if _due(st, p) and p not in st["done"]
                 and st["failed"].get(p, 0) < 2]
        if not queue:
            time.sleep(interval)
            continue
        path = queue[0]
        if not Path(path).is_file():  # network share not mounted yet
            _log(f"waiting for {Path(path).parent}")
            time.sleep(interval)
            continue
        translate_one(cfg, engine, path, st)
        if path in pending:
            pending.remove(path)
