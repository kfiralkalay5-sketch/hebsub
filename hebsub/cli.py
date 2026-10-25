"""Command line entry points for hebsub.

    hebsub translate "/path/to/TV Shows/Show/Season 1/episode.mkv"
    hebsub translate EPISODE.mkv --engine ollama --force
    hebsub scan "/path/to/TV Shows/Show" --limit 5
    hebsub check
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import Config
from .engines import build_engine
from .pipeline import translate_video
from .subsource import bazarr_reachable, find_english, find_hebrew

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".ts"}


def _log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


def cmd_translate(args) -> int:
    cfg = Config.load(args.config)
    engine = build_engine(cfg, args.engine)
    _log(f"engine: {engine.name} ({cfg.get(engine.name + '.model')})")
    out = translate_video(Path(args.video), cfg, engine, _log, force=args.force)
    _log(f"OK -> {out}")
    return 0


def cmd_scan(args) -> int:
    cfg = Config.load(args.config)
    root = Path(args.root)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1
    missing: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXT or not path.is_file():
            continue
        if find_hebrew(path, cfg):
            continue
        missing.append(path)
    print(f"{len(missing)} video(s) without a Hebrew subtitle under {root}")
    for path in missing[: args.limit or len(missing)]:
        eng = find_english(path, cfg)
        print(f"  {'[en]' if eng else '[  ]'} {path.relative_to(root)}")
    if args.translate and missing:
        engine = build_engine(cfg, args.engine)
        for path in missing[: args.limit or len(missing)]:
            _log(f"--- {path.name}")
            try:
                translate_video(path, cfg, engine, _log)
            except Exception as exc:  # noqa: BLE001
                _log(f"FAILED: {exc}")
    return 0


def cmd_check(args) -> int:
    cfg = Config.load(args.config)
    print(f"config           : {cfg.source}")
    print(f"engine           : {cfg.get('engine')}")
    skill = cfg.skill_dir / "scripts" / "srt_pipeline.py"
    print(f"srt_pipeline.py  : {'OK' if skill.is_file() else 'MISSING'} ({skill})")
    for pair in cfg.get("path_map", []) or []:
        p = Path(pair[1])
        print(f"media path       : {'OK' if p.is_dir() else 'MISSING'} ({p})")
    print(f"bazarr           : {'reachable' if bazarr_reachable(cfg) else 'unreachable'}")
    try:
        engine = build_engine(cfg, args.engine)
        reply = engine.complete(
            'Reply with this exact JSON and nothing else: {"ok": true}')
        print(f"engine test      : OK ({engine.name}) -> {reply.strip()[:80]}")
    except Exception as exc:  # noqa: BLE001
        print(f"engine test      : FAILED -> {exc}")
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hebsub")
    parser.add_argument("--config")
    parser.add_argument("--engine", choices=("gemini", "ollama"),
                        help="override the engine from config")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("translate", help="Translate one video's subtitles")
    t.add_argument("video")
    t.add_argument("--force", action="store_true",
                   help="translate even if a Hebrew subtitle already exists")
    t.set_defaults(func=cmd_translate)

    s = sub.add_parser("scan", help="List videos missing Hebrew subtitles")
    s.add_argument("root")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--translate", action="store_true",
                   help="actually translate what was found")
    s.set_defaults(func=cmd_scan)

    c = sub.add_parser("check", help="Verify configuration and engine connectivity")
    c.set_defaults(func=cmd_check)

    su = sub.add_parser("setup", help="Interactive setup: model, ffmpeg, TMDB, Sonarr/Radarr/Bazarr")
    su.set_defaults(func=lambda a: __import__("hebsub.setup_wizard", fromlist=["run"]).run(a.config))

    w = sub.add_parser("watch", help="Run as a service: new downloads first, then the library")
    w.add_argument("folders", nargs="*", help="library folders to translate (besides Sonarr/Radarr paths)")
    w.add_argument("--interval", type=int, default=300, help="seconds between checks when idle")
    w.set_defaults(func=lambda a: __import__("hebsub.watch", fromlist=["run"]).run(a.config, a.folders, a.interval))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
