"""`hebsub setup`: interactive configuration for any OS (terminal or PowerShell).

Checks Ollama and the model, finds ffmpeg, and optionally connects TMDB (character
genders), Sonarr / Radarr (new downloads, library paths) and Bazarr (English subtitles).
Every connection is tested on the spot. API keys go to secrets.json in the data folder,
never into config.json. Prompts are in English because many terminals cannot show
Hebrew right-to-left.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from .paths import HOME, save_secret, secret

RECOMMENDED_MODEL = "hf.co/unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M"
SMALL_MODEL = "gemma3:12b"


def ask(question: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    try:
        answer = input(f"{question}{shown}: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def yes(question: str, default: bool = True) -> bool:
    answer = ask(question + (" (Y/n)" if default else " (y/N)")).lower()
    return default if not answer else answer.startswith("y")


def http_json(url: str, headers: dict | None = None, timeout: int = 15):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def test_arr(url: str, key: str) -> tuple[bool, str]:
    """Sonarr or Radarr v3 API reachable with this key?"""
    try:
        st = http_json(url.rstrip("/") + "/api/v3/system/status", {"X-Api-Key": key})
        return True, f"{st.get('appName', 'OK')} {st.get('version', '')}".strip()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:120]


def test_bazarr(url: str, key: str) -> tuple[bool, str]:
    try:
        st = http_json(url.rstrip("/") + "/api/system/status", {"X-API-KEY": key})
        return True, "Bazarr " + str((st.get("data") or {}).get("bazarr_version", "OK"))
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:120]


def test_tmdb(key: str) -> tuple[bool, str]:
    """Both TMDB key kinds work: the v3 API key and the v4 read access token (eyJ...)."""
    try:
        if key.startswith("eyJ"):
            http_json("https://api.themoviedb.org/3/configuration", {"Authorization": f"Bearer {key}"})
        else:
            http_json("https://api.themoviedb.org/3/configuration?" + urllib.parse.urlencode({"api_key": key}))
        return True, "TMDB OK"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:120]


def root_folders(url: str, key: str) -> list[str]:
    try:
        return [r["path"] for r in http_json(url.rstrip("/") + "/api/v3/rootfolder", {"X-Api-Key": key})]
    except Exception:  # noqa: BLE001
        return []


def map_paths(remote_roots: list[str], existing: list) -> list[list[str]]:
    """Sonarr/Radarr see paths inside their own machine or container (/tv, /data/media);
    ask where each one is on this computer (D:\\TV, \\\\NAS\\share\\TV, /Volumes/...)."""
    pairs = [list(p) for p in existing]
    known = {p[0] for p in pairs}
    for root in remote_roots:
        if root in known:
            continue
        local = ask(f"  Where is '{root}' on this computer? (Enter = same path)", root)
        if not Path(local).is_dir():
            print(f"  ! '{local}' is not reachable from here right now (saved anyway)")
        pairs.append([root, local])
    return pairs


def check_ollama(data: dict) -> None:
    host = ask("Ollama URL", data["ollama"].get("host", "http://127.0.0.1:11434"))
    data["ollama"]["host"] = host
    try:
        tags = [m["name"] for m in http_json(host.rstrip("/") + "/api/tags").get("models", [])]
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Ollama is not reachable ({str(exc)[:80]}). Install it from https://ollama.com and run it.")
        return
    print(f"  Ollama OK ({len(tags)} model(s) installed)")
    print(f"  Recommended model: {RECOMMENDED_MODEL} (needs ~20 GB GPU memory or a 32 GB Mac)")
    print(f"  Smaller model for weaker PCs: {SMALL_MODEL} (~10 GB, lower quality)")
    model = ask("Model to use", data["ollama"].get("model") or RECOMMENDED_MODEL)
    data["ollama"]["model"] = model
    if not any(t == model or t.split(":")[0] == model for t in tags):
        if shutil.which("ollama") and yes(f"  '{model}' is not installed. Download it now with 'ollama pull'?"):
            subprocess.run(["ollama", "pull", model])


def check_ffmpeg(data: dict) -> None:
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ff and fp:
        data["ffmpeg"].update({"enabled": True, "binary": ff, "probe_binary": fp})
        print(f"  ffmpeg OK ({ff})")
        return
    hint = {"win32": "winget install Gyan.FFmpeg", "darwin": "brew install ffmpeg"}.get(sys.platform,
                                                                                         "sudo apt install ffmpeg")
    print(f"  ! ffmpeg not found. It is needed to read English subtitles inside video files. Install: {hint}")


def setup_arr(data: dict, name: str, env: str) -> None:
    if not yes(f"Connect {name.capitalize()}?", default=bool(data[name].get("url"))):
        return
    url = ask(f"  {name.capitalize()} URL", data[name].get("url") or ("http://localhost:8989" if name == "sonarr"
                                                                      else "http://localhost:7878"))
    key = ask(f"  {name.capitalize()} API key (Settings > General)", secret(env) or "")
    ok, msg = test_arr(url, key)
    print(("  OK: " if ok else "  ! Failed: ") + msg)
    if not ok and not yes("  Save anyway?", default=False):
        return
    data[name].update({"url": url, "api_key_env": env, "rescan_after": True})
    save_secret(env, key)
    data["path_map"] = map_paths(root_folders(url, key), data.get("path_map") or [])


def setup_bazarr(data: dict) -> None:
    if not yes("Connect Bazarr (downloads English subtitles when a video has none)?",
               default=bool(data["bazarr"].get("enabled"))):
        return
    url = ask("  Bazarr URL", data["bazarr"].get("url") or "http://localhost:6767")
    key = ask("  Bazarr API key (Settings > General > Security)", secret("BAZARR_API_KEY") or "")
    ok, msg = test_bazarr(url, key)
    print(("  OK: " if ok else "  ! Failed: ") + msg)
    if ok or yes("  Save anyway?", default=False):
        data["bazarr"].update({"enabled": True, "url": url, "api_key_env": "BAZARR_API_KEY"})
        save_secret("BAZARR_API_KEY", key)


def setup_tmdb() -> None:
    print("TMDB tells hebsub which characters are men and which are women (better Hebrew gender).")
    print("  Free key: create an account at themoviedb.org > Settings > API.")
    if not yes("Add a TMDB API key?", default=not secret("TMDB_API_KEY")):
        return
    key = ask("  TMDB API key (v3) or read access token", secret("TMDB_API_KEY") or "")
    ok, msg = test_tmdb(key)
    print(("  OK: " if ok else "  ! Failed: ") + msg)
    if ok or yes("  Save anyway?", default=False):
        save_secret("TMDB_API_KEY", key)


def run(config_path: str | None = None) -> int:
    target = Path(config_path) if config_path else HOME / "config.json"
    base = target if target.is_file() else Path(__file__).resolve().parent / "default_config.json"
    data = json.loads(base.read_text(encoding="utf-8"))
    for k in ("ollama", "ffmpeg", "bazarr", "sonarr", "radarr"):
        data.setdefault(k, {})
    print(f"hebsub setup. Settings: {target}\n")
    print("1) Translation model (Ollama)")
    check_ollama(data)
    print("\n2) ffmpeg")
    check_ffmpeg(data)
    print("\n3) Character genders (TMDB)")
    setup_tmdb()
    print("\n4) Sonarr / Radarr (optional: translate new downloads first, find your library)")
    setup_arr(data, "sonarr", "SONARR_API_KEY")
    setup_arr(data, "radarr", "RADARR_API_KEY")
    print("\n5) Bazarr (optional)")
    setup_bazarr(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {target}. API keys are in {HOME / 'secrets.json'}.")
    print("Next: 'hebsub check', then 'hebsub translate <video>' or 'hebsub watch'.")
    return 0
