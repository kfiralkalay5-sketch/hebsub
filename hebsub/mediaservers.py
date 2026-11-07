"""After a new .he.srt is written next to a video, tell the library apps so it shows up at once:
Sonarr / Radarr rescan the series or movie, Plex scans the video's folder, Jellyfin / Emby are told
the folder changed. Every part is optional (configured or not) and best effort: a failure is
logged, never raised, because the subtitle itself is already in place.

The apps often run in Docker and see the library under their own paths (/tv, /data/media). The
folder is translated with the app's own path_map when one is given ([[app path, local path], ...]),
else by the library folder names the app reports (local .../TV Shows/X/Season 1 = /data/TV Shows/X/
Season 1 when Plex has a library at /data/TV Shows). When neither works, the whole library section
is rescanned instead.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath, PureWindowsPath

EPISODE = re.compile(r"[Ss]\d{1,2}[Ee]\d{1,3}")
SEASON_DIR = re.compile(r"^(season\s*\d+|specials|s\d{1,2})$", re.I)


def _http(url: str, headers: dict, body: dict | None = None, method: str | None = None, timeout: int = 30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = dict(headers, **({"Content-Type": "application/json"} if data is not None else {}))
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        return None


def _parts(path: str) -> tuple[str, ...]:
    """Path components of a path from any OS (a Plex on Windows reports D:\\TV, one in Docker /tv)."""
    pure = PureWindowsPath(path) if ("\\" in path or re.match(r"^[A-Za-z]:", path)) else PurePosixPath(path)
    return pure.parts


def _join(root: str, rest: tuple[str, ...]) -> str:
    sep = "\\" if ("\\" in root or re.match(r"^[A-Za-z]:", root)) else "/"
    return sep.join([root.rstrip("\\/")] + list(rest))


def server_path(local: str | Path, roots: list[str], path_map: list | None = None) -> str | None:
    """`local` as the app sees it: an explicit [app path, local path] pair first, else the app's
    library root whose last folder name appears in the local path."""
    local = str(local)
    for pair in path_map or []:
        if len(pair) == 2 and local.startswith(pair[1]):
            return _join(pair[0], tuple(p for p in re.split(r"[\\/]", local[len(pair[1]):]) if p))
    lp = _parts(local)
    for root in roots:
        name = _parts(root)[-1] if _parts(root) else ""
        if name and name in lp:
            i = len(lp) - 1 - lp[::-1].index(name)  # the last occurrence: the library folder itself
            return _join(root, lp[i + 1:])
    return None


def _kind(video: Path) -> str:
    return "show" if EPISODE.search(video.name) else "movie"


def _item_folder(video: Path) -> Path:
    """The series folder for an episode in Show/Season 1/, else the video's own folder."""
    return video.parent.parent if SEASON_DIR.match(video.parent.name) else video.parent


def arr_rescan(cfg, video: Path, log) -> None:
    """RescanSeries / RescanMovie for the item whose folder holds the video (matched by folder name,
    so it works whatever paths Sonarr/Radarr see in their containers)."""
    service = "sonarr" if _kind(video) == "show" else "radarr"
    base, key = (cfg.get(f"{service}.url") or "").rstrip("/"), cfg.secret(f"{service}.api_key_env")
    if not base or not key or not cfg.get(f"{service}.rescan_after", True):
        return
    folder = _item_folder(video).name
    try:
        items = _http(f"{base}/api/v3/{'series' if service == 'sonarr' else 'movie'}", {"X-Api-Key": key}) or []
        match = next((i for i in items if _parts(i.get("path") or "x")[-1] == folder), None)
        if not match:
            log(f"{service}: no item for folder '{folder}', no rescan")
            return
        body = ({"name": "RescanSeries", "seriesId": match["id"]} if service == "sonarr"
                else {"name": "RescanMovie", "movieIds": [match["id"]]})
        _http(f"{base}/api/v3/command", {"X-Api-Key": key}, body)
        log(f"{service}: rescan requested ({match.get('title', folder)})")
    except Exception as exc:  # noqa: BLE001
        log(f"{service}: rescan failed ({str(exc)[:120]})")


def plex_refresh(cfg, video: Path, log) -> None:
    """Partial scan of the video's folder (GET /library/sections/<id>/refresh?path=...), or of the
    whole matching library when the folder cannot be placed."""
    base, token = (cfg.get("plex.url") or "").rstrip("/"), cfg.secret("plex.token_env")
    if not base or not token:
        return
    hdr = {"X-Plex-Token": token, "Accept": "application/json"}
    try:
        dirs = ((_http(f"{base}/library/sections", hdr) or {}).get("MediaContainer") or {}).get("Directory") or []
        want = _kind(video)
        for d in dirs:
            roots = [loc.get("path", "") for loc in d.get("Location") or []]
            path = server_path(video.parent, roots, cfg.get("plex.path_map"))
            if path:
                q = urllib.parse.urlencode({"path": path})
                _http(f"{base}/library/sections/{d['key']}/refresh?{q}", hdr)
                log(f"plex: scan requested for {path}")
                return
        for d in dirs:
            if d.get("type") == want:
                _http(f"{base}/library/sections/{d['key']}/refresh", hdr)
                log(f"plex: folder not matched, library '{d.get('title')}' scan requested")
    except Exception as exc:  # noqa: BLE001
        log(f"plex: refresh failed ({str(exc)[:120]})")


def jellyfin_refresh(cfg, video: Path, log) -> None:
    """Jellyfin and Emby: POST /Library/Media/Updated for the folder, else a full library scan."""
    base, key = (cfg.get("jellyfin.url") or "").rstrip("/"), cfg.secret("jellyfin.api_key_env")
    if not base or not key:
        return
    hdr = {"X-Emby-Token": key, "Authorization": f'MediaBrowser Token="{key}"'}
    try:
        folders = _http(f"{base}/Library/VirtualFolders", hdr) or []
        roots = [loc for f in folders for loc in f.get("Locations") or []]
        path = server_path(video.parent, roots, cfg.get("jellyfin.path_map"))
        if path:
            _http(f"{base}/Library/Media/Updated", hdr, {"Updates": [{"Path": path, "UpdateType": "Modified"}]})
            log(f"jellyfin: update sent for {path}")
        else:
            _http(f"{base}/Library/Refresh", hdr, method="POST")
            log("jellyfin: folder not matched, library scan requested")
    except Exception as exc:  # noqa: BLE001
        log(f"jellyfin: refresh failed ({str(exc)[:120]})")


def after_write(cfg, video: Path, log) -> None:
    for step in (arr_rescan, plex_refresh, jellyfin_refresh):
        step(cfg, Path(video), log)
