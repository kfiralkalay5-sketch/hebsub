"""Who plays whom: character names with gender from TMDB, for gender-correct Hebrew.

The English subtitles often never show whether a character is a man or a woman, and
the model's guess from dialogue was the main source of wrong Hebrew gender. TMDB lists
every character of a film or episode with the actor's gender, which is the character's
gender in nearly all live-action cases.

TMDB ids come from Sonarr/Radarr (same NAS paths), falling back to a title search.
Results are cached per title/episode in ~/hebsub/cast_cache/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from .paths import HOME

API = "https://api.themoviedb.org/3"
CACHE = HOME / "cast_cache"
GENDER = {1: "feminine", 2: "masculine"}
EPISODE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})")
TITLES = re.compile(r"^(dr|mr|mrs|ms|miss|prof|professor|sgt|sergeant|capt|captain|lt|"
                    r"lieutenant|det|detective|agent|officer|nurse|father|sister|uncle|aunt|"
                    r"general|colonel|major|judge|officer|principal|headmaster|headmistress|dean|coach|"
                    r"sheriff|deputy|chief|commander|mayor|senator|president|governor|reverend|pastor|"
                    r"bishop|king|queen|prince|princess|lord|lady|sir|dame|brother|mother|grandma|grandpa|"
                    r"granny|auntie|nanny|madam|madame|count|countess|duke|duchess|emperor|empress)\.?\s+", re.I)
_arr: dict[str, list] = {}
# python.org builds ship without system CAs; use macOS's bundle (as hebsub-cli does)
_SSL = ssl.create_default_context(cafile="/etc/ssl/cert.pem") if Path("/etc/ssl/cert.pem").is_file() else None


def _token() -> str | None:
    return (os.environ.get("TMDB_API_KEY") or "").strip().strip('"').strip("'") or None


def _tmdb(path: str, **params) -> dict:
    tok = _token()
    if not tok:
        raise RuntimeError("TMDB_API_KEY not set")
    headers = {"accept": "application/json"}
    if tok.startswith("eyJ"):  # v4 read access token
        headers["Authorization"] = f"Bearer {tok}"
    else:  # v3 api key
        params["api_key"] = tok
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30, context=_SSL) as r:
        return json.loads(r.read())


def _arr_items(cfg, name: str) -> list:
    """Sonarr series / Radarr movies, fetched once per process."""
    if name not in _arr:
        try:
            key = cfg.secret(f"{name}.api_key_env")
            api = "series" if name == "sonarr" else "movie"
            req = urllib.request.Request(f"{cfg.get(name + '.url').rstrip('/')}/api/v3/{api}",
                                         headers={"X-Api-Key": key or ""})
            with urllib.request.urlopen(req, timeout=60) as r:
                _arr[name] = json.loads(r.read())
        except Exception:  # noqa: BLE001 - fall back to a title search
            _arr[name] = []
    return _arr[name]


def _title_year(folder: str) -> tuple[str, str | None]:
    m = re.search(r"[\(\[](\d{4})[\)\]]", folder)
    title = re.sub(r"[\(\[]\d{4}[\)\]]", "", folder).replace(".", " ").strip(" -_")
    return title, (m.group(1) if m else None)


def identify(video: Path, cfg=None) -> dict | None:
    """{"kind": "tv"|"movie", "id": tmdb id, "season", "episode"} for a library path."""
    parts = video.parts
    if "TV Shows" in parts:
        series = parts[parts.index("TV Shows") + 1]
        m = EPISODE.search(video.name)
        if not m:
            return None
        sid = None
        if cfg is not None:
            for s in _arr_items(cfg, "sonarr"):
                if Path(s.get("path") or "").name == series and s.get("tmdbId"):
                    sid = s["tmdbId"]
        if sid is None:
            title, year = _title_year(series)
            res = _tmdb("/search/tv", query=title, **({"first_air_date_year": year} if year else {}))
            sid = res["results"][0]["id"] if res.get("results") else None
        return {"kind": "tv", "id": sid, "season": int(m.group(1)), "episode": int(m.group(2))} if sid else None
    if "extra_subs" in parts:  # training pairs the user dropped in: "house.S01E05" or a title
        m = EPISODE.search(video.name)
        title = (video.name[: m.start()] if m else video.name).replace(".", " ").strip()
        res = _tmdb("/search/tv" if m else "/search/movie", query=title)
        if not res.get("results"):
            return None
        rid = res["results"][0]["id"]
        return ({"kind": "tv", "id": rid, "season": int(m.group(1)), "episode": int(m.group(2))}
                if m else {"kind": "movie", "id": rid})
    if "Movies" in parts:
        folder = parts[parts.index("Movies") + 1]
        if cfg is not None:
            for mv in _arr_items(cfg, "radarr"):
                if Path(mv.get("path") or "").name == folder and mv.get("tmdbId"):
                    return {"kind": "movie", "id": mv["tmdbId"]}
        title, year = _title_year(folder)
        res = _tmdb("/search/movie", query=title, **({"year": year} if year else {}))
        return {"kind": "movie", "id": res["results"][0]["id"]} if res.get("results") else None
    return None


def _roles(person: dict) -> list[str]:
    if person.get("roles"):  # aggregate credits
        names = [r.get("character", "") for r in person["roles"]]
    else:
        names = [person.get("character", "")]
    out = []
    for n in names:
        for one in re.split(r"\s*/\s*|\s+and\s+", n or ""):
            one = re.sub(r"\((voice|uncredited|archive footage|as .*?)\)", "", one, flags=re.I).strip(" '\"")
            if one and not re.fullmatch(r"(self|himself|herself|narrator)", one, re.I):
                out.append(one)
    return out


def fetch(ident: dict) -> list[dict]:
    """[{name, gender, actor}] for the film or episode (main cast + guests), cached."""
    key = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    cached = CACHE / f"{key}.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    people = []
    if ident["kind"] == "movie":
        people = _tmdb(f"/movie/{ident['id']}/credits").get("cast", [])[:60]
    else:
        try:
            ep = _tmdb(f"/tv/{ident['id']}/season/{ident['season']}/episode/{ident['episode']}/credits")
            people = ep.get("cast", []) + ep.get("guest_stars", [])
        except Exception:  # noqa: BLE001 - episode unknown to TMDB: series cast still helps
            pass
        people += _tmdb(f"/tv/{ident['id']}/aggregate_credits").get("cast", [])[:40]
    out, seen = [], set()
    for p in people:
        g = GENDER.get(p.get("gender"))
        for name in _roles(p):
            if (name, g) in seen:
                continue
            seen.add((name, g))
            out.append({"name": name, "gender": g or "unknown", "actor": p.get("name", "")})
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def cast_for(video: Path, cfg=None) -> list[dict]:
    """Best effort: [] when TMDB is not configured or the title is not found."""
    if not _token():
        return []
    try:
        ident = identify(Path(video), cfg)
        return fetch(ident) if ident else []
    except Exception:  # noqa: BLE001
        return []


def _keys(name: str) -> set[str]:
    """Ways the subtitles may call this character: full name, first, last, with/without title."""
    n = re.sub(r"\s+", " ", re.sub(r"[\"'“”]", "", name)).strip()
    n = re.sub(r"^(the|a|an)\s+", "", n, flags=re.I)  # "The Captain" is called "Captain"
    bare = TITLES.sub("", n)
    words = bare.split()
    keys = {n.lower(), bare.lower()}
    if len(words) > 1:
        keys |= {words[0].lower(), words[-1].lower()}
        m = TITLES.match(n)
        if m:
            keys.add(f"{m.group(1).lower()}. {words[-1].lower()}".replace("..", "."))
            keys.add(f"{m.group(1).lower()} {words[-1].lower()}")
    return {k for k in keys if len(k) > 1}


def gender_index(cast: list[dict]) -> dict[str, str]:
    """name key -> gender, dropping keys shared by people of different genders."""
    idx: dict[str, set] = {}
    for c in cast:
        if c["gender"] == "unknown":
            continue
        for k in _keys(c["name"]):
            idx.setdefault(k, set()).add(c["gender"])
    return {k: next(iter(g)) for k, g in idx.items() if len(g) == 1}


def lookup(index: dict[str, str], name: str) -> str | None:
    for k in sorted(_keys(name), key=len, reverse=True):
        if k in index:
            return index[k]
    return None


def in_text(cast: list[dict], text: str, limit: int = 30) -> list[dict]:
    """Cast members whose name (or first/last name) actually appears in the subtitles."""
    words = set(re.findall(r"[a-z][a-z'\-]+", text.lower()))
    out, seen = [], set()
    for c in cast:
        if c["gender"] == "unknown" or c["name"] in seen:
            continue
        ks = _keys(c["name"])
        if any(k in words or (" " in k and k in text.lower()) for k in ks):
            seen.add(c["name"])
            out.append(c)
    return out[:limit]


def correct_scenes(scenes: list[dict], cast: list[dict]) -> int:
    """Fix participant genders in a scene map from the cast list; returns changes."""
    idx = gender_index(cast)
    changed = 0
    for sc in scenes:
        for p in sc.get("participants", []):
            if not isinstance(p, dict) or p.get("name") in (None, "", "unnamed"):
                continue
            g = lookup(idx, p["name"])
            if g and p.get("gender") != g:
                p["gender"] = g
                changed += 1
    return changed
