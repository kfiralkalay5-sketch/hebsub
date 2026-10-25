"""Configuration loading for hebsub."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from .paths import HOME

DEFAULT_CONFIG_PATHS = [
    Path(os.environ.get("HEBSUB_CONFIG", "")) if os.environ.get("HEBSUB_CONFIG") else None,
    HOME / "config.json",
    Path(__file__).resolve().parent.parent / "config.json",
]


class Config:
    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._d = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        candidates = [Path(path)] if path else [p for p in DEFAULT_CONFIG_PATHS if p]
        for c in candidates:
            if c.is_file():
                with c.open(encoding="utf-8") as fh:
                    return cls(json.load(fh), c)
        packaged = Path(__file__).resolve().parent / "default_config.json"
        if not path and packaged.is_file():  # a fresh install works out of the box
            with packaged.open(encoding="utf-8") as fh:
                return cls(json.load(fh), packaged)
        raise FileNotFoundError(
            "No config.json found. Looked in: "
            + ", ".join(str(c) for c in candidates)
        )

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def secret(self, dotted_env_key: str) -> str | None:
        """Read a secret from the environment variable named in the config."""
        name = self.get(dotted_env_key)
        if not name:
            return None
        return os.environ.get(name) or None

    # -- convenience paths -------------------------------------------------
    @property
    def skill_dir(self) -> Path:
        # the bundled CueIvrit scripts (MIT) when no skill_dir is configured
        return Path(self.get("skill_dir") or Path(__file__).resolve().parent / "vendor" / "cueivrit").expanduser()

    @property
    def work_root(self) -> Path:
        p = Path(self.get("work_root", str(HOME / "work"))).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_file(self) -> Path:
        p = Path(self.get("db_file", str(HOME / "hebsub.db"))).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_file(self) -> Path:
        p = Path(self.get("log_file", str(HOME / "hebsub.log"))).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def map_path(self, remote: str) -> str:
        """Translate a path as Sonarr/Radarr sees it into a local path."""
        for pair in self.get("path_map", []) or []:
            if len(pair) == 2 and remote.startswith(pair[0]):
                return pair[1] + remote[len(pair[0]):]
        return remote
