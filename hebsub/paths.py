"""Where hebsub keeps its data: work dirs, series memory, caches, models.

HEBSUB_HOME overrides everything. Otherwise ~/hebsub when it exists (the original install),
else the platform's user data folder:
  Windows  %APPDATA%\\hebsub
  macOS    ~/Library/Application Support/hebsub
  Linux    $XDG_DATA_HOME/hebsub or ~/.local/share/hebsub
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _default() -> Path:
    legacy = Path.home() / "hebsub"
    if legacy.is_dir():
        return legacy
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "hebsub"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "hebsub"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "hebsub"


HOME = Path(os.environ["HEBSUB_HOME"]).expanduser() if os.environ.get("HEBSUB_HOME") else _default()


def pid_alive(pid: int) -> bool:
    """Is this process still running? os.kill(pid, 0) would TERMINATE it on Windows."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


SECRETS = HOME / "secrets.json"


def secret(name: str) -> str | None:
    """An API key: the environment variable first, then secrets.json in the data folder
    (written by `hebsub setup`, readable only by the user on macOS/Linux)."""
    val = os.environ.get(name)
    if val:
        return val.strip().strip('"').strip("'") or None
    try:
        import json
        return (json.loads(SECRETS.read_text(encoding="utf-8")).get(name) or "").strip() or None
    except (OSError, ValueError):
        return None


def save_secret(name: str, value: str) -> None:
    import json
    data = {}
    try:
        data = json.loads(SECRETS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    data[name] = value
    SECRETS.parent.mkdir(parents=True, exist_ok=True)
    SECRETS.write_text(json.dumps(data, indent=1), encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(SECRETS, 0o600)
