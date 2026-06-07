"""Tiny persisted settings for the GUI (e.g. last-browsed folders).

A single JSON object is read/written at a per-user path:

- ``STEMS_GUI_CONFIG`` (env) if set - points directly at the file (used by tests);
- else ``%APPDATA%/stems/gui.json`` on Windows;
- else ``$XDG_CONFIG_HOME/stems/gui.json`` or ``~/.config/stems/gui.json``.

Both helpers are best-effort: any error is swallowed and treated as "no
settings", because a config hiccup must never stop the window from opening.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def config_path() -> Path:
    override = os.environ.get("STEMS_GUI_CONFIG")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "stems" / "gui.json"


def load_settings() -> dict:
    """Return the saved settings dict, or ``{}`` if absent/unreadable."""
    try:
        data = json.loads(config_path().read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    """Write ``data`` as JSON, creating parent dirs. Never raises."""
    try:
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), "utf-8")
    except Exception:
        pass
