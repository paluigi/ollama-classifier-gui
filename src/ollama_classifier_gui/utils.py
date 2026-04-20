"""Utility helpers for the Ollama Classifier GUI.

- `check_system_dependencies()` → returns a list of missing Linux requirements.
- `show_missing_dependencies(page, missing)` → displays an alert dialog.
- `config_path()` → OS‑appropriate config directory + file.
- `load_config()` / `save_config(data)` → JSON read/write.
"""

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# System‑dependency detection (Linux only)
# ---------------------------------------------------------------------------

def _is_linux() -> bool:
    return platform.system() == "Linux"


def _linux_missing_requirements() -> List[str]:
    """Return a list of missing system binaries/libraries on Linux.

    Required for this app:
    * zenity – FilePicker GUI
    * libsecret‑1‑0 – SecureStorage backend
    * One keyring service: gnome-keyring, kwalletmanager, or secret-service
    """
    missing: List[str] = []
    # zenity
    if shutil.which("zenity") is None:
        missing.append("zenity (install with: sudo apt install zenity)")
    # libsecret development/runtime library (runtime required)
    # we just check the shared object existence
    libsecret = shutil.which("libsecret-1.so") or shutil.which("libsecret-1.so.0")
    if libsecret is None:
        missing.append("libsecret-1-0 (install with: sudo apt install libsecret-1-0)")
    # keyring service – at least one must be present
    keyring_candidates = [
        "gnome-keyring-daemon",
        "kwalletmanager5",
        "secret-service",
    ]
    if not any(shutil.which(k) for k in keyring_candidates):
        missing.append(
            "A keyring service (gnome-keyring, kwalletmanager, or secret-service)"
        )
    return missing


def check_system_dependencies() -> List[str]:
    """Public wrapper – returns [] on non‑Linux or when everything is present."""
    if not _is_linux():
        return []
    return _linux_missing_requirements()

# ---------------------------------------------------------------------------
# Config file handling (JSON) – stored in OS‑specific user config folder
# ---------------------------------------------------------------------------

def _default_config() -> Dict[str, str]:
    return {
        "host": "http://localhost:11434",
        "model": "llama3.2",
        "theme": "system",  # "light", "dark", or "system"
    }


def config_path() -> Path:
    """Return the full path to the JSON config file.

    Linux:   $XDG_CONFIG_HOME/ollama-classifier-gui/config.json (fallback `~/.config`)
    macOS:   ~/Library/Application Support/ollama-classifier-gui/config.json
    Windows: %APPDATA%\ollama-classifier-gui\config.json
    """
    system = platform.system()
    if system == "Linux":
        base = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    elif system == "Darwin":  # macOS
        base = os.path.expanduser("~/Library/Application Support")
    elif system == "Windows":
        base = os.getenv("APPDATA", "")
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    cfg_dir = Path(base) / "ollama-classifier-gui"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.json"


def load_config() -> Dict[str, str]:
    cfg_file = config_path()
    if not cfg_file.is_file():
        # create a fresh default file
        save_config(_default_config())
        return _default_config()
    try:
        with cfg_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # ensure required keys exist
        for k, v in _default_config().items():
            data.setdefault(k, v)
        return data
    except Exception:
        # corrupted file – reset to defaults
        save_config(_default_config())
        return _default_config()


def save_config(data: Dict[str, str]) -> None:
    cfg_file = config_path()
    with cfg_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Helper to show missing‑dependency alert (to be called from Flet UI)
# ---------------------------------------------------------------------------

async def show_missing_dependencies(page, missing: List[str]):
    """Display a modal dialog listing missing system requirements."""
    import flet as ft

    if not missing:
        return
    msg = "The following required system components are missing:\n\n" + "\n".join(missing)
    dlg = ft.AlertDialog(
        title=ft.Text("Missing System Dependencies"),
        content=ft.Text(msg, selectable=True),
        actions=[
            ft.TextButton("OK", on_click=lambda e: page.close(dlg)),
        ],
    )
    page.overlay.append(dlg)
    await page.update_async()
    await dlg.wait_dismiss()

# End of utils.py
