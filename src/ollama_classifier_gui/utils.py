"""Utility helpers for the Ollama Classifier GUI.

- `check_system_dependencies()` → returns a list of missing Linux requirements.
- `show_missing_dependencies(page, missing)` → displays an alert dialog.
- `config_path()` → OS‑appropriate config directory + file.
- `load_config()` / `save_config(data)` → JSON read/write.
- `app_version()` / `app_repository_url()` → package metadata (from pyproject.toml).
"""

import json
import os
import platform
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND_TYPES = ["ollama", "vllm", "sglang", "llamacpp"]

DEFAULT_ENDPOINTS = {
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    "llamacpp": "http://localhost:8080/v1",
}

# ---------------------------------------------------------------------------
# System‑dependency detection (Linux only)
# ---------------------------------------------------------------------------


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _linux_missing_requirements() -> list[str]:
    """Return a list of missing system binaries/libraries on Linux.

    Required for this app:
    * zenity – FilePicker GUI
    * libsecret‑1‑0 – SecureStorage backend
    * One keyring service: gnome-keyring, kwalletmanager, or secret-service
    """
    missing: list[str] = []
    if shutil.which("zenity") is None:
        missing.append("zenity (install with: sudo apt install zenity)")
    libsecret = shutil.which("libsecret-1.so") or shutil.which("libsecret-1.so.0")
    if libsecret is None:
        import ctypes.util

        if not ctypes.util.find_library("secret-1"):
            missing.append(
                "libsecret-1-0 (install with: sudo apt install libsecret-1-0)"
            )
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


def check_system_dependencies() -> list[str]:
    """Public wrapper – returns [] on non‑Linux or when everything is present."""
    if not _is_linux():
        return []
    return _linux_missing_requirements()


# ---------------------------------------------------------------------------
# Config file handling (JSON) – stored in OS‑specific user config folder
# ---------------------------------------------------------------------------


def _default_config() -> dict[str, str]:
    return {
        "backend_type": "ollama",
        "endpoint": "http://localhost:11434",
        "model": "llama3.2",
        "theme": "system",
        "batch_size": "1",
        "max_calls": "1",
        "output_format": "top_label",
    }


def config_path() -> Path:
    """Return the full path to the JSON config file.

    Linux:   $XDG_CONFIG_HOME/ollama-classifier-gui/config.json (fallback `~/.config`)
    macOS:   ~/Library/Application Support/ollama-classifier-gui/config.json
    Windows: %APPDATA%\\ollama-classifier-gui\\config.json
    """
    system = platform.system()
    if system == "Linux":
        base = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    elif system == "Darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif system == "Windows":
        base = os.getenv("APPDATA", "")
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    cfg_dir = Path(base) / "ollama-classifier-gui"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.json"


def load_config() -> dict[str, str]:
    cfg_file = config_path()
    if not cfg_file.is_file():
        save_config(_default_config())
        return _default_config()
    try:
        with cfg_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in _default_config().items():
            data.setdefault(k, v)
        return data
    except Exception:
        save_config(_default_config())
        return _default_config()


def save_config(data: dict[str, str]) -> None:
    cfg_file = config_path()
    with cfg_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Package metadata (read from pyproject.toml via importlib.metadata)
# ---------------------------------------------------------------------------

PACKAGE_NAME = "ollama-classifier-gui"


def _project_urls() -> dict[str, str]:
    """Return a mapping of project-URL label → URL from installed metadata."""
    try:
        from importlib.metadata import metadata

        entries = metadata(PACKAGE_NAME).get_all("Project-URL") or []
    except Exception:
        return {}
    urls: dict[str, str] = {}
    for entry in entries:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    return urls


def app_version() -> str:
    """Return the application version (from package metadata)."""
    try:
        from importlib.metadata import version

        return version(PACKAGE_NAME)
    except Exception:
        return "0.0.0"


def app_repository_url() -> str:
    """Return the GitHub repository URL (Repository, then Homepage fallback)."""
    urls = _project_urls()
    return urls.get("repository") or urls.get("homepage") or ""


# ---------------------------------------------------------------------------
# Helper to show missing‑dependency alert (to be called from Flet UI)
# ---------------------------------------------------------------------------


async def show_missing_dependencies(page, missing: list[str]):
    """Display a modal dialog listing missing system requirements."""
    import flet as ft

    if not missing:
        return
    msg = "The following required system components are missing:\n\n" + "\n".join(
        missing
    )
    dlg = ft.AlertDialog(
        title=ft.Text("Missing System Dependencies"),
        content=ft.Text(msg, selectable=True),
        actions=[
            ft.TextButton("OK", on_click=lambda e: page.pop_dialog()),
        ],
    )
    page.show_dialog(dlg)
