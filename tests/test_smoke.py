"""Smoke tests for ollama-classifier-gui.

These guard against import/build-time regressions (e.g. Flet API drift, broken
Tabs/TabBarView invariants) without launching a GUI, so they are safe to run in
headless CI.
"""

import flet as ft
import polars as pl

from ollama_classifier_gui import main as main_mod
from ollama_classifier_gui.utils import (
    BACKEND_TYPES,
    DEFAULT_ENDPOINTS,
    app_repository_url,
    app_version,
    check_system_dependencies,
    load_config,
    save_config,
)


def test_entry_point_is_callable():
    assert callable(main_mod.main)
    assert callable(main_mod._flet_main)


def test_backend_constants():
    assert "ollama" in BACKEND_TYPES
    assert "vllm" in BACKEND_TYPES
    assert "sglang" in BACKEND_TYPES
    assert "llamacpp" in BACKEND_TYPES
    for b in BACKEND_TYPES:
        assert b in DEFAULT_ENDPOINTS


def test_config_roundtrip(tmp_path, monkeypatch):
    import ollama_classifier_gui.utils as utils

    monkeypatch.setattr(utils, "config_path", lambda: tmp_path / "config.json")
    cfg = load_config()
    assert cfg["backend_type"] == "ollama"
    assert cfg["max_calls"] == "1"  # default max_calls in config
    cfg["model"] = "test-model"
    save_config(cfg)
    assert (tmp_path / "config.json").is_file()
    reloaded = load_config()
    assert reloaded["model"] == "test-model"
    assert reloaded["max_calls"] == "1"  # persisted


def test_excel_export_works(tmp_path):
    """Regression guard: polars write_excel must not take ``engine`` and needs xlsxwriter."""
    df = pl.DataFrame(
        {"text": ["a", "b"], "prediction": ["pos", "neg"], "confidence": [0.9, 0.4]}
    )
    out = tmp_path / "results.xlsx"
    df.write_excel(str(out))
    assert out.is_file() and out.stat().st_size > 0


def test_tabs_tabbar_tabbarview_invariants():
    """Regression guard for the Schema Tabs structure that previously failed to render.

    ft.Tabs.length must match both ft.TabBar.tabs and ft.TabBarView.controls; the
    split structure (TabBar + TabBarView) is mandatory in Flet 0.84 because
    ft.Tab has no ``content`` parameter.
    """
    tabbar = ft.TabBar(
        tabs=[ft.Tab(label="Manual Labels"), ft.Tab(label="Load from File")]
    )
    view = ft.TabBarView(
        controls=[ft.Column([ft.Text("a")]), ft.Column([ft.Text("b")])]
    )
    tabs = ft.Tabs(length=2, selected_index=0, content=ft.Column([tabbar, view]))
    assert tabs.length == len(tabbar.tabs) == len(view.controls) == 2


def test_check_system_dependencies_returns_list():
    result = check_system_dependencies()
    assert isinstance(result, list)


def test_load_excel_with_sheets_reads_first_sheet(tmp_path):
    """Shared Excel loader used by both data and schema handlers."""
    df = pl.DataFrame({"label": ["pos", "neg"], "desc": ["good", "bad"]})
    path = tmp_path / "labels.xlsx"
    df.write_excel(str(path))
    sheet_names, loaded = main_mod.OllamaClassifierApp._load_excel_with_sheets(
        str(path)
    )
    assert sheet_names == ["Sheet1"]
    assert loaded.columns == ["label", "desc"]


def test_app_version_is_nonempty():
    v = app_version()
    assert isinstance(v, str)
    assert v.strip() != ""
    assert all(c.isdigit() or c == "." for c in v)


def test_app_repository_url_is_https():
    url = app_repository_url()
    assert url.startswith("https://")
    assert "github.com" in url
