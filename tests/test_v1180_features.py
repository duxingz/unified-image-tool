# -*- coding: utf-8 -*-
"""v1.18.0 — 小独爱心/小独夜 skins in, persona out (user decision)."""
import pytest

from app.core.config import default_config
from app.ui import theme


def test_xiaodu_skins_are_registered_and_distinct():
    # "light"/"dark"/"persona" are ALIASES now (v1.19.0), so they resolve to
    # one of the two skins by design -- only the real skins are distinct
    assert theme.theme_qss("xiaodu") != theme.theme_qss("xiaodunight")
    for name in ("xiaodu", "xiaodunight", "light", "dark", "persona"):
        assert len(theme.theme_qss(name)) > 3000


def test_xiaodu_palette_keys_match_between_skins():
    """The two vendored palettes must carry the exact same roles —
    self-painting widgets read current_palette() and index them, and the
    shared vendored QSS template formats every key."""
    assert set(theme._xh.LIGHT.keys()) == set(theme._xh.DARK.keys())
    assert "canvas" in theme._xh.LIGHT and "accent" in theme._xh.LIGHT


def test_removed_skin_names_resolve_through_aliases():
    """v1.19.0 removed the original light/dark skins (persona left in
    v1.18.0): all three legacy names must resolve through the aliases."""
    assert not hasattr(theme, "PERSONA")
    assert not hasattr(theme, "LIGHT") and not hasattr(theme, "DARK")
    assert not hasattr(theme, "_QSS_TEMPLATE")
    assert "persona" not in theme._THEMES
    assert theme.theme_qss("light") == theme.XIAODU_QSS
    assert theme.theme_qss("persona") == theme.XIAODU_QSS
    assert theme.theme_qss("dark") == theme.XIAODUNIGHT_QSS


def test_persona_theme_survives_settings_dialog_replaced_by_xiaodu(qapp, tmp_path,
                                                                   monkeypatch):
    from app.ui.settings_dialog import SettingsDialog
    cfg = default_config()
    cfg["theme"] = "xiaodu"
    dlg = SettingsDialog(cfg, None)
    assert dlg.theme_combo.currentData() == "xiaodu"
    cfg["theme"] = "xiaodunight"
    dlg2 = SettingsDialog(cfg, None)
    assert dlg2.theme_combo.currentData() == "xiaodunight"
    # a legacy persona config lands on the first entry (小独爱心), not an error
    cfg["theme"] = "persona"
    dlg3 = SettingsDialog(cfg, None)
    assert dlg3.theme_combo.currentData() == "xiaodu"


def test_legacy_config_names_apply(qapp):
    """Old configs (light/dark/persona) keep working through the aliases."""
    theme.apply_theme(qapp, "persona")
    assert theme.current_theme() == "xiaodu"
    theme.apply_theme(qapp, "dark")
    assert theme.current_theme() == "xiaodunight"
    theme.apply_theme(qapp, "light")
    assert theme.current_theme() == "xiaodu"
    assert theme.current_palette()["accent"] == "#2478A8"
