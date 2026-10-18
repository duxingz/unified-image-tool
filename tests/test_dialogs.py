# -*- coding: utf-8 -*-
"""Offscreen UI tests for dialogs (no display needed via QT_QPA_PLATFORM,
set in conftest before PySide6 is imported)."""
from conftest import AutoStub

from app.core import config as config_mod
from app.core.config import default_config
from app.core.selection import AUTO_EFFECT_CHOICES


def test_save_dialog_result_config(qapp, tmp_path):
    from app.ui.save_dialog import SaveDialog

    dlg = SaveDialog(None, default_out=str(tmp_path), default_fmt="JPG",
                     default_strip_meta="strip")
    cfg = dlg.result_config()
    required = {"out_dir", "name_mode", "prefix", "start", "digits", "fmt",
                "quality", "scale_mode", "percent", "width", "height",
                "strip_metadata"}
    assert required <= set(cfg)
    assert cfg["out_dir"] == str(tmp_path)
    assert cfg["fmt"] == "JPG"
    assert cfg["strip_metadata"] is True


def test_settings_dialog_auto_effect_uses_english_ids(qapp, tmp_path, monkeypatch):
    # never touch the real %APPDATA% config
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")

    from app.ui.settings_dialog import SettingsDialog

    cfg = default_config()
    assert cfg["auto_effect"] == "blur"
    dlg = SettingsDialog(cfg)
    ids = [dlg.auto_effect_combo.itemData(i)
           for i in range(dlg.auto_effect_combo.count())]
    assert ids == [eff for eff, _ in AUTO_EFFECT_CHOICES]

    # a legacy config storing the Chinese label still restores correctly
    legacy = dict(cfg)
    legacy["auto_effect"] = "常规模糊"
    dlg2 = SettingsDialog(legacy)
    assert dlg2.auto_effect_combo.currentData() == "blur"


def test_tool_panel_auto_settings_returns_english_effect(qapp):
    from app.ui.tool_panel import ToolPanel

    fake_main = AutoStub(
        config=default_config(),
        sticker_library={},
        canvas=AutoStub(current_strength=5, current_effect="blur"),
    )
    panel = ToolPanel(fake_main)
    settings = panel.auto_settings()
    assert settings["auto_effect"] == "blur"
    assert settings["auto_effect"] in ("blur", "black_lines", "white_mist",
                                       "light", "pixel")