# -*- coding: utf-8 -*-
"""v1.14.5 — settings that were silently not remembered, plus the new defaults.

Reported: "自动打码的强度我改过之后没有记忆效果". Root cause: the 自动打码 tab's
widgets were only ever written by the Settings dialog. Changing them in the main
panel was read back on the next start (_apply_auto_defaults) but never SAVED,
because closeEvent only collected the manual edit defaults. Two bugs:

1. seven auto-tab settings were not persisted at all (strength/effect/dilate/
   confidence/IoU/regions/detect size);
2. the region checkboxes were not even RESTORED, so a saved selection was
   ignored on every start.

Also: the manual and auto strength defaults move 10 -> 15.
"""
import json

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from app.core import config as config_mod
from app.core.config import UI_VERSION, default_config, load_config


class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, *a, **k):
        return []


OPEN = []


@pytest.fixture(autouse=True)
def _close_windows():
    yield
    while OPEN:
        w = OPEN.pop()
        try:
            w.close()
        except Exception:
            pass


@pytest.fixture()
def no_dialogs(monkeypatch):
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes),
                            raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self: 1, raising=False)


def _win(qapp, tmp_path, monkeypatch, config=None):
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(StubDetector(), config or default_config())
    OPEN.append(w)
    return w


def _saved() -> dict:
    return json.loads(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))


# =========================================== the reported bug: persistence

def test_auto_tab_changes_survive_a_restart(qapp, tmp_path, monkeypatch,
                                           no_dialogs):
    """Change every auto-tab control, close, and read the file back."""
    w = _win(qapp, tmp_path, monkeypatch)
    tp = w.tool_panel
    tp.auto_strength_spin.setValue(37)
    tp.mask_dilate_spin.setValue(4)
    tp.conf_spin.setValue(0.55)
    tp.iou_spin.setValue(0.42)
    idx = tp.auto_effect_combo.findData("pixel")
    tp.auto_effect_combo.setCurrentIndex(idx)
    for key, cb in tp.class_checks.items():
        cb.setChecked(key == "anus")
    idx = tp.input_size_combo.findData(640)
    if idx >= 0:
        tp.input_size_combo.setCurrentIndex(idx)

    w.close()
    saved = _saved()
    assert saved["auto_strength"] == 37, "自动强度还是没有记忆"
    assert saved["auto_mask_dilate"] == 4
    assert abs(saved["conf_threshold"] - 0.55) < 1e-6
    assert abs(saved["iou_threshold"] - 0.42) < 1e-6
    assert saved["auto_effect"] == "pixel"
    assert saved["selected_regions"] == ["anus"]
    if idx >= 0:
        assert saved["detect_size"] == 640


def test_auto_tab_values_come_back_in_the_panel(qapp, tmp_path, monkeypatch,
                                               no_dialogs):
    """A fresh window must show what was saved (this also covers the checkbox
    restore, which was missing entirely)."""
    cfg = default_config()
    cfg.update({"auto_strength": 66, "auto_mask_dilate": 7, "conf_threshold": 0.61,
                "iou_threshold": 0.33, "auto_effect": "white_mist",
                "selected_regions": ["anus"], "detect_size": 640})
    w = _win(qapp, tmp_path, monkeypatch, cfg)
    tp = w.tool_panel
    assert tp.auto_strength_spin.value() == 66
    assert tp.mask_dilate_spin.value() == 7
    assert abs(tp.conf_spin.value() - 0.61) < 1e-6
    assert abs(tp.iou_spin.value() - 0.33) < 1e-6
    assert tp.auto_effect_combo.currentData() == "white_mist"
    assert [k for k, cb in tp.class_checks.items() if cb.isChecked()] == ["anus"]
    assert tp.input_size_combo.currentData() == 640
    # the canvas must censor with the restored values
    assert w.canvas.auto_params["auto_strength"] == 66
    assert w.canvas.auto_params["auto_effect"] == "white_mist"


def test_manual_defaults_are_persisted_too(qapp, tmp_path, monkeypatch,
                                          no_dialogs):
    """The manual side already worked -- keep it working."""
    w = _win(qapp, tmp_path, monkeypatch)
    w.canvas.set_brush_size(44)
    w.canvas.current_strength = 23
    w.canvas.current_effect = "pixel"
    w.close()
    ed = _saved()["edit_defaults"]
    assert ed["brush_size"] == 44
    assert ed["strength"] == 23
    assert ed["effect"] == "pixel"


# ================================================== the new default: 15

def test_strength_defaults_are_15():
    cfg = default_config()
    assert cfg["auto_strength"] == 15
    assert cfg["edit_defaults"]["strength"] == 15


def test_canvas_and_panel_default_to_15(qapp, tmp_path, monkeypatch, no_dialogs):
    from app.ui.editor_canvas import EditorCanvas
    cv = EditorCanvas()
    assert cv.current_strength == 15
    assert cv.auto_params["auto_strength"] == 15

    cfg = default_config()
    del cfg["auto_strength"]                      # simulate a config without it
    del cfg["edit_defaults"]
    w = _win(qapp, tmp_path, monkeypatch, cfg)
    assert w.tool_panel.auto_strength_spin.value() == 15
    assert w.tool_panel.strength_spin.value() == 15


def test_migration_bumps_only_an_untouched_10(tmp_path, monkeypatch):
    """A stored 10 was the old built-in default ("never touched") -> 15.
    Any other number is a deliberate choice and must survive."""
    for stored, want in ((10, 15), (42, 42), (7, 7)):
        data = default_config()
        data["auto_strength"] = stored
        data["edit_defaults"]["strength"] = stored
        data["ui_version"] = "1.7.2"              # the previous schema
        config_mod.CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False),
                                          encoding="utf-8")
        loaded = load_config()
        assert loaded["auto_strength"] == want, f"stored {stored}"
        assert loaded["edit_defaults"]["strength"] == want, f"stored {stored}"
        assert loaded["ui_version"] == UI_VERSION
        # the migration is persisted, so it can never run twice
        assert json.loads(config_mod.CONFIG_PATH.read_text(
            encoding="utf-8"))["ui_version"] == UI_VERSION


def test_migration_runs_once_and_leaves_new_configs_alone(tmp_path, monkeypatch):
    data = default_config()
    data["auto_strength"] = 10
    data["ui_version"] = UI_VERSION               # already current
    config_mod.CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False),
                                      encoding="utf-8")
    loaded = load_config()
    assert loaded["auto_strength"] == 10, "an already-migrated config was touched"


def test_legacy_config_without_marker_gets_15(tmp_path, monkeypatch):
    data = default_config()
    data.pop("ui_version")
    data["auto_strength"] = 3
    config_mod.CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False),
                                      encoding="utf-8")
    loaded = load_config()
    assert loaded["auto_strength"] == 15
    assert loaded["edit_defaults"]["strength"] == 15
