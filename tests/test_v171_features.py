# -*- coding: utf-8 -*-
"""v1.7.1: full export-dialog memory, window geometry, double-click fit,
auto-open export folder, _dirty initialization."""
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from app.core.config import default_config


# ------------------------------------------------------------------ A: export memory

def test_save_dialog_restores_last_export(qapp):
    from app.ui.save_dialog import SaveDialog

    dlg = SaveDialog(None, default_out=r"C:\nowhere",
                     default_fmt="PNG", default_strip_meta="preserve",
                     last_export={
                         "out_dir": r"D:\last\out", "fmt": "JPG",
                         "quality": 85, "scale_mode": "percent",
                         "percent": 50, "width": 800, "height": 600,
                         "strip_metadata": True,
                     })
    assert dlg.out_edit.text() == r"D:\last\out"
    assert dlg.fmt_combo.currentText() == "JPG"
    assert dlg.quality_spin.value() == 85
    assert dlg.scale_combo.currentData() == "percent"
    assert dlg.percent_spin.value() == 50
    assert dlg.strip_meta_check.isChecked() is True


def test_save_dialog_without_memory_uses_defaults(qapp):
    from app.ui.save_dialog import SaveDialog

    dlg = SaveDialog(None, default_out=r"C:\src", default_fmt="PNG",
                     default_strip_meta="preserve")
    assert dlg.out_edit.text() == r"C:\src"
    assert dlg.quality_spin.value() == 100
    assert dlg.scale_combo.currentData() == "none"
    assert dlg.strip_meta_check.isChecked() is False


def test_main_window_persists_full_last_export(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod
    from PySide6.QtWidgets import QMessageBox

    for n in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, n,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "cfg" / "config.json")

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    out = tmp_path / "out"
    out.mkdir()
    p = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (4, 4, 4)).save(p)

    w = MainWindow(StubDet(), config_mod.default_config())
    w._add_paths([str(p)])

    fake_cfg = {
        "out_dir": str(out), "name_mode": "preserve", "prefix": "IMG",
        "start": 1, "digits": 3, "fmt": "JPG", "quality": 90,
        "scale_mode": "percent", "percent": 75, "width": 800, "height": 600,
        "strip_metadata": True,
    }

    class FakeDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return 1  # Accepted

        def result_config(self):
            return dict(fake_cfg)

    monkeypatch.setattr("app.ui.main_window.SaveDialog", FakeDialog)
    # don't actually run the export worker: config write-back is the target,
    # and a dangling worker thread would emit done() during a LATER test's
    # event handling and pop a modal dialog that hangs the whole run
    monkeypatch.setattr("app.ui.main_window.BatchExportWorker.start",
                        lambda self: None)
    w.open_save_dialog()

    le = w.config["last_export"]
    assert le["out_dir"] == str(out)
    assert le["fmt"] == "JPG"
    assert le["quality"] == 90
    assert le["scale_mode"] == "percent" and le["percent"] == 75
    assert le["strip_metadata"] is True
    # rename memory untouched shape
    assert set(w.config["last_rename"]) == {"name_mode", "prefix", "start", "digits"}


# ------------------------------------------------------------------ B: window geometry

def test_window_geometry_saved_and_restored(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "cfg" / "config.json")
    for n in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, n,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), config_mod.default_config())
    w.resize(1100, 700)
    w.show()
    QTest.qWait(50)
    w.close()
    assert config_mod.default_config()["window_geometry"] != w.config.get(
        "window_geometry")
    assert w.config.get("window_geometry")

    w2 = MainWindow(StubDet(), w.config)
    w2.show()
    QTest.qWait(50)
    # offscreen platform may clamp to the virtual screen; assert it restored
    # to ~the saved size rather than the 1360x840 default
    assert abs(w2.width() - 1100) <= 250
    assert abs(w2.height() - 700) <= 250
    w2.close()


# ------------------------------------------------------------------ C: double-click fit

def test_double_click_fit_toggle(qapp, tmp_path):
    from PySide6.QtCore import QPoint

    from app.ui.editor_canvas import EditorCanvas

    c = EditorCanvas()
    c.resize(400, 300)
    c.show()
    # small image: plenty of real blank viewport around it at any zoom
    c.set_image(np.random.default_rng(2).integers(
        0, 255, size=(80, 100, 3), dtype=np.uint8))
    blank = QPoint(380, 280)  # viewport corner: real blank at any zoom

    c.double_click_fit = True
    c._apply_zoom(1 / 1.15)  # zoom out a step
    assert c.transform().m11() < c._fit_scale
    QTest.mouseDClick(c.viewport(), Qt.LeftButton, Qt.NoModifier, blank)
    assert abs(c.transform().m11() - c._fit_scale) < 1e-6  # back to fit

    # toggle off: double click leaves the zoom alone
    c._apply_zoom(1 / 1.15)
    zoom_before = c.transform().m11()
    c.double_click_fit = False
    QTest.mouseDClick(c.viewport(), Qt.LeftButton, Qt.NoModifier, blank)
    assert abs(c.transform().m11() - zoom_before) < 1e-6


# ------------------------------------------------------------------ D: auto open

def test_auto_open_export_folder(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from app.ui.main_window import MainWindow

    opened = []
    monkeypatch.setattr(
        "app.ui.main_window.QDesktopServices.openUrl",
        staticmethod(lambda url: opened.append(url.toString())),
    )
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    from PySide6.QtWidgets import QDialog
    monkeypatch.setattr(QDialog, "exec", lambda self: 1)  # result dialog

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    cfg = default_config()
    cfg["auto_open_export_folder"] = True
    w = MainWindow(StubDet(), cfg)
    w._last_export_dir = str(tmp_path)

    w._on_batch_done({"total": 2, "processed": 2, "skipped": 0, "failed": 0,
                      "details": []})
    assert opened and opened[0].startswith("file://")

    # toggle off -> no auto open
    opened.clear()
    cfg["auto_open_export_folder"] = False
    w._on_batch_done({"total": 2, "processed": 2, "skipped": 0, "failed": 0,
                      "details": []})
    assert opened == []


# ------------------------------------------------------------------ E: _dirty init

def test_dirty_initialized_no_close_crash(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from app.ui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    for n in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, n, staticmethod(lambda *a, **k: None))

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    assert w._dirty_paths == set()
    assert w.close() is True  # no AttributeError on the very first close