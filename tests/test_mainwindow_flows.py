# -*- coding: utf-8 -*-
"""MainWindow flow tests (offscreen): batch censor regression, stats bar,
image removal and theme switching."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.selection import RectSelection


@pytest.fixture()
def no_dialogs(monkeypatch):
    """No-op modal dialogs so flows do not block offscreen."""
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Yes))


class StubDetector:
    """Deterministic detector: every image gets the same square mask."""
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[10:34, 10:34] = True
        return [SimpleNamespace(mask=mask, bbox=(10, 10, 34, 34),
                                class_id=3, class_name="vagina",
                                confidence=0.9)]


def make_images(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (160, 120), (i * 60, 100, 150)).save(p)
        paths.append(str(p))
    return paths


def pump_until(app, worker, predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while (worker is None or worker.isRunning()) or not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            pytest.fail("timed out waiting for batch flow")
        time.sleep(0.01)


def test_batch_censor_first_image_keeps_mask(qapp, tmp_path, no_dialogs):
    """Regression: after batch censor the first (current) image must keep the
    worker's mask. It used to be overwritten by the stale canvas state when
    the done-handler re-loaded the current image."""
    from app.ui.main_window import MainWindow

    paths = make_images(tmp_path, 3)
    w = MainWindow(StubDetector(), default_config())
    w._add_paths(paths)  # current = paths[0]; canvas auto_mask is empty
    w.batch_auto_censor()
    pump_until(qapp, w.auto_worker,
               lambda: w.auto_worker is not None
               and not w.auto_worker.isRunning()
               and w.canvas.auto_mask is not None
               and w.canvas.auto_mask.any())

    state = w.image_states[paths[0]]
    assert state["auto_mask"] is not None and state["auto_mask"].any()
    assert w.canvas.auto_mask is not None and w.canvas.auto_mask.any()
    assert "已打码" in w.thumbnail_list.item(0).toolTip()
    assert "已打码 3" in w.thumb_stats_label.text()

    # other images keep their masks too
    for p in paths:
        assert w.image_states[p]["auto_mask"].any()


def test_thumb_stats_counts(qapp, tmp_path, no_dialogs):
    from app.ui.main_window import MainWindow

    paths = make_images(tmp_path, 3)
    w = MainWindow(StubDetector(), default_config())
    w._add_paths(paths)
    assert w.thumb_stats_label.text() == "共 3 张 ｜ 已打码 0 ｜ 未打码 3"

    w.canvas.selections = [RectSelection(5, 5, 20, 20, effect="blur", strength=5)]
    w.canvas.state_changed.emit()
    assert "共 3 张 ｜ 已打码 1" in w.thumb_stats_label.text()
    assert "未打码 2" in w.thumb_stats_label.text()


def test_remove_images_updates_list_and_current(qapp, tmp_path, no_dialogs):
    from app.ui.main_window import MainWindow

    paths = make_images(tmp_path, 3)
    w = MainWindow(StubDetector(), default_config())
    w._add_paths(paths)
    assert w.thumbnail_list.count() == 3

    w._remove_selected_images([paths[0]])  # removing the current image
    assert w.thumbnail_list.count() == 2
    assert paths[0] not in w.image_states
    assert w.current_path == paths[1]  # jumps to the first remaining row
    assert w.canvas.image_np is not None
    assert "共 2 张" in w.thumb_stats_label.text()

    w._remove_selected_images(paths[1:])  # remove the rest
    assert w.thumbnail_list.count() == 0
    assert w.current_path is None
    assert w.canvas.image_np is None
    assert "共 0 张" in w.thumb_stats_label.text()


def test_theme_switching_applies_qss(qapp):
    from app.ui import theme

    apply = theme.apply_theme
    apply(qapp, "xiaodunight")
    assert qapp.styleSheet() == theme.XIAODUNIGHT_QSS
    apply(qapp, "xiaodu")
    assert qapp.styleSheet() == theme.XIAODU_QSS


def test_settings_dialog_has_theme_combo(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")

    from app.ui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(default_config())
    assert dlg.theme_combo.currentData() == "xiaodu"
    dlg.theme_combo.setCurrentIndex(1)
    assert dlg.theme_combo.currentData() == "xiaodunight"