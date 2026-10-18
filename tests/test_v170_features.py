# -*- coding: utf-8 -*-
"""v1.7.0: in-app rename, group drag reorder, autoscroll, 9-step width,
clear-list entries."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from app.core.config import default_config
from app.ui.thumbnail_list import ThumbnailList


def _build_list(tmp_path, names=("1_a.png", "2_b.png", "3_c.png", "4_d.png")):
    tl = ThumbnailList()
    tl.resize(340, 600)
    tl.show()
    for name in names:
        p = tmp_path / name
        Image.new("RGB", (48, 48), (11, 22, 33)).save(p)
        tl.add_paths([str(p)])
    return tl


# ------------------------------------------------------------------ ① rename

def test_inline_rename_stores_display_name(qapp, tmp_path):
    tl = _build_list(tmp_path)
    item = tl.item(0)
    path = tl.item_paths()[0]

    tl._on_item_changed(item)  # text still the disk name -> clears
    assert tl.display_name(path) == "1_a.png"

    tl._suspend_item_changed = False
    item.setText("我的图A")  # what the inline editor commits
    tl._on_item_changed(item)
    assert tl._display_names[path] == "我的图A"
    assert tl.display_name(path) == "我的图A"
    assert item.text() == "我的图A"

    # empty text restores the disk name
    item.setText("")
    tl._on_item_changed(item)
    assert path not in tl._display_names
    assert tl.display_name(path) == "1_a.png"


def test_batch_rename_by_list_order(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    paths = []
    for name in ("x1.png", "x2.png", "x3.png"):
        p = tmp_path / name
        Image.new("RGB", (16, 16), (3, 3, 3)).save(p)
        paths.append(str(p))
    w._add_paths(paths)

    w._apply_batch_rename(paths, "TRIP", 7, 4, clear=False)
    assert w.thumbnail_list.display_name(paths[0]) == "TRIP0007"
    assert w.thumbnail_list.display_name(paths[2]) == "TRIP0009"
    assert w.config["display_names"][paths[1]] == "TRIP0008"

    # clear restores disk-name captions
    w._apply_batch_rename(paths, "X", 1, 3, clear=True)
    assert w.thumbnail_list.display_name(paths[0]) == "x1.png"
    assert w.config["display_names"] == {}


def test_name_sort_prefers_display_names(qapp, tmp_path):
    tl = _build_list(tmp_path, names=("a.png", "b.png", "c.png"))
    tl.set_display_name(str(tmp_path / "a.png"), "zzz")
    tl.sort_paths(lambda p: natural_key_wrap(tl.display_name(p)))
    names = [Path(p).name for p in tl.item_paths()]
    assert names == ["b.png", "c.png", "a.png"]  # display "zzz" sorts last


def natural_key_wrap(text):
    from app.core.numbering import natural_key
    return natural_key(text)


# ------------------------------------------------------------------ ② group drag

def test_group_drag_moves_whole_selection(qapp, tmp_path):
    tl = _build_list(tmp_path)
    # select items 0 and 1 (1_a, 2_b)
    for i in (0, 1):
        tl.item(i).setSelected(True)
    tl.setCurrentItem(tl.item(1))

    start = tl.visualItemRect(tl.item(1)).center()
    target = tl.visualItemRect(tl.item(3))
    end = QPoint(target.left() + target.width() // 4, target.center().y())
    QTest.mousePress(tl.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    for f in (0.3, 0.6, 1.0):
        QTest.mouseMove(tl.viewport(), QPoint(
            int(start.x() + (end.x() - start.x()) * f),
            int(start.y() + (end.y() - start.y()) * f)))
    QTest.mouseRelease(tl.viewport(), Qt.LeftButton, Qt.NoModifier, end)

    names = [Path(p).name for p in tl.item_paths()]
    assert names == ["3_c.png", "1_a.png", "2_b.png", "4_d.png"], names
    # the whole group stays selected and contiguous
    selected = {Path(p).name for p in tl.selected_paths()}
    assert selected == {"1_a.png", "2_b.png"}
    assert tl.current_path() == str(tmp_path / "2_b.png")


# ------------------------------------------------------------------ ③ autoscroll

def test_autoscroll_engages_at_viewport_edges(qapp, tmp_path):
    tl = _build_list(tmp_path)
    # enough items to overflow the viewport
    for i in range(12):
        p = tmp_path / f"pad_{i:02d}.png"
        Image.new("RGB", (48, 48), (50, 50, 50)).save(p)
        tl.add_paths([str(p)])
    assert tl._autoscroll_timer.isActive() is False

    QTest.mousePress(tl.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(100, 300))
    tl._start_reorder()
    vbar = tl.verticalScrollBar()
    vbar.setValue(vbar.maximum())  # park at the bottom first
    value = vbar.value()

    pos = QPoint(100, 5)  # near the top edge
    tl._update_reorder(pos)
    assert tl._autoscroll_dir == -1
    assert tl._autoscroll_timer.isActive() is True
    for _ in range(5):
        tl._autoscroll_tick()
    assert vbar.value() < value  # scrolled up

    pos = QPoint(100, tl.viewport().height() // 2)  # away from edges
    tl._update_reorder(pos)
    assert tl._autoscroll_dir == 0
    QTest.mouseRelease(tl.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    assert tl._autoscroll_timer.isActive() is False


# ------------------------------------------------------------------ ④ nine-step width

def test_left_panel_nine_step_switching(qapp, tmp_path):
    from PySide6.QtWidgets import QMessageBox
    for n in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.Yes))

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    w.resize(2400, 900)  # wide enough for the 8-column standard (976px)
    w.show()
    grid = w.thumbnail_list.gridSize().width()
    overhead = 16
    widths = [0] + [n * grid + overhead for n in range(1, 9)]

    def snap(pos):
        w._snap_left_panel_width(pos, 0)
        return w.splitter.sizes()[0]

    # crossing midpoints steps directly to the standard width
    for col in range(0, 9):
        got = snap(widths[col] + 4)  # just past a standard, before midpoint
        assert got == widths[col], f"col {col}: got {got}"
    # crossing the midpoint to the next standard jumps there
    mid = (widths[2] + widths[3]) / 2 + 2
    assert snap(mid) == widths[3]
    # col 0 collapses the whole panel
    assert snap(4) == 0
    # and dragging back right restores 1 column
    assert snap(widths[1] + 2) == widths[1]


# ------------------------------------------------------------------ ⑤ clear list

def test_clear_all_empties_everything(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "cfg" / "config.json")

    w = MainWindow(StubDet(), default_config())
    paths = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        Image.new("RGB", (16, 16), (2, 2, 2)).save(p)
        paths.append(str(p))
    w._add_paths(paths)
    w.thumbnail_list.set_display_name(paths[0], "展示名")
    w._save_display_names()
    assert w.thumbnail_list.count() == 2

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    for n in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, n, staticmethod(lambda *a, **k: None))
    w.clear_all_images()

    assert w.thumbnail_list.count() == 0
    assert w.image_states == {}
    assert w.current_path is None
    assert w.canvas.image_np is None
    assert w.config["display_names"] == {}
    assert "共 0 张" in w.thumb_stats_label.text()

    # closing after clearing must NOT warn about unsaved edits (a stale
    # dirty path whose content is gone is filtered by _unsaved_paths)
    w._dirty_paths = {"C:\\ghost\\never_loaded.png"}
    calls = {"n": 0}
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: calls.update(n=calls["n"] + 1)
                                     or QMessageBox.Yes))
    assert w.close() is True
    assert calls["n"] == 0