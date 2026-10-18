# -*- coding: utf-8 -*-
"""v1.6.5: hand-rolled live drag reorder (items get pushed aside in real
time, selection survives, no forbidden cursor) + 5-step left panel snap."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from app.ui.thumbnail_list import ThumbnailList


def _list_with(tmp_path, names=("1_a.png", "2_b.png", "3_c.png", "4_d.png")):
    tl = ThumbnailList()
    tl.resize(400, 600)
    tl.show()
    for name in names:
        p = tmp_path / name
        Image.new("RGB", (16, 16), (7, 8, 9)).save(p)
        tl.add_paths([str(p)])
    return tl


def _item_center(tl, row) -> QPoint:
    r = tl.visualItemRect(tl.item(row))
    return r.center()


def _drag_item_to(tl, from_row, to_item_row, right_half=False):
    """Simulate a hand drag of item[from_row] onto item[to_item_row]."""
    start = _item_center(tl, from_row)
    QTest.mousePress(tl.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    # cross the 4px threshold, then land on the target's center/half
    target_rect = tl.visualItemRect(tl.item(to_item_row))
    cx = target_rect.center().x() + (target_rect.width() // 4 if right_half else -target_rect.width() // 4)
    cy = target_rect.center().y()
    for f in (0.3, 0.6, 1.0):
        QTest.mouseMove(tl.viewport(), QPoint(
            int(start.x() + (cx - start.x()) * f),
            int(start.y() + (cy - start.y()) * f),
        ))
    QTest.mouseRelease(tl.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(cx, cy))


def test_live_reorder_pushes_items_aside(qapp, tmp_path):
    tl = _list_with(tmp_path)
    order0 = [Path(p).name for p in tl.item_paths()]
    assert order0 == ["1_a.png", "2_b.png", "3_c.png", "4_d.png"]

    # drag item 0 onto item 2's left half -> item 0 lands at index 2
    _drag_item_to(tl, 0, 2, right_half=False)
    names = [Path(p).name for p in tl.item_paths()]
    assert names == ["2_b.png", "1_a.png", "3_c.png", "4_d.png"], names
    # the dragged item stays selected and paths stay in sync
    assert tl.current_path() == str(tmp_path / "1_a.png")
    assert tl._paths == tl.item_paths()
    # a highlight flash is armed for the dragged item
    assert str(tmp_path / "1_a.png") in tl._highlight


def test_live_reorder_right_half_inserts_after(qapp, tmp_path):
    tl = _list_with(tmp_path)
    # drag item 3 onto item 0's RIGHT half -> lands at index 1
    _drag_item_to(tl, 3, 0, right_half=True)
    names = [Path(p).name for p in tl.item_paths()]
    assert names == ["1_a.png", "4_d.png", "2_b.png", "3_c.png"], names


def test_drag_over_itself_keeps_order(qapp, tmp_path):
    tl = _list_with(tmp_path)
    _drag_item_to(tl, 1, 1, right_half=False)
    names = [Path(p).name for p in tl.item_paths()]
    assert names == ["1_a.png", "2_b.png", "3_c.png", "4_d.png"]
    assert tl.current_path() == str(tmp_path / "2_b.png")


def test_modifier_click_does_not_reorder(qapp, tmp_path):
    tl = _list_with(tmp_path)
    # Ctrl+press+move must NOT start a reorder (multi-select semantics)
    start = _item_center(tl, 0)
    QTest.mousePress(tl.viewport(), Qt.LeftButton, Qt.ControlModifier, start)
    QTest.mouseMove(tl.viewport(), QPoint(start.x() - 60, start.y() - 40))
    QTest.mouseMove(tl.viewport(), _item_center(tl, 2))
    QTest.mouseRelease(tl.viewport(), Qt.LeftButton, Qt.ControlModifier,
                       _item_center(tl, 2))
    names = [Path(p).name for p in tl.item_paths()]
    assert names == ["1_a.png", "2_b.png", "3_c.png", "4_d.png"]


# ------------------------------------------------------------------ ② five-step snap

def test_left_panel_snaps_to_five_columns(qapp, tmp_path):
    from PySide6.QtWidgets import QMessageBox
    for n in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.Yes))

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config := __import__(
        "app.core.config", fromlist=["default_config"]).default_config())
    w.resize(1360, 840)
    w.show()  # splitter layout only engages once shown
    grid = w.thumbnail_list.gridSize().width()
    overhead = 16

    def snap(pos):
        w._snap_left_panel_width(pos, 0)
        return w.splitter.sizes()[0]

    for col in (1, 2, 3, 4, 5):
        near = col * grid + overhead - 10  # within tolerance
        got = snap(near)
        assert got == col * grid + overhead, f"col {col}: {got}"

    # far outside any standard width is clamped to 1 or 5 columns
    got = snap(3 * grid + overhead + 200)
    assert got in (3 * grid + overhead, 4 * grid + overhead, 5 * grid + overhead)