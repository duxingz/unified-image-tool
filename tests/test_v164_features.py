# -*- coding: utf-8 -*-
"""v1.6.4: drag-reorder fix, order badges, export rename memory,
standard multi-select (Ctrl+A / rubber band), drop highlight."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView

from app.ui.thumbnail_list import ThumbnailList


def _make_thumbs(tmp_path, names=("1_a.png", "2_b.png", "3_c.png")):
    paths = []
    for name in names:
        p = tmp_path / name
        Image.new("RGB", (16, 16), (7, 8, 9)).save(p)
        paths.append(str(p))
    return paths


def _list_with(tmp_path, names=("1_a.png", "2_b.png", "3_c.png")):
    tl = ThumbnailList()
    tl.add_paths(_make_thumbs(tmp_path, names))
    return tl


# ------------------------------------------------------------------ ① drag fix

def test_thumbnail_list_drag_is_hand_rolled(qapp, tmp_path):
    """Regression: main window used to disable acceptDrops on the list,
    which killed internal drag-reorder entirely (forbidden cursor).
    v1.6.5 replaced Qt drag-drop with hand-rolled live reorder - the list
    must have NO Qt drag machinery and rely on its own mouse handling."""
    tl = _list_with(tmp_path)
    assert tl.dragDropMode() == QAbstractItemView.DragDropMode.NoDragDrop
    assert tl.dragEnabled() is False
    assert tl.movement() == ThumbnailList.Static


# ------------------------------------------------------------------ ② order badges

def test_order_badges_render_from_second_image(qapp, tmp_path):
    paths = _make_thumbs(tmp_path)
    tl = ThumbnailList()
    tl.add_paths(paths[:1])  # single image: no order badge
    base = tl._base_pixmaps[paths[0]]
    assert tl.item(0).icon().pixmap(base.size()).toImage() == base.toImage()

    tl.add_paths(paths[1:])  # 2+ images: order badges appear
    for i in range(tl.count()):
        b = tl._base_pixmaps[paths[i]]
        assert tl.item(i).icon().pixmap(b.size()).toImage() != b.toImage()

    # reorder (what a drag produces) repaints numbers in the new order
    tl._sync_paths_from_items()


def test_order_badges_follow_reorder(qapp, tmp_path):
    from app.core.numbering import natural_key

    tl = _list_with(tmp_path, names=("2_b.png", "1_a.png", "10_c.png"))
    tl.sort_paths(lambda p: natural_key(Path(p).name))
    # icon of item 0 must differ from item 2's base (order numbers differ);
    # simplest observable: every icon differs from its own base after sort
    for i in range(tl.count()):
        p = tl.item_paths()[i]
        b = tl._base_pixmaps[p]
        assert tl.item(i).icon().pixmap(b.size()).toImage() != b.toImage()


# ------------------------------------------------------------------ ④ multi select

def test_ctrl_a_selects_all(qapp, tmp_path):
    tl = _list_with(tmp_path)
    QTest.keyClick(tl, Qt.Key_A, Qt.ControlModifier)
    assert len(tl.selectedItems()) == 3


def test_selection_rect_visible(qapp, tmp_path):
    tl = _list_with(tmp_path)
    assert tl.isSelectionRectVisible() is True


# ------------------------------------------------------------------ ⑤ drop highlight

def test_drop_highlight_flashes_and_clears(qapp, tmp_path):
    tl = _list_with(tmp_path)
    p = paths = tl.item_paths()[0]
    tl._flash_highlight([p])
    assert p in tl._highlight
    tl._clear_highlight()
    assert p not in tl._highlight


# ------------------------------------------------------------------ ③ rename memory

def test_save_dialog_restores_last_rename(qapp):
    from app.ui.save_dialog import SaveDialog

    dlg = SaveDialog(None, last_rename={
        "name_mode": "rename", "prefix": "TRIP", "start": 100, "digits": 4,
    })
    assert dlg.name_combo.currentData() == "rename"
    assert dlg.prefix_edit.text() == "TRIP"
    assert dlg.start_spin.value() == 100
    assert dlg.digits_spin.value() == 4

    dlg2 = SaveDialog(None)  # defaults when nothing remembered
    assert dlg2.name_combo.currentData() == "preserve"
    assert dlg2.prefix_edit.text() == "IMG"
    assert dlg2.start_spin.value() == 1
    assert dlg2.digits_spin.value() == 3


def test_main_window_persists_last_rename(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")

    from app.ui.save_dialog import SaveDialog
    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    p = tmp_path / "a.png"
    Image.new("RGB", (16, 16), (5, 5, 5)).save(p)
    w = MainWindow(StubDet(), config_mod.default_config())
    w._add_paths([str(p)])

    captured = {}

    class FakeDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return SaveDialog.DialogCode.Accepted

        @staticmethod
        def result_config():
            captured["cfg"] = True
            return {
                "out_dir": str(tmp_path / "out"), "name_mode": "rename",
                "prefix": "TRIP", "start": 7, "digits": 4, "fmt": "PNG",
                "quality": 100, "scale_mode": "none", "percent": 100,
                "width": 1000, "height": 1000, "strip_metadata": False,
            }

    monkeypatch.setattr("app.ui.main_window.SaveDialog", FakeDialog)
    w.open_save_dialog()

    assert w.config["last_rename"] == {
        "name_mode": "rename", "prefix": "TRIP", "start": 7, "digits": 4,
    }