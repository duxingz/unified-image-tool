# -*- coding: utf-8 -*-
"""v1.6.1: drag 1:1 panning, zoom clamp + horizontal wheel, zoom indicator,
view padding (export-safe), time sort, drag-reorder sync, Windows-like sort."""
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.numbering import natural_key
from app.ui.thumbnail_list import ThumbnailList
from app.ui.editor_canvas import EditorCanvas


def _canvas(qapp, size=(400, 600)):
    c = EditorCanvas()
    c.resize(640, 480)
    c.show()
    c.set_image(np.random.default_rng(1).integers(
        0, 255, size=(size[1], size[0], 3), dtype=np.uint8))
    return c


def _zoom_ratio(c):
    return c.transform().m11() / (c._fit_scale or 1.0)


# ------------------------------------------------------------------ ① drag panning

def test_panning_is_1_to_1_with_zoom(qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    c = _canvas(qapp)  # 400x600 portrait: vertical overflow at 1.5x
    c.scale(1.5, 1.5)  # zoomed in -> scrollbars active
    vbar = c.verticalScrollBar()
    before = vbar.value()
    # drag exactly 100 viewport pixels UP (vertical scrollbar value rises)
    start = QPoint(320, 400)
    end = QPoint(320, 300)
    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(c.viewport(), end)
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, end)
    moved = vbar.value() - before
    assert abs(moved - 100) <= 3, f"expected ~100px pan, got {moved}"


# ------------------------------------------------------------------ ②③ zoom clamp

def test_zing_zoom_clamped_5_to_1600_percent(qapp):
    c = _canvas(qapp)
    fit = c._fit_scale
    for _ in range(50):
        c._apply_zoom(1.15)
    assert _zoom_ratio(c) <= 16.0 + 1e-6
    for _ in range(100):
        c._apply_zoom(1 / 1.15)
    assert _zoom_ratio(c) >= 0.05 - 1e-6  # floor is 5% (user-specified)


def test_zoom_emits_ratio_and_indicator(qapp):
    c = _canvas(qapp)
    seen = []
    c.zoomChanged.connect(lambda r: seen.append(round(r, 3)))
    c._apply_zoom(1.15)
    assert seen and seen[-1] > 1.0
    c._rebuild(fit=True)  # reload/fit resets to 100%
    assert seen[-1] == 1.0


# ------------------------------------------------------------------ ⑤ view padding

def test_padding_only_when_zoomed_in(qapp):
    c = _canvas(qapp)
    h, w = c.image_np.shape[:2]
    rect = c._scene.sceneRect()
    assert rect.top() == 0 and rect.left() == 0
    assert rect.width() == w and rect.height() == h  # fitted: exact content

    c._apply_zoom(2.0)
    assert c._padded is True
    padded = c._scene.sceneRect()
    assert padded.left() < 0 and padded.top() < 0
    assert padded.width() > w and padded.height() > h

    # zoom back below fit -> content rect restored exactly
    for _ in range(6):
        c._apply_zoom(1 / 1.15)
    assert c._padded is False
    back = c._scene.sceneRect()
    assert back.left() == 0 and back.width() == w


def test_padding_never_leaks_into_preview(qapp):
    """Zoom + pan into the padded area: composed preview stays the image size."""
    c = _canvas(qapp)
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    c._apply_zoom(2.0)
    # pan far into the padded region
    for _ in range(5):
        QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(320, 240))
        QTest.mouseMove(c.viewport(), QPoint(50, 50))
        QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(50, 50))
    preview = c._make_preview_image()
    h, w = c.image_np.shape[:2]
    assert preview.shape[:2] == (h, w)  # never any padding/borders


# ------------------------------------------------------------------ ⑦ windows-like sort

def test_natural_key_matches_windows_examples():
    names = ["037.png", "037-1.png", "037-2.png", "037-12.png", "38.png"]
    assert sorted(names, key=natural_key) == [
        "037.png", "037-1.png", "037-2.png", "037-12.png", "38.png"
    ]
    assert sorted(["img_10.png", "img_2.png", "img_1.png"], key=natural_key) == [
        "img_1.png", "img_2.png", "img_10.png"
    ]


# ------------------------------------------------------------------ ⑥ time sort + drag sync

def test_sort_by_mtime(qapp, tmp_path):
    paths = []
    for i, name in enumerate(("old.png", "mid.png", "new.png")):
        p = tmp_path / name
        Image.new("RGB", (8, 8), (i, i, i)).save(p)
        mtime = 1700000000 + i * 3600
        os.utime(p, (mtime, mtime))
        paths.append(str(p))
    tl = ThumbnailList()
    tl.add_paths(paths)
    tl.sort_paths(lambda p: Path(p).stat().st_mtime)
    assert [Path(p).name for p in tl.item_paths()] == ["old.png", "mid.png", "new.png"]
    tl.sort_paths(lambda p: Path(p).stat().st_mtime, reverse=True)
    assert Path(tl.item_paths()[0]).name == "new.png"


def test_sync_paths_after_item_reorder(qapp, tmp_path):
    paths = []
    for name in ("a.png", "b.png", "c.png"):
        p = tmp_path / name
        Image.new("RGB", (8, 8), (3, 3, 3)).save(p)
        paths.append(str(p))
    tl = ThumbnailList()
    tl.add_paths(paths)
    # what InternalMove leaves behind: items in a new order
    items = [tl.item(2), tl.item(0), tl.item(1)]
    for _ in range(tl.count()):
        tl.takeItem(0)
    for it in items:
        tl.addItem(it)
    tl._sync_paths_from_items()
    assert [Path(p).name for p in tl.item_paths()] == ["c.png", "a.png", "b.png"]