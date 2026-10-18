# -*- coding: utf-8 -*-
"""v1.6.0 features: crop removal, thumbnail sorting, Ctrl+Z manual undo,
and the unified eraser (manual + auto censor, dilate-safe)."""
import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.export import compose_from_state
from app.core.image_ops import compose_image
from app.core.selection import MaskSelection, RectSelection


def _noise_img(size=(96, 96), seed=3):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(size[1], size[0], 3),
                                        dtype=np.uint8))


# ------------------------------------------------------------------ crop removal

def test_crop_tool_removed(qapp):
    from app.ui.editor_canvas import EditorCanvas
    from app.ui.tool_panel import ToolPanel
    from conftest import AutoStub
    from types import SimpleNamespace

    assert not hasattr(EditorCanvas, "_apply_crop")
    fake = AutoStub(
        config=default_config(), sticker_library={},
        canvas=AutoStub(current_strength=7, current_effect="blur",
                        auto_mask=None,
                        state_changed=SimpleNamespace(emit=lambda *a, **k: None)),
    )
    panel = ToolPanel(fake)
    assert "crop" not in panel.tool_buttons
    assert "eraser" in panel.tool_buttons


# ------------------------------------------------------------------ sorting

def test_sort_paths_orders_and_keeps_selection(qapp, tmp_path):
    from pathlib import Path

    from app.core.numbering import natural_key
    from app.ui.thumbnail_list import ThumbnailList

    tl = ThumbnailList()
    paths = []
    for name in ("img_10.png", "img_2.png", "img_1.png"):
        p = tmp_path / name
        Image.new("RGB", (16, 16), (9, 9, 9)).save(p)
        paths.append(str(p))
    tl.add_paths(paths)
    tl.sort_paths(lambda p: natural_key(Path(p).name))
    assert [Path(p).name for p in tl.item_paths()] == [
        "img_1.png", "img_2.png", "img_10.png"
    ]
    # descending
    tl.sort_paths(lambda p: natural_key(Path(p).name), reverse=True)
    assert Path(tl.item_paths()[0]).name == "img_10.png"
    # selection preserved across reorder and follows its item
    tl.setCurrentItem(tl._item_for_path(str(tmp_path / "img_10.png")))
    tl.sort_paths(lambda p: natural_key(Path(p).name))
    assert tl.current_path() == str(tmp_path / "img_10.png")
    assert tl.item_paths()[-1] == str(tmp_path / "img_10.png")  # sorted last


# ------------------------------------------------------------------ Ctrl+Z manual undo

def _canvas_with_manual_and_auto(qapp):
    from app.ui.editor_canvas import EditorCanvas

    canvas = EditorCanvas()
    canvas.set_image(np.full((80, 96, 3), 120, dtype=np.uint8))
    canvas.auto_mask = np.zeros((80, 96), dtype=bool)
    canvas.auto_mask[20:60, 20:60] = True
    canvas.selections = [RectSelection(5, 5, 30, 30, effect="blur", strength=6)]
    canvas._rebuild()
    return canvas


def test_ctrl_z_undoes_manual_but_not_auto(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    canvas = _canvas_with_manual_and_auto(qapp)
    # snapshot the state BEFORE the manual edit (real press-time snapshot)
    canvas.selections = []
    canvas._push_manual_undo()
    canvas.selections = [RectSelection(5, 5, 30, 30, effect="blur", strength=6)]
    assert canvas.selections and canvas.auto_mask.any()

    QTest.keyClick(canvas, Qt.Key_Z, Qt.ControlModifier)
    assert not canvas.selections
    assert canvas.auto_mask.any()  # auto censor untouched


# ------------------------------------------------------------------ eraser compose layer

def test_erase_subtracts_after_dilation_compose():
    img = _noise_img((96, 96), seed=7)
    # auto mask covers [40:72] with dilate=10 -> effective [30:82]; erase the
    # center band [50:62] which is well inside the dilated area
    mask = np.zeros((96, 96), dtype=bool)
    mask[40:72, 40:72] = True
    erase = np.zeros((96, 96), dtype=bool)
    erase[50:62, 50:62] = True

    auto = dict(auto_effect="blur", auto_strength=10, auto_mask_dilate=10)
    out = compose_image(img, [], [MaskSelection(mask)], [], {},
                        auto_params=auto, erase_mask=erase)
    arr, base = np.array(out), np.array(img)
    # erased center is fully restored (no dilation leak)
    assert np.array_equal(arr[52:60, 52:60], base[52:60, 52:60])
    # regions outside the erased band (inside dilated area) stay censored
    assert not np.array_equal(arr[36:44, 36:44], base[36:44, 36:44])
    assert not np.array_equal(arr[70:78, 70:78], base[70:78, 70:78])


def test_erase_also_targets_manual_selections():
    img = _noise_img((64, 64), seed=11)
    sel = RectSelection(10, 10, 40, 40, effect="blur", strength=8)
    erase = np.zeros((64, 64), dtype=bool)
    erase[20:24, 20:24] = True
    out = compose_image(img, [sel], [], [], {}, params={}, erase_mask=erase)
    arr, base = np.array(out), np.array(img)
    # erased spot of the manual selection is restored
    assert np.array_equal(arr[20:24, 20:24], base[20:24, 20:24])
    # the rest of the selection is still blurred
    assert not np.array_equal(arr[12:18, 12:18], base[12:18, 12:18])


def test_eraser_canvas_stroke_and_manual_undo(qapp):
    from app.ui.editor_canvas import EditorCanvas

    canvas = EditorCanvas()
    canvas.set_image(np.array(_noise_img((80, 96), seed=5)))
    h, w = canvas.image_np.shape[:2]
    canvas.auto_mask = np.zeros((h, w), dtype=bool)
    canvas.auto_mask[20:60, 20:60] = True
    canvas._rebuild()

    canvas.set_tool("eraser")
    canvas._push_undo()
    canvas._push_manual_undo()
    canvas._apply_erase_points([(40, 40)])
    assert canvas.erase_mask is not None and canvas.erase_mask.any()
    before = canvas.erase_mask.copy()

    canvas.undo_manual()
    assert canvas.erase_mask is not None and not canvas.erase_mask.any()

    canvas._apply_erase_points([(40, 40)])
    assert np.array_equal(canvas.erase_mask, before)


def test_export_writes_erase_mask_through_state(tmp_path):
    from app.core.export import export_batch

    img = _noise_img((64, 64), seed=13)
    p = tmp_path / "src.png"
    img.save(p)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    erase = np.zeros((64, 64), dtype=bool)
    erase[28:36, 28:36] = True
    states = {str(p): {
        "pil": img, "selections": [], "stickers": [],
        "auto_mask": mask, "erase_mask": erase,
        "orig_size": (64, 64),
    }}
    cfg = {
        "out_dir": str(tmp_path / "out"), "name_mode": "preserve",
        "prefix": "OUT", "start": 1, "digits": 3, "fmt": "PNG",
        "quality": 100, "scale_mode": "none", "percent": 100,
        "width": 1000, "height": 1000, "strip_metadata": False,
    }
    auto_s = {
        "conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
        "auto_strength": 10, "auto_mask_dilate": 8, "input_size": 1024,
    }
    result = export_batch([str(p)], states, default_config(), auto_s, cfg, {})
    assert result["processed"] == 1
    with Image.open(tmp_path / "out" / "src.png") as out:
        arr = np.array(out.convert("RGB"))
    base = np.array(img)
    assert np.array_equal(arr[30:34, 30:34], base[30:34, 30:34])  # erased
    assert not np.array_equal(arr[20:26, 20:26], base[20:26, 20:26])  # censored