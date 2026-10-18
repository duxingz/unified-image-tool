# -*- coding: utf-8 -*-
"""v1.5.3 features: app icon, pointer-drag panning, delete ordering,
export mask dilation consistency."""
import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.export import export_batch
from app.core.selection import MaskSelection, RectSelection, Sticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ icon

def test_icon_assets_exist():
    png = PROJECT_ROOT / "app" / "assets" / "icon.png"
    ico = PROJECT_ROOT / "app" / "assets" / "icon.ico"
    assert png.exists() and ico.exists()
    with Image.open(ico) as im:
        # a Windows icon set contains the 256px entry
        assert any(s[0] >= 128 for s in im.info.get("sizes", []))


# ------------------------------------------------------------------ pointer drag

def test_pointer_left_drag_pans_canvas(qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from app.ui.editor_canvas import EditorCanvas

    canvas = EditorCanvas()
    canvas.resize(640, 480)
    canvas.show()
    rng = np.random.default_rng(1)
    canvas.set_image(rng.integers(0, 255, size=(400, 600, 3), dtype=np.uint8))
    canvas.set_tool("pointer")
    canvas.scale(2.0, 2.0)  # zoomed in -> scrollbars appear

    hbar = canvas.horizontalScrollBar()
    before = hbar.value()
    start = QPoint(400, 240)
    QTest.mousePress(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(canvas.viewport(), QPoint(250, 160))
    QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(250, 160))
    assert hbar.value() > before  # dragged left -> scrolled right
    canvas.close()


# ------------------------------------------------------------------ delete ordering

def _canvas_with_all_items(qapp):
    from app.ui.editor_canvas import EditorCanvas

    canvas = EditorCanvas()
    canvas.set_image(np.zeros((120, 160, 3), dtype=np.uint8))
    canvas.selections = [RectSelection(10, 10, 40, 40, effect="blur", strength=5)]
    canvas.stickers = [Sticker("s1", 80, 60)]
    canvas.auto_mask = np.zeros((120, 160), dtype=bool)
    canvas.auto_mask[50:70, 50:70] = True
    return canvas


def test_delete_selected_order_is_selection_sticker_automasks(qapp):
    canvas = _canvas_with_all_items(qapp)
    assert canvas.selections and canvas.stickers and canvas.auto_mask.any()

    canvas.delete_selected()  # 1: manual selection
    assert not canvas.selections
    assert canvas.stickers and canvas.auto_mask.any()

    canvas.delete_selected()  # 2: sticker
    assert not canvas.stickers
    assert canvas.auto_mask.any()

    canvas.delete_selected()  # 3: auto mask
    assert not canvas.auto_mask.any()

    canvas.undo()  # undo restores the auto mask
    assert canvas.auto_mask.any()


# ------------------------------------------------------------------ export dilation consistency

def _auto(effect="blur", strength=8, dilate=0):
    return {
        "conf": 0.3, "iou": 0.7, "classes": [],
        "auto_effect": effect, "auto_strength": strength,
        "auto_mask_dilate": dilate, "input_size": 1024,
    }


def test_export_batch_applies_mask_dilate(tmp_path):
    grad = np.tile(np.linspace(0, 200, 64, dtype=np.uint8), (64, 1))
    pil = Image.fromarray(np.stack([grad, grad // 2, grad // 4], axis=-1))
    p = tmp_path / "src.png"
    pil.save(p)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    state = {
        "pil": pil, "rgb": np.array(pil), "selections": [], "stickers": [],
        "auto_mask": mask,
    }
    states = {str(p): state}
    cfg_base = {
        "out_dir": "", "name_mode": "preserve", "prefix": "OUT",
        "start": 1, "digits": 3, "fmt": "PNG", "quality": 100,
        "scale_mode": "none", "percent": 100, "width": 1000, "height": 1000,
        "strip_metadata": False,
    }
    for tag, dilate in (("d0", 0), ("d10", 10)):
        out = tmp_path / tag
        cfg = dict(cfg_base, out_dir=str(out))
        export_batch([str(p)], states, default_config(), _auto(dilate=dilate), cfg, {})
    arr0 = np.array(Image.open(tmp_path / "d0" / "src.png"))
    arr10 = np.array(Image.open(tmp_path / "d10" / "src.png"))
    assert not np.array_equal(arr0, arr10)
    # dilate=10 enlarges the censored region beyond the raw mask corner
    probe = (slice(21, 25), slice(21, 25))  # just outside the raw mask
    assert np.array_equal(arr0[probe], np.array(pil)[probe])
    assert not np.array_equal(arr10[probe], np.array(pil)[probe])


# ------------------------------------------------------------------ settings dialog

def test_settings_dialog_uses_spinbox_for_strength(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")

    from PySide6.QtWidgets import QSpinBox

    from app.ui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(default_config())
    assert isinstance(dlg.auto_strength, QSpinBox)
    assert dlg.auto_strength.value() == 15