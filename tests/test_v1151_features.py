# -*- coding: utf-8 -*-
"""v1.15.1 — mini/main detection-input parity, eraser-over-sticker polish,
copy-on-write undo snapshots."""
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPointF

from app.ui.editor_canvas import EditorCanvas
from tests.test_v1150_features import _drive, texture

W, H = 400, 300


def _black_canvas(qapp):
    cv = EditorCanvas()
    bg = Image.new("RGB", (W, H), (0, 0, 0))
    cv.image_np = np.array(bg)
    cv.original_pil = bg
    cv.sticker_library = {}
    cv._rebuild()
    return cv


# ================================== 1: mini detection input == main window

def test_mini_detects_at_edit_resolution_and_composes_full(qapp, tmp_path):
    """Large photo: the DETECTOR must see the same 2048-capped edit image the
    main window feeds it, while the OUTPUT stays at the original resolution
    (MaskSelection.to_mask upscales the mask, same as the export path)."""
    from app.ui.mini_window import MiniWindow

    src = tmp_path / "big.png"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 256, (2600, 1400, 3), dtype=np.uint8),
                    "RGB").save(src)

    class ShapeDet:
        def __init__(self):
            self.seen = None

        def detect(self, bgr, **kw):
            self.seen = bgr.shape
            m = np.zeros(bgr.shape[:2], dtype=bool)
            m[100:400, 100:600] = True
            return [type("D", (), {"mask": m})()]

    det = ShapeDet()
    mini = MiniWindow(det, {"selected_regions": ["vagina"],
                            "auto_effect": "pixel", "auto_strength": 10,
                            "auto_mask_dilate": 0})
    out, n = mini._compose_censored(str(src))
    assert n == 1
    h, w = det.seen[:2]
    # load_resized caps the long edge at 2048: 2600 -> 2048, 1400 -> 1102
    assert (h, w) == (2048, 1102), f"detector saw {det.seen[:2]}"
    assert out.size == (1400, 2600), "compose must keep original resolution"
    # the censor actually landed, upscaled into the full-res output
    orig = np.array(Image.open(src).convert("RGB")).astype(int)
    diff = np.abs(np.array(out.convert("RGB")).astype(int) - orig).max(axis=2)
    assert diff[150:450, 150:700].max() > 40
    assert diff[:80, :80].max() == 0


# ============================== 2: eraser drag keeps stickers intact

def test_eraser_drag_keeps_stickers_intact(qapp):
    """Erasing straight through a sticker used to punch holes into it for
    the whole drag (sticker popped back at release); the live frame must
    keep the sticker over the erased area, matching the release rebuild."""
    cv = _black_canvas(qapp)
    cv.sticker_library = {"dot": Image.new("RGBA", (40, 40), (255, 0, 0, 255))}
    cv.add_sticker("dot", x=200, y=150, width=80, height=80)
    assert not cv.stickers[0].watermark
    cv.set_tool("brush")
    cv.set_brush_size(30)
    cv.current_effect = "pixel"
    cv.current_strength = 12
    _drive(cv, [("press", (60, 150)), ("move", (150, 150)),
                ("release", (150, 150))])
    cv.set_tool("eraser")
    _drive(cv, [("press", (180, 150)), ("move", (220, 150))])
    live = cv._live_img
    assert (live[150, 200][0] > 200 and live[150, 200][1] < 80), \
        "sticker hole during eraser drag"
    _drive(cv, [("release", (240, 150))])
    final = cv._composed_base
    assert final[150, 200][0] > 200 and final[150, 200][1] < 80


def test_eraser_over_sticker_matches_release_pixels(qapp):
    """Full-frame check: the live frame drawn during the drag must be
    pixel-identical to the release rebuild inside the erased stroke."""
    cv = _black_canvas(qapp)
    cv.sticker_library = {"dot": Image.new("RGBA", (40, 40), (255, 0, 0, 255))}
    cv.add_sticker("dot", x=200, y=150, width=80, height=80)
    # some censor content first: an empty canvas makes the eraser press a
    # deliberate no-op (_has_censor_content guard)
    cv.set_tool("brush")
    cv.set_brush_size(30)
    cv.current_effect = "pixel"
    cv.current_strength = 12
    _drive(cv, [("press", (60, 250)), ("move", (150, 250)),
                ("release", (150, 250))])
    cv.set_tool("eraser")
    _drive(cv, [("press", (180, 150)), ("move", (220, 150))])
    live = cv._live_img.copy()
    _drive(cv, [("release", (240, 150))])
    final = cv._composed_base
    stroke = np.zeros((H, W), dtype=bool)
    stroke[135:166, 165:236] = True   # erase stroke footprint (brush 30)
    diff = np.abs(live.astype(int) - final.astype(int)).max(axis=2)
    assert diff[stroke].max() == 0, "live erase frame != release rebuild"


# ============================ 3: copy-on-write undo snapshot integrity

def test_undo_snapshot_survives_later_erases(qapp):
    """v1.15.1 shares mask arrays between snapshots (copy-on-write): the
    erase commit REBINDS a new array, so a history entry taken earlier must
    still show exactly the state it captured."""
    cv = _black_canvas(qapp)
    cv.set_tool("brush")
    cv.set_brush_size(30)
    cv.current_effect = "blur"
    cv.current_strength = 10
    _drive(cv, [("press", (60, 80)), ("move", (200, 80)),
                ("release", (200, 80))])
    cv.set_tool("eraser")
    _drive(cv, [("press", (100, 150)), ("move", (140, 150)),
                ("release", (140, 150))])
    e1 = cv.erase_mask
    # the snapshot holding e1 enters the stack at erase #2's PRESS
    _drive(cv, [("press", (240, 150))])
    snap = cv._undo_stack[-1][3]
    assert e1 is snap, "snapshots must share the array (copy-on-write)"
    _drive(cv, [("move", (280, 150)), ("release", (280, 150))])
    assert not np.array_equal(cv.erase_mask, e1)
    assert np.array_equal(snap, e1), "in-place mutation corrupted history"
    cv.undo()
    assert np.array_equal(cv.erase_mask, e1)
    cv.redo()
    assert not np.array_equal(cv.erase_mask, e1)


def test_undo_snapshot_keeps_auto_mask_intact(qapp):
    """Same contract for the auto mask: set -> snapshot -> replace -> undo
    must restore the original detection result byte for byte."""
    cv = _black_canvas(qapp)
    m1 = np.zeros((H, W), dtype=bool)
    m1[40:90, 40:120] = True
    cv.set_auto_mask(m1)                  # pushes undo
    m2 = np.zeros((H, W), dtype=bool)
    m2[150:220, 200:300] = True
    cv.set_auto_mask(m2)                  # pushes undo again
    snap = cv._undo_stack[-1][2]
    assert np.array_equal(snap, m1)
    cv.undo()
    assert np.array_equal(cv.auto_mask, m1)
