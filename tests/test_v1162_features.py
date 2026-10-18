# -*- coding: utf-8 -*-
"""v1.16.2 — sticker-drag throttling, zero-copy eraser source, dead token."""
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPointF

from app.ui.editor_canvas import EditorCanvas
from tests.test_v1150_features import _drive
from tests.test_v1151_features import _black_canvas

W, H = 400, 300


def test_sticker_drag_renders_are_throttled(qapp):
    """A sticker drag used to recompose the WHOLE image per mouse event
    (100+ Hz raw events); the 30ms throttle must collapse them, and the
    release must flush the last pending position exactly."""
    cv = _black_canvas(qapp)
    cv.sticker_library = {"dot": Image.new("RGBA", (20, 20), (255, 0, 0, 255))}
    cv.add_sticker("dot", x=150, y=150, width=100, height=100)
    cv._sel_sticker = 0
    cv.set_tool("pointer")
    calls = []
    real = cv._make_preview_image
    cv._make_preview_image = lambda: (calls.append(1), real())[1]

    cv.mousePressEvent(_ev(cv, "press", 200, 200))       # br handle
    assert cv._sticker_drag == "br"
    for i in range(1, 11):                               # 10 rapid events
        cv.mouseMoveEvent(_ev(cv, "move", 200 + i * 2, 200 + i))
    n_during = len(calls)
    assert n_during <= 3, f"throttle failed: {n_during} recomposes for 10 events"
    cv.mouseReleaseEvent(_ev(cv, "release", 260, 240))
    # the release applies its own position: geometry is EXACT
    st = cv.stickers[0]
    assert abs(st.width - 160) < 1.0 and abs(st.height - 140) < 1.0


def _ev(cv, kind, x, y):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    ev_map = {
        "press": (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton,
                  Qt.MouseButton.LeftButton),
        "move": (QEvent.Type.MouseMove, Qt.MouseButton.NoButton,
                 Qt.MouseButton.LeftButton),
        "release": (QEvent.Type.MouseButtonRelease, Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton),
    }
    t, btn, buttons = ev_map[kind]
    vp = cv.mapFromScene(QPointF(x, y))
    return QMouseEvent(t, QPointF(vp), QPointF(vp), btn, buttons,
                       Qt.KeyboardModifier.NoModifier)


def test_image_np_stays_pristine_through_eraser(qapp):
    """v1.16.2 uses image_np directly as the eraser's restore source (zero
    copy): it must therefore survive the whole drag byte-for-byte."""
    cv = _black_canvas(qapp)
    pristine = cv.image_np.copy()
    cv.set_tool("brush")
    cv.set_brush_size(24)
    cv.current_effect = "pixel"
    cv.current_strength = 12
    _drive(cv, [("press", (60, 150)), ("move", (340, 150)),
                ("release", (340, 150))])
    cv.set_tool("eraser")
    _drive(cv, [("press", (100, 150)), ("move", (160, 150)),
                ("release", (200, 150))])
    assert np.array_equal(cv.image_np, pristine), \
        "image_np was mutated -- the zero-copy eraser source is unsafe"
    assert cv.erase_mask.any()


def test_canvas_backdrop_token_is_gone():
    """v1.16.2 dead-code sweep: CANVAS_BACKDROP had no consumer anywhere."""
    import app.ui.theme as theme
    assert not hasattr(theme, "CANVAS_BACKDROP")
