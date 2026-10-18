# -*- coding: utf-8 -*-
"""v1.14.3 — two user-reported issues.

1. "用橡皮擦擦过的地方不能再次手动打码" -- compose_image subtracts the erase
   layer from manual selections too (so the eraser can remove hand-drawn
   censoring), but the layer has no notion of time, so a spot that was erased
   could never be censored again. A newly drawn selection now punches through
   the eraser inside its own area.
2. "点击画布之后小键盘方向键不能换图" -- the keypad test must not depend on a
   single Qt flag. Covered here: the real-keypad path (KeypadModifier), the
   NumLock-ON virtual-key path, and the Fn-overlay path where a "numpad" sends
   MAIN-ROW digit codes (no modifier at all), while real arrow keys must stay
   unclaimed so the canvas keeps scrolling.
"""
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from app.core.image_ops import compose_image
from app.core.selection import RectSelection
from app.ui.editor_canvas import EditorCanvas

OPEN = []


@pytest.fixture(autouse=True)
def _close_windows():
    yield
    while OPEN:
        w = OPEN.pop()
        try:
            w.close()
        except Exception:
            pass


@pytest.fixture()
def no_dialogs(monkeypatch):
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes),
                            raising=False)
    monkeypatch.setattr(QDialog, "exec", lambda self: 1, raising=False)


def _checker(w=200, h=150):
    a = (np.indices((h, w)).sum(axis=0) % 2)
    rgb = np.where(a[..., None] == 0, 20, 230).astype(np.uint8)
    return Image.fromarray(np.repeat(rgb, 3, axis=2), "RGB")


def _contrast(img, box=(60, 100, 60, 140)):
    """High-contrast checkerboard pixels in a region; blur kills them."""
    y0, y1, x0, x1 = box
    a = np.array(img.convert("RGB")).astype(int)[y0:y1, x0:x1]
    return int(((a < 60).any(axis=2) | (a > 190).any(axis=2)).sum())


@pytest.fixture()
def canvas(qapp):
    cv = EditorCanvas()
    cv.image_np = np.array(_checker())
    cv.original_pil = _checker()
    cv.sticker_library = {}
    return cv


def _draw_rect(cv, x1, y1, x2, y2):
    """Drive the REAL rect-tool mouse path so the commit code runs."""
    cv.tool = "rect"
    p1 = cv.mapFromScene(QPointF(x1, y1))
    p2 = cv.mapFromScene(QPointF(x2, y2))
    cv.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(p1), QPointF(p1),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier))
    cv.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(p2), QPointF(p2),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier))


# ==================================================== issue 1: eraser hole

def test_manual_censor_works_again_after_erasing(canvas):
    """The reported failure, asserted on composed PIXELS."""
    full = _contrast(_checker())
    assert full > 1000, "fixture sanity"

    # 1) draw a manual censor -> the region is censored
    _draw_rect(canvas, 60, 60, 140, 100)
    assert canvas.selections, "the rect tool did not commit a selection"
    censored = _contrast(compose_image(canvas.original_pil, canvas.selections,
                                       [], [], {}))
    assert censored < full * 0.2, f"not censored ({censored})"

    # 2) erase the same spot -> restored
    canvas.erase_mask = np.zeros(canvas.image_np.shape[:2], dtype=bool)
    canvas.erase_mask[55:105, 55:145] = True
    erased = _contrast(compose_image(canvas.original_pil, canvas.selections,
                                     [], [], {}, erase_mask=canvas.erase_mask))
    assert erased > full * 0.8, f"erase did not restore ({erased})"

    # 3) censoring it AGAIN by hand must work (this used to be impossible)
    _draw_rect(canvas, 60, 60, 140, 100)
    again = _contrast(compose_image(canvas.original_pil, canvas.selections,
                                    [], [], {}, erase_mask=canvas.erase_mask))
    assert again < full * 0.2, (
        f"a newly drawn censor was swallowed by the erase layer ({again})")
    # and the erase layer keeps only the OUTSIDE of the new selection
    assert canvas.erase_mask is not None
    assert not canvas.erase_mask[65:95, 65:135].any(), \
        "the new selection should have punched through the erased area"


def test_erase_outside_a_new_selection_survives(canvas):
    """Punching through must be confined to the new selection's own area."""
    canvas.erase_mask = np.zeros(canvas.image_np.shape[:2], dtype=bool)
    canvas.erase_mask[10:40, 10:40] = True          # far away from the new rect
    _draw_rect(canvas, 100, 80, 160, 120)
    assert canvas.erase_mask[10:40, 10:40].all(), "unrelated erase lost"
    assert not canvas.erase_mask[105:115, 105:155].any()


def test_eraser_still_removes_manual_censoring(canvas):
    """The pre-existing behaviour must not regress: erasing AFTER drawing a
    censor still restores the region (it has a dedicated regression test too)."""
    _draw_rect(canvas, 60, 60, 140, 100)
    canvas.erase_mask = np.zeros(canvas.image_np.shape[:2], dtype=bool)
    canvas._apply_erase_points([(x, y) for x in range(60, 141, 5)
                                for y in range(60, 101, 5)])
    restored = _contrast(compose_image(canvas.original_pil, canvas.selections,
                                       [], [], {}, erase_mask=canvas.erase_mask))
    assert restored > _contrast(_checker()) * 0.7, f"eraser stopped working ({restored})"


# ==================================================== issue 2: numpad keys

class _FakeKey:
    """Minimal stand-in so the virtual-key branch can be exercised: a
    synthesised QKeyEvent always reports nativeVirtualKey() == 0."""
    def __init__(self, key, mods=Qt.KeyboardModifier.NoModifier, vk=0):
        self._k, self._m, self._vk = key, mods, vk

    def key(self):
        return self._k

    def modifiers(self):
        return self._m

    def nativeVirtualKey(self):
        return self._vk


def _win(qapp, tmp_path, monkeypatch, no_dialogs, n=4):
    import app.ui.main_window as mw_mod
    from app.core.config import default_config
    from app.ui.main_window import MainWindow

    class Stub:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(Stub(), default_config())
    OPEN.append(w)
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.fromarray(np.full((120, 160, 3), 40 + i * 40, dtype=np.uint8)).save(p)
        paths.append(str(p))
    w.thumbnail_list.add_paths(paths)
    w._load_path(paths[0], save_current=False)
    qapp.processEvents()
    return w, paths


@pytest.mark.parametrize("key,mods,vk,want", [
    # v1.14.4: the REAL arrow keys, claimed globally
    (Qt.Key_Up, Qt.KeyboardModifier.NoModifier, 0, -1),
    (Qt.Key_Left, Qt.KeyboardModifier.NoModifier, 0, -1),
    (Qt.Key_Down, Qt.KeyboardModifier.NoModifier, 0, 1),
    (Qt.Key_Right, Qt.KeyboardModifier.NoModifier, 0, 1),
    # numeric keypad, NumLock off
    (Qt.Key_Up, Qt.KeyboardModifier.KeypadModifier, 0, -1),
    (Qt.Key_Left, Qt.KeyboardModifier.KeypadModifier, 0, -1),
    (Qt.Key_Down, Qt.KeyboardModifier.KeypadModifier, 0, 1),
    (Qt.Key_Right, Qt.KeyboardModifier.KeypadModifier, 0, 1),
    # numeric keypad, NumLock on
    (Qt.Key_8, Qt.KeyboardModifier.KeypadModifier, 0, -1),
    (Qt.Key_4, Qt.KeyboardModifier.KeypadModifier, 0, -1),
    (Qt.Key_2, Qt.KeyboardModifier.KeypadModifier, 0, 1),
    (Qt.Key_6, Qt.KeyboardModifier.KeypadModifier, 0, 1),
    # identified by virtual key instead of the modifier flag
    (Qt.Key_8, Qt.KeyboardModifier.NoModifier, 0x68, -1),
    (Qt.Key_2, Qt.KeyboardModifier.NoModifier, 0x62, 1),
])
def test_pad_delta_recognises_navigation_keys(qapp, tmp_path, monkeypatch,
                                              no_dialogs, key, mods, vk, want):
    w, _ = _win(qapp, tmp_path, monkeypatch, no_dialogs)
    assert w._pad_delta(_FakeKey(key, mods, vk)) == want


@pytest.mark.parametrize("key,mods,vk", [
    # a modifier makes it somebody else's key
    (Qt.Key_8, Qt.KeyboardModifier.ControlModifier, 0),
    (Qt.Key_Up, Qt.KeyboardModifier.ControlModifier, 0),
    (Qt.Key_8, Qt.KeyboardModifier.ShiftModifier, 0),
    # v1.14.4: TOP-ROW digits are no longer navigation. v1.14.3 guessed they
    # came from Fn-overlay keypads; the user clarified they meant real arrows.
    (Qt.Key_8, Qt.KeyboardModifier.NoModifier, 0),
    (Qt.Key_4, Qt.KeyboardModifier.NoModifier, 0),
    (Qt.Key_2, Qt.KeyboardModifier.NoModifier, 0),
    (Qt.Key_6, Qt.KeyboardModifier.NoModifier, 0),
    # numpad 5 is the centre key; letters are not ours
    (Qt.Key_5, Qt.KeyboardModifier.KeypadModifier, 0),
    (Qt.Key_A, Qt.KeyboardModifier.NoModifier, 0),
])
def test_pad_delta_ignores_non_navigation_keys(qapp, tmp_path, monkeypatch,
                                               no_dialogs, key, mods, vk):
    w, _ = _win(qapp, tmp_path, monkeypatch, no_dialogs)
    assert w._pad_delta(_FakeKey(key, mods, vk)) is None


def test_arrow_keys_navigate_with_canvas_focused(qapp, tmp_path, monkeypatch,
                                                 no_dialogs):
    """The user's scenario: click the canvas, then press an arrow key."""
    w, _ = _win(qapp, tmp_path, monkeypatch, no_dialogs)
    w.show()
    for _ in range(3):
        qapp.processEvents()
    monkeypatch.setattr(w, "isActiveWindow", lambda: True)
    w.canvas.setFocus()
    qapp.processEvents()

    for key, mods, want_row in ((Qt.Key_Up, Qt.KeyboardModifier.NoModifier, 0),
                                (Qt.Key_Left, Qt.KeyboardModifier.NoModifier, 0),
                                (Qt.Key_Down, Qt.KeyboardModifier.NoModifier, 2),
                                (Qt.Key_Right, Qt.KeyboardModifier.NoModifier, 2),
                                (Qt.Key_8, Qt.KeyboardModifier.KeypadModifier, 0),
                                (Qt.Key_2, Qt.KeyboardModifier.KeypadModifier, 2)):
        w.thumbnail_list.setCurrentRow(1)
        ev = QKeyEvent(QEvent.Type.KeyPress, key, mods)
        qapp.sendEvent(w.canvas, ev)
        qapp.processEvents()
        assert w.thumbnail_list.currentRow() == want_row, \
            f"key={key} mods={mods} did not navigate"


def test_numpad_leaves_text_fields_alone(qapp, tmp_path, monkeypatch, no_dialogs):
    """A spin box must keep its keypad (typing a value beats navigation)."""
    from PySide6.QtWidgets import QSpinBox
    w, _ = _win(qapp, tmp_path, monkeypatch, no_dialogs)
    w.show()
    for _ in range(3):
        qapp.processEvents()
    monkeypatch.setattr(w, "isActiveWindow", lambda: True)
    spin = QSpinBox(w)
    spin.setRange(0, 99)
    spin.setValue(10)
    spin.show()                     # a child added after show() needs this to
    spin.setFocus()                 # be focusable at all
    qapp.processEvents()
    assert w.focusWidget() is spin, "fixture sanity: the spin box has no focus"
    w.thumbnail_list.setCurrentRow(1)
    qapp.sendEvent(spin, QKeyEvent(QEvent.Type.KeyPress, Qt.Key_8,
                                   Qt.KeyboardModifier.KeypadModifier))
    qapp.processEvents()
    assert w.thumbnail_list.currentRow() == 1, "navigation stole the keypad"


def test_keydebug_is_gone(qapp, tmp_path, monkeypatch, no_dialogs):
    """v1.15.0 cleanup: the v1.14.3 key diagnostic was retired with its
    question settled -- it must not write a log next to the user config."""
    from app.core import config as config_mod
    w, _ = _win(qapp, tmp_path, monkeypatch, no_dialogs)
    assert not hasattr(w, "_keydebug")
    assert not hasattr(w, "_keydebug_reset")
    monkeypatch.setattr(w, "isActiveWindow", lambda: True)
    w.canvas.setFocus()
    qapp.processEvents()
    qapp.sendEvent(w.canvas, QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up,
                                       Qt.KeyboardModifier.KeypadModifier))
    qapp.processEvents()
    assert not (config_mod.CONFIG_DIR / "keydebug.log").exists()
