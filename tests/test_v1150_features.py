# -*- coding: utf-8 -*-
"""v1.15.0 — mosaic-residue fix, sticker free stretch, pipeline action lock,
mini-window unification, dead-code cleanup.

The reported "brush/eraser leaves mosaic residue" had TWO real causes:

a) the pixel-mosaic grid was anchored to the MASK BBOX: partially erasing a
   mosaic shrank the bbox, and the next rebuild re-phased EVERY remaining
   block (measured 167/255 on 79% of remaining pixels) -- the pattern the
   user saw while erasing was replaced at release;
b) the eraser live preview crashed on manual-only censoring (_overlay_mask
   is None there), so erasing a hand-drawn mosaic had NO live feedback at
   all; the release then re-phased per (a).
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog

from app.core.image_ops import compose_image
from app.core.selection import BrushSelection, Sticker, draw_brush_stroke

W, H = 400, 300


def texture():
    rng = np.random.default_rng(11)
    return Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), "RGB")


def _mask(points, width):
    m = Image.new("L", (W, H), 0)
    draw_brush_stroke(ImageDraw.Draw(m), points, width)
    return np.array(m) > 127


def _compose(selections, erase=None):
    return np.array(compose_image(
        texture(), selections, [], [], {}, params={}, erase_mask=erase
    ).convert("RGB"))


# ============================================ a: mosaic grid absolute anchor

def test_partial_erase_keeps_remaining_mosaic_pattern():
    """Erase one END of a pixel-mosaic stroke: the remaining mosaic must be
    byte-identical to the pre-erase mosaic (it used to re-phase 167/255)."""
    sel = BrushSelection([(40, 120), (360, 120)], 24, effect="pixel", strength=12)
    full = _compose([sel])
    erase = np.zeros((H, W), dtype=bool)
    erase[:, :110] = True                      # cut the left end off
    final = _compose([sel], erase=erase)
    sm = np.array(sel.to_mask((W, H)).convert("L")) > 127
    remain = sm & ~erase
    assert remain.sum() > 1000
    diff = np.abs(final.astype(int) - full.astype(int)).max(axis=2)
    assert diff[remain].max() == 0, "remaining mosaic re-phased after erase"
    assert np.abs(final[erase].astype(int) - np.array(texture())[erase].astype(int)).max() == 0


def test_mosaic_blocks_are_cell_aligned():
    """The grid must anchor to absolute block cells: the same content cell
    yields the same block colour no matter where the mask bbox sits."""
    from app.core.effects import apply_pixel_mosaic_mask

    img = np.array(texture())
    mask = np.zeros((H, W), dtype=bool)
    mask[50:150, 30:230] = True
    out = apply_pixel_mosaic_mask(img, mask, block_size=10)
    # the SAME absolute cell inside both masks must get the same block
    # colour, regardless of where each mask's bbox starts
    mask2 = np.zeros((H, W), dtype=bool)
    mask2[60:160, 40:240] = True               # shifted +10/+10 (one block)
    out2 = apply_pixel_mosaic_mask(img, mask2, block_size=10)
    cell = (slice(60, 70), slice(40, 50))      # inside both masks
    c1 = out[cell].reshape(-1, 3).mean(axis=0)
    c2 = out2[cell].reshape(-1, 3).mean(axis=0)
    assert np.allclose(c1, c2, atol=1e-4), "block colour depends on mask bbox"


# ====================================== b: eraser live preview (crash + WYSI)

@pytest.fixture()
def canvas(qapp):
    cv = __import__("app.ui.editor_canvas", fromlist=["EditorCanvas"]).EditorCanvas()
    cv.image_np = np.array(texture())
    cv.original_pil = texture()
    cv.sticker_library = {}
    cv._rebuild()
    return cv


def _event(cv, kind, x, y, mods=Qt.KeyboardModifier.NoModifier):
    """Build a mouse event for SCENE coords (x, y): an unshown QGraphicsView
    is not at an identity transform, so the scene point must be mapped to the
    viewport first (same round trip the real mouse path does)."""
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
    return QMouseEvent(t, QPointF(vp), QPointF(vp), btn, buttons, mods)


def _drive(cv, kinds_and_points, mods=Qt.KeyboardModifier.NoModifier):
    for kind, (x, y) in kinds_and_points:
        ev = _event(cv, kind, x, y, mods)
        if kind == "press":
            cv.mousePressEvent(ev)
        elif kind == "move":
            cv.mouseMoveEvent(ev)
        else:
            cv.mouseReleaseEvent(ev)


def test_eraser_live_preview_works_on_manual_only_censor(canvas):
    """Manual-only censoring -> _overlay_mask is None -> the eraser live
    preview used to raise TypeError on every frame (no feedback at all)."""
    canvas.set_brush_size(24)
    canvas.set_tool("brush")
    canvas.current_effect = "pixel"
    canvas.current_strength = 12
    _drive(canvas, [("press", (60, 150)), ("move", (340, 150)),
                    ("release", (340, 150))])
    assert len(canvas.selections) == 1
    assert canvas._overlay_mask is None
    canvas.set_tool("eraser")
    # must NOT raise, and must actually change the live image
    _drive(canvas, [("press", (100, 150)), ("move", (130, 150))])
    base = canvas._composed_base
    live = canvas._live_img
    assert not np.array_equal(base[140:160, 90:140], live[140:160, 90:140])
    # release commits the erase; erased area returns to the pristine original
    _drive(canvas, [("release", (160, 150))])
    orig = np.array(texture())
    erase_zone = np.array(canvas.erase_mask) if isinstance(canvas.erase_mask, np.ndarray) else canvas.erase_mask
    assert erase_zone[145:155, 100:140].any()
    after = canvas._composed_base
    assert np.abs(after[erase_zone].astype(int) - orig[erase_zone].astype(int)).max() == 0


def test_pixel_preview_matches_final_frame_for_frame(canvas):
    """The block-aligned crop keeps the drag preview equal to the release
    for the pixel effect at every frame (grid phase must not jump)."""
    canvas.set_tool("brush")
    canvas.set_brush_size(24)
    canvas.current_effect = "pixel"
    canvas.current_strength = 12
    pts = [(x, 150) for x in range(60, 341, 8)]
    canvas.mousePressEvent(_event(canvas, "press", *pts[0]))
    for p in pts[1:]:
        canvas.mouseMoveEvent(_event(canvas, "move", *p))
        canvas._render_drag_stroke()
        live = canvas._live_img.copy()
        # final for the stroke drawn SO FAR == live frame
        so_far = pts[:pts.index(p) + 1]
        sel = BrushSelection(so_far, 24, effect="pixel", strength=12)
        final = np.array(compose_image(
            texture(), [sel], [], [], {}, params={}).convert("RGB"))
        diff = np.abs(live.astype(int) - final.astype(int)).max(axis=2)
        stroke_mask = _mask(so_far, 24)
        assert diff[stroke_mask].max() <= 2
    canvas.mouseReleaseEvent(_event(canvas, "release", *pts[-1]))


# ================================================== c: sticker free stretch

def _sticker_canvas(canvas):
    cv = canvas
    lib = {"dot": Image.new("RGBA", (20, 20), (255, 0, 0, 255))}
    cv.sticker_library = lib
    cv.add_sticker("dot", x=150, y=150, width=100, height=100)
    cv._sel_sticker = 0
    cv.set_tool("pointer")
    return cv


def test_corner_drag_stretches_freely(qapp, canvas):
    """dx != dy must stretch width/height INDEPENDENTLY (was ratio-locked)."""
    cv = _sticker_canvas(canvas)
    cv._begin_sticker_drag(QPointF(200, 200))          # br corner at (200,200)
    assert cv._sticker_drag == "br"
    cv._drag_sticker_to(QPointF(260, 210))             # fix=tl(100,100)
    st = cv.stickers[0]
    assert abs(st.width - 160) < 1.0
    assert abs(st.height - 110) < 1.0, "height must follow dy independently"


def test_shift_corner_drag_keeps_ratio(qapp, canvas, monkeypatch):
    cv = _sticker_canvas(canvas)
    monkeypatch.setattr(QApplication, "keyboardModifiers",
                        staticmethod(lambda: Qt.ShiftModifier))
    cv._begin_sticker_drag(QPointF(200, 200))
    cv._drag_sticker_to(QPointF(260, 210))
    st = cv.stickers[0]
    assert abs(st.width / st.height - 1.0) < 0.05, "Shift must keep 1:1"


def test_watermark_stays_proportional(qapp, canvas):
    """The anchor contract: watermark height derives from the source aspect
    (like rotation), so a corner drag must NOT stretch it freely."""
    cv = _sticker_canvas(canvas)
    st = cv.stickers[0]
    st.watermark = True
    st.nx, st.ny, st.size_frac, st.anchor = 0.5, 0.5, 100 / 300, "center"
    cv._begin_sticker_drag(QPointF(200, 200))
    cv._drag_sticker_to(QPointF(260, 210))
    assert abs(st.width / st.height - 1.0) < 0.05


def test_free_stretched_sticker_exports_non_uniform(qapp, canvas):
    """paste path must honour independent width/height at export res too."""
    lib = {"dot": Image.new("RGBA", (20, 20), (255, 0, 0, 255))}
    st = Sticker("dot", 150, 150, width=60, height=120)
    bg = Image.new("RGB", (W, H), (0, 0, 0))   # black bg: only the sticker is red
    img = compose_image(bg, [], [], [st], lib, params={})
    arr = np.array(img.convert("RGB"))
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    ys, xs = np.where(red)
    assert 50 <= xs.max() - xs.min() + 1 <= 64
    assert 110 <= ys.max() - ys.min() + 1 <= 124


# ================================================= d: pipeline action lock

def test_describe_marks_locked_action():
    from app.core.export import describe_pipeline_action
    plain = describe_pipeline_action(
        {"type": "resize", "params": {"percent": 50}})
    locked = describe_pipeline_action(
        {"type": "resize", "params": {"percent": 50}, "locked": True})
    assert locked.startswith("🔒") and not plain.startswith("🔒")


def _panel_win(qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw_mod
    from app.core.config import default_config
    from app.ui.main_window import MainWindow

    class StubDetector:
        def __init__(self, *a, **k):
            pass

        def detect(self, *a, **k):
            return []

        def pick_providers(self):
            return []

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    return MainWindow(StubDetector(), default_config())


def test_locked_action_blocks_editing(qapp, tmp_path, monkeypatch):
    from app.ui import tool_panel as tp_mod
    w = _panel_win(qapp, tmp_path, monkeypatch)
    tp = w.tool_panel
    tp._pipe_pending = [
        {"type": "resize", "params": {"percent": 50}, "locked": True},
        {"type": "convert", "params": {"fmt": "JPG", "quality": 90}},
    ]
    tp.refresh_pipeline_actions()
    assert "🔒" in tp.pipe_actions.item(0).text()
    assert "🔒" not in tp.pipe_actions.item(1).text()

    def _boom(*a, **k):
        raise AssertionError("dialog must not open for a locked action")
    monkeypatch.setattr(tp_mod.ActionConfigDialog, "__init__", _boom)
    tp.pipe_actions.setCurrentRow(0)
    tp._edit_pipeline_action()          # must silently refuse (message box)
    assert tp._pipe_pending[0]["params"]["percent"] == 50
    # unlock via the same code path the context menu uses
    tp._set_pipeline_lock(0, False)
    assert "locked" not in tp._pipe_pending[0]
    assert "🔒" not in tp.pipe_actions.item(0).text()


def test_unlocked_action_edits_again(qapp, tmp_path, monkeypatch):
    """Full loop: locked refuses the dialog -> unlock -> edit applies."""
    w = _panel_win(qapp, tmp_path, monkeypatch)
    tp = w.tool_panel
    tp._pipe_pending = [{"type": "resize", "params": {"percent": 50},
                         "locked": True}]
    tp.refresh_pipeline_actions()
    tp.pipe_actions.setCurrentRow(0)

    class FakeDialog:
        def __init__(self, action_type, params, main, parent=None):
            assert action_type == "resize"
            assert params == {"percent": 50}

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_params(self):
            return {"mode": "percent", "percent": 80, "width": 1920,
                    "height": 1080}

    import app.ui.tool_panel as tp_mod
    monkeypatch.setattr(tp_mod, "ActionConfigDialog", FakeDialog)
    tp._edit_pipeline_action()          # still locked -> refused
    assert tp._pipe_pending[0]["params"]["percent"] == 50
    tp._set_pipeline_lock(0, False)     # unlock via the context-menu path
    tp._edit_pipeline_action()          # now the dialog runs
    assert tp._pipe_pending[0]["params"]["percent"] == 80
    assert "locked" not in tp._pipe_pending[0]


def test_preset_roundtrip_keeps_lock(qapp, tmp_path, monkeypatch):
    w = _panel_win(qapp, tmp_path, monkeypatch)
    tp = w.tool_panel
    w.config["batch_pipelines"] = [{
        "name": "锁定的方案",
        "actions": [{"type": "resize", "params": {"percent": 60},
                     "locked": True}],
        "out_dir": "", "on_conflict": "dedupe",
    }]
    tp.refresh_pipeline_presets()
    ix = tp.pipe_preset.findData("锁定的方案")
    tp.pipe_preset.setCurrentIndex(ix)
    tp.refresh_pipeline_from_preset()
    assert tp._pipe_pending[0]["locked"] is True
    assert "🔒" in tp.pipe_actions.item(0).text()


# ================================== e: mini window uses main-window params

class MiniStubDetector:
    """Records detect kwargs, returns one fixed blob mask."""

    def __init__(self, blob=(60, 60, 160, 160)):
        self.blob = blob
        self.calls = []

    def detect(self, bgr, conf_threshold=0.3, iou_threshold=0.7,
               allowed_classes=None, input_size=1024):
        self.calls.append({
            "conf": conf_threshold, "iou": iou_threshold,
            "classes": list(allowed_classes or []), "size": input_size,
        })
        x1, y1, x2, y2 = self.blob
        m = np.zeros((200, 200), dtype=bool)
        m[y1:y2, x1:x2] = True
        return [type("D", (), {"mask": m})()]


def test_mini_uses_main_window_auto_params(qapp, tmp_path):
    from app.ui.mini_window import MiniWindow
    src = tmp_path / "in.png"
    texture().resize((200, 200)).save(src)
    cfg = {
        "conf_threshold": 0.42, "iou_threshold": 0.66,
        "selected_regions": ["vagina"], "detect_size": 640,
        "auto_effect": "pixel", "auto_strength": 9, "auto_mask_dilate": 0,
        "params": {},
    }
    det = MiniStubDetector()
    mini = MiniWindow(det, cfg)
    out, n = mini._compose_censored(str(src))
    assert n == 1 and out is not None
    # detection received the MAIN-WINDOW parameters, not hardcoded defaults
    assert det.calls[0] == {"conf": 0.42, "iou": 0.66,
                            "classes": ["vagina"], "size": 640}
    # the censor actually applied the configured pixel effect
    orig = np.array(Image.open(src).convert("RGB"))
    diff = np.abs(np.array(out.convert("RGB")).astype(int) - orig.astype(int)).max(axis=2)
    assert diff[70:150, 80:140].max() > 40
    assert diff[:40, :40].max() == 0


def test_mini_reports_no_detection(qapp, tmp_path):
    from app.ui.mini_window import MiniWindow
    src = tmp_path / "in.png"
    texture().resize((200, 200)).save(src)
    det = MiniStubDetector()
    det.blob = (0, 0, 0, 0)          # empty mask -> union stays empty
    mini = MiniWindow(det, {"selected_regions": ["vagina"]})
    out, n = mini._compose_censored(str(src))
    assert out is None and n == 0


# ================================================ f: dead-code cleanup locks

def test_retired_symbols_are_gone():
    import app.core.config as config_mod
    import app.core.image_ops as image_ops
    import app.ui.editor_canvas as ec_mod

    assert not hasattr(config_mod, "save_preset")
    assert not hasattr(config_mod, "load_preset")
    assert not hasattr(image_ops, "convert_image")
    assert not hasattr(image_ops, "load_thumbnail")
    assert not hasattr(ec_mod.EditorCanvas, "fit_view")
    assert not hasattr(ec_mod.EditorCanvas, "apply_current_effect")
    assert not hasattr(ec_mod.EditorCanvas, "get_auto_mask")
    assert not hasattr(ec_mod.EditorCanvas, "add_auto_mask")
    assert not hasattr(ec_mod.EditorCanvas, "duplicate_to_mask_selection")
    cfg = config_mod.default_config()
    for gone in ("mosaic_type", "output", "presets"):
        assert gone not in cfg
    with pytest.raises(ModuleNotFoundError):
        import app.core.processor  # noqa: F401
