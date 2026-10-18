# -*- coding: utf-8 -*-
"""v1.10.0 watermark feature tests.

Watermarks reuse the sticker system with anchor+fraction placement so the
same watermark lands at the same spot regardless of image resolution and
orientation, stays independently movable/resizable on each image, survives
full-resolution export, and is reported through a dedicated badge.
"""
import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.export import compose_from_state, export_batch
from app.core.image_ops import render_text_watermark
from app.core.selection import Sticker


def _src(tmp_path, w=320, h=240, name="src.png"):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[:, :, 1] = 120
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return str(p)


def _auto():
    return {"auto_effect": "blur", "auto_strength": 8, "auto_mask_dilate": 0}


def _cfg(tmp_path, tag):
    return {"out_dir": str(tmp_path / tag), "name_mode": "preserve",
            "prefix": "OUT", "start": 1, "digits": 3, "fmt": "PNG",
            "quality": 100, "scale_mode": "none", "percent": 100,
            "width": 1000, "height": 1000, "strip_metadata": False}


@pytest.fixture()
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))


# ------------------------------------------------------------- placement
@pytest.mark.parametrize("w,h", [(1080, 1920), (1920, 1080)])
def test_watermark_bottom_right_any_orientation(tmp_path, w, h):
    src = _src(tmp_path, w=w, h=h, name=f"o{w}x{h}.png")
    st = Sticker("wm", 0, 0, 10, 10, nx=1.0, ny=1.0, size_frac=0.2,
                 anchor="bottom-right", watermark=True)
    lib = {"wm": Image.new("RGBA", (200, 100), (255, 0, 0, 255))}
    out = compose_from_state(src, {src: {
        "pil": Image.new("RGB", (w, h), (200, 120, 0)),
        "rgb": np.zeros((h, w, 3), dtype=np.uint8),
        "selections": [], "stickers": [st],
        "auto_mask": np.zeros((h, w), dtype=bool), "orig_size": (w, h),
    }}, lib, _auto(), {})
    a = np.array(out)
    m = (a[:, :, 0] > 200) & (a[:, :, 1] < 80)
    ys, xs = np.where(m)
    assert xs.min() > w * 0.7  # anchored right
    assert xs.max() == w - 1   # touches the right edge
    assert ys.max() == h - 1   # touches the bottom edge
    # size = 0.2 * SHORT side (design: watermark scales with the short edge)
    assert (xs.max() - xs.min() + 1) == pytest.approx(min(w, h) * 0.2, rel=0.1)


def test_watermark_top_left_anchor(tmp_path):
    w, h = 400, 300
    src = _src(tmp_path, w=w, h=h)
    st = Sticker("wm", 0, 0, 10, 10, nx=0.0, ny=0.0, size_frac=0.1,
                 anchor="top-left", watermark=True)
    lib = {"wm": Image.new("RGBA", (200, 100), (255, 0, 0, 255))}
    out = compose_from_state(src, {src: {
        "pil": Image.new("RGB", (w, h), (200, 120, 0)),
        "rgb": np.zeros((h, w, 3), dtype=np.uint8),
        "selections": [], "stickers": [st],
        "auto_mask": np.zeros((h, w), dtype=bool), "orig_size": (w, h),
    }}, lib, _auto(), {})
    a = np.array(out)
    m = (a[:, :, 0] > 200) & (a[:, :, 1] < 80)
    ys, xs = np.where(m)
    assert xs.min() == 0
    assert ys.min() == 0


def test_plain_sticker_unaffected(tmp_path):
    """Legacy plain stickers still use their pixel position/size."""
    src = _src(tmp_path, w=320, h=240)
    st = Sticker("wm", 160, 120, 40, 20)  # not a watermark
    lib = {"wm": Image.new("RGBA", (200, 100), (255, 0, 0, 255))}
    out = compose_from_state(src, {src: {
        "pil": Image.new("RGB", (320, 240), (200, 120, 0)),
        "rgb": np.zeros((240, 320, 3), dtype=np.uint8),
        "selections": [], "stickers": [st],
        "auto_mask": np.zeros((240, 320), dtype=bool), "orig_size": (320, 240),
    }}, lib, _auto(), {})
    a = np.array(out)
    m = (a[:, :, 0] > 200) & (a[:, :, 1] < 80)
    ys, xs = np.where(m)
    assert xs.min() == 140 and xs.max() == 179  # 40 wide, centered at 160
    assert ys.min() == 110 and ys.max() == 129  # 20 tall, centered at 120


def test_watermark_opacity(tmp_path):
    src = _src(tmp_path)
    st = Sticker("wm", 0, 0, 10, 10, nx=1.0, ny=1.0, size_frac=0.2,
                 anchor="bottom-right", watermark=True, opacity=0.5)
    lib = {"wm": Image.new("RGBA", (200, 100), (255, 0, 0, 255))}
    out = compose_from_state(src, {src: {
        "pil": Image.new("RGB", (320, 240), (200, 120, 0)),
        "rgb": np.zeros((240, 320, 3), dtype=np.uint8),
        "selections": [], "stickers": [st],
        "auto_mask": np.zeros((240, 320), dtype=bool), "orig_size": (320, 240),
    }}, lib, _auto(), {})
    a = np.array(out)
    # opaque would give channel (255,0,0); at 50% over (200,120,0) -> ~(228,60,0)
    m = (a[:, :, 0] > 200) & (a[:, :, 0] < 250) & (a[:, :, 1] < 80)
    ys, xs = np.where(m)
    assert len(xs) > 0  # faded red present


# --------------------------------------------------------------- text wm
def test_render_text_watermark():
    layer = render_text_watermark("Hello水印", font_size=48,
                                  color=(255, 255, 255, 255), opacity=0.5)
    assert layer.mode == "RGBA"
    assert min(layer.size) > 10
    ext = layer.split()[-1].getextrema()
    assert ext[0] == 0 and 0 < ext[1] <= 128  # dimmed to ~50%


def test_render_text_watermark_empty():
    layer = render_text_watermark("   ")
    assert layer.size == (1, 1)


# ------------------------------------------------------------ export path
def test_watermark_export_skips_fastpath(tmp_path):
    """A watermarked image must NOT take the byte-copy fast path (the export
    must contain the watermark)."""
    src = _src(tmp_path)
    st = Sticker("wm", 0, 0, 10, 10, nx=1.0, ny=1.0, size_frac=0.2,
                 anchor="bottom-right", watermark=True)
    lib = {"wm": Image.new("RGBA", (200, 100), (255, 0, 0, 255))}
    states = {src: {
        "pil": Image.new("RGB", (320, 240), (200, 120, 0)),
        "rgb": np.zeros((240, 320, 3), dtype=np.uint8),
        "selections": [], "stickers": [st],
        "auto_mask": np.zeros((240, 320), dtype=bool), "orig_size": (320, 240),
    }}
    cfg = _cfg(tmp_path, "out")
    r = export_batch([src], states, default_config(), _auto(), cfg, lib)
    assert r["processed"] == 1
    with open(src, "rb") as f1, open(tmp_path / "out" / "src.png", "rb") as f2:
        assert f1.read() != f2.read()  # must be re-encoded, not copied
    a = np.array(Image.open(tmp_path / "out" / "src.png").convert("RGB"))
    m = (a[:, :, 0] > 200) & (a[:, :, 1] < 80)
    assert m.any()  # watermark actually painted


# ------------------------------------------------------------- UI flows
class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        return []


def _mk_window(qapp, tmp_path, no_dialogs, monkeypatch, n=3):
    """MainWindow with the sticker library folder redirected to tmp_path so
    text-watermark persistence never touches the real app/stickers dir."""
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow
    from pathlib import Path
    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(StubDetector(), default_config())
    paths = []
    for i in range(n):
        arr = np.zeros((240, 320, 3), dtype=np.uint8)
        arr[:, :] = 60 + i * 30
        p = tmp_path / f"img{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    w.thumbnail_list.add_paths(paths)
    w._load_path(paths[0], save_current=False)
    return w, paths


def test_batch_apply_watermark_all(qapp, tmp_path, no_dialogs, monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, no_dialogs, monkeypatch)
    w.tool_panel.wm_text.setText("WM")
    w.create_text_watermark_from_panel()
    name = w.tool_panel.wm_asset.currentData()
    assert name == "WM"
    w.apply_watermark(scope="all", name=name, anchor="bottom-right",
                      size_frac=0.15, opacity=0.7)
    for p in paths:
        sts = w.image_states[p].get("stickers", [])
        assert len(sts) == 1
        assert sts[0].watermark and sts[0].size_frac == pytest.approx(0.15)
    # stats bar shows the dedicated watermark count
    assert "已水印 3" in w.thumb_stats_label.text()


def test_watermark_independent_per_image(qapp, tmp_path, no_dialogs, monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, no_dialogs, monkeypatch)
    w.create_text_watermark("WM", color=(255, 255, 255, 255))
    w.apply_watermark(scope="all", name="WM", anchor="bottom-right",
                      size_frac=0.15, opacity=1.0)
    # move the watermark on image 0 only; image 1 must be untouched
    st0 = w.image_states[paths[0]]["stickers"][0]
    st1 = w.image_states[paths[1]]["stickers"][0]
    st0.nx, st0.ny = 0.3, 0.3
    assert st1.nx == 1.0 and st1.ny == 1.0


def test_thumb_badge_marks_watermark(qapp, tmp_path, no_dialogs, monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, no_dialogs, monkeypatch)
    marks = w._thumb_marks(paths[0])
    assert not marks.get("watermark") and not marks.get("sticker")
    w.create_text_watermark("WM", color=(255, 255, 255, 255))
    w.apply_watermark(scope="current", name="WM", anchor="bottom-right",
                      size_frac=0.1, opacity=1.0)
    marks = w._thumb_marks(paths[0])
    assert marks["watermark"] is True
    assert marks["sticker"] is False  # watermarks are NOT plain stickers


def test_canvas_select_move_resize(qapp, tmp_path, no_dialogs, monkeypatch):
    """pointer tool: click selects, drag moves, corner scales (uniform)."""
    from PySide6.QtCore import QPointF
    # drain any stale cross-thread signals (e.g. a worker from an earlier
    # module) so they cannot fire mid-test and move the canvas state
    for _ in range(5):
        qapp.processEvents()
    w, paths = _mk_window(qapp, tmp_path, no_dialogs, monkeypatch)
    try:
        w.create_text_watermark("WM", color=(255, 0, 0, 255))
        # watermark is in the library but library images come from
        # _load_stickers (disk); inject directly like the app does
        w.canvas.sticker_library["WM"] = w.sticker_library["WM"]
        w.canvas.add_sticker("WM")
        assert len(w.canvas.stickers) == 1
        st = w.canvas.stickers[0]
        n_undo = len(w.canvas._undo_stack)
        r0 = w.canvas._sticker_rect_scene(st)
        # click selects without ADDING a no-op undo (add_sticker pushed 1)
        assert w.canvas._begin_sticker_drag(QPointF(r0.center().x(), r0.center().y()))
        assert w.canvas._sel_sticker == 0
        w.canvas._sticker_drag = None
        assert len(w.canvas._undo_stack) == n_undo
        # drag moves it (relative: from the rect center by +80,+60)
        c0 = r0.center()
        target = QPointF(c0.x() + 80, c0.y() + 60)
        w.canvas._begin_sticker_drag(QPointF(c0.x(), c0.y()))
        w.canvas._drag_sticker_to(target)
        assert (st.x, st.y) == pytest.approx((target.x(), target.y()))
        assert len(w.canvas._undo_stack) == n_undo + 1
        # corner handle FREE-stretches since v1.15.0 (grab BR, pull +40/+30
        # -> width +40, height +30 independently; Shift would keep the ratio)
        w.canvas._sticker_drag = None
        r = w.canvas._sticker_rect_scene(st)
        ok = w.canvas._begin_sticker_drag(QPointF(r.right(), r.bottom()))
        assert ok and w.canvas._sticker_drag == "br"
        w.canvas._drag_sticker_to(QPointF(r.right() + 40, r.bottom() + 30))
        assert st.width == pytest.approx(r.width() + 40)
        assert st.height == pytest.approx(r.height() + 30)  # free stretch
        # Ctrl+Z (manual undo) restores; re-fetch the sticker after undoing
        w.canvas.undo_manual()
        st2 = w.canvas.stickers[0]
        assert (st2.x, st2.y) == pytest.approx((target.x(), target.y()))
        assert st2.width == pytest.approx(r.width())
    finally:
        w.close()
