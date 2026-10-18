# -*- coding: utf-8 -*-
"""v1.10.1 watermark bug-fix regressions (user-reported).

1. Watermark could not be MOVED: dragging updated nx/ny but the anchor stayed
   set and place() ignores nx/ny for anchored stickers, so the watermark
   snapped back to its corner every frame. First manual move now detaches the
   anchor into a free (nx/ny) position.
2. Watermarks and stickers are ONE system: text watermark assets are
   persisted into the sticker library folder and survive restarts.
3. Could not switch images after batch-applying watermarks: batch ops append
   to states of never-opened images (no "pil"/"orig_size"), and _load_path
   crashed on those incomplete states. Incomplete states now load properly
   and preserve the appended content.
"""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from PySide6.QtCore import QPointF


@pytest.fixture()
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))


class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        return []


def _mk_window(qapp, tmp_path, monkeypatch, n=3):
    """MainWindow with the sticker library folder redirected to tmp_path."""
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow
    from app.core.config import default_config

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(StubDetector(), default_config())
    paths = []
    for i in range(n):
        arr = np.full((240, 320, 3), 60 + i * 30, dtype=np.uint8)
        p = tmp_path / f"img{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    w.thumbnail_list.add_paths(paths)
    w._load_path(paths[0], save_current=False)
    return w, paths


def _move(canvas, st, to_xy):
    """Simulate a pointer drag of sticker st to scene point to_xy."""
    r = canvas._sticker_rect_scene(st)
    canvas._begin_sticker_drag(QPointF(r.center().x(), r.center().y()))
    canvas._drag_sticker_to(QPointF(to_xy[0], to_xy[1]))
    canvas._sticker_drag = None


# ------------------------------------------------- bug 1: movable watermark
def test_watermark_move_detaches_anchor(qapp, tmp_path, no_dialogs,
                                        monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.create_text_watermark("WM", color=(255, 255, 255, 255))
    w.apply_watermark(scope="all", name="WM", anchor="bottom-right",
                      size_frac=0.2, opacity=1.0)
    w.thumbnail_list.setCurrentRow(1)

    st = w.canvas.stickers[0]
    _move(w.canvas, st, (100, 80))
    r = w.canvas._sticker_rect_scene(st)
    assert st.anchor == "center"  # free mode after first drag
    assert (round(r.center().x()), round(r.center().y())) == (100, 80)

    # second drag also follows (no residual anchor math)
    _move(w.canvas, st, (200, 150))
    r2 = w.canvas._sticker_rect_scene(st)
    assert (round(r2.center().x()), round(r2.center().y())) == (200, 150)

    # the moved position is relative: it survives a state round-trip
    w.canvas.state_changed.emit()  # save current state
    st_state = w.image_states[w.current_path]["stickers"][0]
    assert st_state.anchor == "center"
    assert st_state.nx == pytest.approx(200 / 320)
    w.close()


def test_watermark_resize_after_move(qapp, tmp_path, no_dialogs, monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.create_text_watermark("WM", color=(255, 255, 255, 255))
    w.apply_watermark(scope="all", name="WM", anchor="bottom-right",
                      size_frac=0.2, opacity=1.0)
    w.thumbnail_list.setCurrentRow(1)
    st = w.canvas.stickers[0]
    _move(w.canvas, st, (100, 80))

    r = w.canvas._sticker_rect_scene(st)
    ok = w.canvas._begin_sticker_drag(QPointF(r.right(), r.bottom()))
    assert ok and w.canvas._sticker_drag == "br"
    w.canvas._drag_sticker_to(QPointF(r.right() + 30, r.bottom() + 15))
    w.canvas._sticker_drag = None
    r2 = w.canvas._sticker_rect_scene(st)
    assert r2.width() == pytest.approx(r.width() + 30)
    assert st.size_frac == pytest.approx(0.2 + 30 / 240)  # short edge 240

    # undo restores both position and size
    w.canvas.undo_manual()
    st2 = w.canvas.stickers[0]
    assert (st2.x, st2.y) == pytest.approx((100, 80))
    assert st2.size_frac == pytest.approx(0.2)
    w.close()


def test_delete_selected_watermark(qapp, tmp_path, no_dialogs, monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.create_text_watermark("WM", color=(255, 255, 255, 255))
    w.apply_watermark(scope="current", name="WM", anchor="bottom-right",
                      size_frac=0.2, opacity=1.0)
    assert len(w.canvas.stickers) == 1
    w.canvas._sel_sticker = 0          # select (the old _set_sel_sticker was dead app code)
    w.canvas.delete_selected()
    assert len(w.canvas.stickers) == 0
    # canvas -> state sync removed it from the state too
    w.canvas.state_changed.emit()
    assert len(w.image_states[paths[0]].get("stickers", [])) == 0
    # undo brings the watermark back
    w.canvas.undo()
    assert len(w.canvas.stickers) == 1
    w.close()


# ------------------------------------------ bug 2: one system (persistence)
def test_text_watermark_persisted_and_reloadable(qapp, tmp_path, no_dialogs,
                                                 monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.create_text_watermark("持久水印", color=(255, 255, 255, 255))
    persisted = tmp_path / "app" / "stickers" / "持久水印.png"
    assert persisted.exists()
    w.close()

    # a fresh window (restart simulation) loads the asset from disk
    w2, _ = _mk_window(qapp, tmp_path, monkeypatch)
    assert "持久水印" in w2.sticker_library
    w2.close()


def test_center_anchor_watermark_starts_middle(qapp, tmp_path, no_dialogs,
                                               monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.create_text_watermark("WM", color=(255, 255, 255, 255))
    w.apply_watermark(scope="current", name="WM", anchor="center",
                      size_frac=0.1, opacity=1.0)
    st = w.canvas.stickers[0]
    r = w.canvas._sticker_rect_scene(st)
    # image 320x240 -> center (160,120)
    assert (round(r.center().x()), round(r.center().y())) == (160, 120)
    assert (st.nx, st.ny) == (0.5, 0.5)
    w.close()


# ------------------------------------- bug 3: switching to unopened images
def test_switch_to_never_opened_image_keeps_watermark(qapp, tmp_path,
                                                      no_dialogs, monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.create_text_watermark("WM", color=(255, 255, 255, 255))
    # batch-apply to ALL (only img0 was ever loaded; img1/img2 have partial
    # states carrying just the watermark sticker)
    w.apply_watermark(scope="all", name="WM", anchor="bottom-right",
                      size_frac=0.2, opacity=1.0)
    for i in (1, 2):
        w.thumbnail_list.setCurrentRow(i)
        assert Path(w.current_path).name == Path(paths[i]).name
        # canvas loaded and the watermark survived the first open
        assert w.canvas.image_np is not None
        assert len(w.canvas.stickers) == 1
        assert w.canvas.stickers[0].watermark
        # state is now complete
        state = w.image_states[paths[i]]
        assert "pil" in state and "orig_size" in state
    w.close()


def test_partial_state_load_failure_preserves_state(qapp, tmp_path,
                                                    no_dialogs, monkeypatch):
    """If the file itself fails to load, a partial state must not be lost."""
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow
    from app.core.config import default_config
    from app.core.selection import Sticker

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(StubDetector(), default_config())
    missing = tmp_path / "missing.png"
    partial = {"stickers": [Sticker("WM", 0, 0, 10, 10, nx=0.5, ny=0.5,
                                    size_frac=0.2, anchor="center",
                                    watermark=True)]}
    w.image_states[str(missing)] = partial
    w.thumbnail_list.add_paths([str(missing)])
    w.thumbnail_list.setCurrentRow(0)  # triggers _load_path -> load failure
    # failed load: current_path must NOT switch, partial state intact
    assert w.current_path is None
    assert w.image_states[str(missing)] is partial
    w.close()
