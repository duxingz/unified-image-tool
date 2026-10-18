# -*- coding: utf-8 -*-
"""v1.14.2 — asset rename/delete must fix EVERY reference, not just the live one.

Two real bugs found by an end-to-end bug hunt:

1. ``rename_sticker_asset`` only re-pointed ``image_states`` + ``canvas.stickers``.
   The undo/redo snapshots are DEEP COPIES taken at edit time, so after a rename
   undoing a sticker edit restored the dead name and the sticker rendered as
   nothing -- silently, in the preview AND in the exported file.
2. The saved watermark presets / pipelines store the asset NAME as a string, so
   after a rename applying them failed with a misleading "asset no longer
   exists" message instead of working.

Both are asserted on the delivered bytes, not on internal state alone.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from app.core.export import export_batch

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
def dialogs(monkeypatch):
    """Auto-answer dialogs but RECORD the text, so tests can assert on it."""
    seen = {}

    def fake_question(_parent, title, text, *a, **k):
        seen["question"] = text
        return QMessageBox.Yes

    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes),
                            raising=False)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    return seen


class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, *a, **k):
        return []


def _asset(tmp_path, name, color=(255, 0, 0, 255)):
    d = tmp_path / "app" / "stickers"
    d.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (60, 30), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((0, 0, 59, 29), fill=color)
    im.save(d / f"{name}.png")
    return im


def _mk_window(qapp, tmp_path, monkeypatch, name="logo"):
    import app.ui.main_window as mw_mod
    from app.core.config import default_config
    from app.ui.main_window import MainWindow

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    _asset(tmp_path, name)
    w = MainWindow(StubDetector(), default_config())
    OPEN.append(w)
    img = tmp_path / "img.png"
    Image.fromarray(np.full((150, 200, 3), 90, dtype=np.uint8)).save(img)
    w.thumbnail_list.add_paths([str(img)])
    w._load_path(str(img), save_current=False)
    qapp.processEvents()
    return w, img


def _export_red_pixels(w, img, tmp_path, tag):
    """Export through the real batch path and count the sticker's pixels."""
    w.canvas.state_changed.emit()
    QApplication.instance().processEvents()
    out = tmp_path / f"out_{tag}"
    out.mkdir(exist_ok=True)
    export_batch([str(img)], w.image_states, w.config,
                 {"auto_effect": "blur", "auto_strength": 10,
                  "auto_mask_dilate": 0},
                 {"out_dir": str(out), "fmt": "PNG", "quality": 100,
                  "name_mode": "preserve", "scale_mode": "none",
                  "strip_metadata": False}, w.sticker_library or {})
    files = list(out.glob("*.png"))
    assert files, "nothing was exported"
    a = np.array(Image.open(files[0]).convert("RGB")).astype(int)
    return int(((a[:, :, 0] > 150) & (a[:, :, 1] < 100) & (a[:, :, 2] < 100)).sum())


def _drag_once(w):
    """One geometry edit -> exactly one undo snapshot."""
    w.canvas._sel_sticker = 0
    w.canvas._sticker_drag = "move"
    w.canvas._sticker_grab_dx = 0.0
    w.canvas._sticker_grab_dy = 0.0
    w.canvas._s_drag = {"pushed": False}
    w.canvas._drag_sticker_to(QPointF(115, 85))
    w.canvas._sticker_drag = None


# ================================================== bug 1: rename vs history

def test_undo_after_asset_rename_still_renders_and_exports(
        qapp, tmp_path, monkeypatch, dialogs):
    """The reported failure: rename -> undo -> the sticker vanished from the
    EXPORTED file. Asserted on the exported pixels."""
    w, img = _mk_window(qapp, tmp_path, monkeypatch)
    w.canvas.add_sticker("logo", x=100, y=75, width=60, height=30)
    _drag_once(w)
    assert _export_red_pixels(w, img, tmp_path, "before") > 0

    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("brand", True)))
    w.rename_sticker_asset("logo")
    qapp.processEvents()

    w.canvas.undo()
    qapp.processEvents()
    st = w.canvas.stickers[0]
    assert st.name == "brand", f"undo restored a dead asset name {st.name!r}"
    assert st.name in w.sticker_library
    assert _export_red_pixels(w, img, tmp_path, "after") > 0, \
        "the sticker disappeared from the exported file after undo"


def test_manual_undo_stack_is_repointed_too(qapp, tmp_path, monkeypatch, dialogs):
    """Ctrl+Z uses a separate 3-tuple stack -- it must be re-pointed as well."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    w.canvas.add_sticker("logo", x=100, y=75, width=60, height=30)
    _drag_once(w)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("brand", True)))
    w.rename_sticker_asset("logo")
    qapp.processEvents()

    w.canvas.undo_manual()          # the Ctrl+Z path
    qapp.processEvents()
    st = w.canvas.stickers[0]
    assert st.name in w.sticker_library, \
        f"manual undo left a dead name {st.name!r}"


def test_rename_updates_presets_and_pipelines(qapp, tmp_path, monkeypatch,
                                             dialogs):
    """Saved presets/pipelines store the name as a string; they must follow."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, name="wm")
    w.config["watermark_presets"] = [{
        "name": "P1",
        "items": [{"asset": "wm", "anchor": "bottom-right",
                   "size_frac": 0.3, "opacity": 1.0}]}]
    w.config["batch_pipelines"] = [{
        "name": "B1",
        "actions": [{"type": "watermark",
                     "params": {"asset": "wm", "anchor": "center",
                                "size_frac": 0.3, "opacity": 1.0}}]}]
    w.tool_panel._pipe_pending = [{
        "type": "watermark",
        "params": {"asset": "wm", "anchor": "center",
                   "size_frac": 0.3, "opacity": 1.0}}]

    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("wm2", True)))
    w.rename_sticker_asset("wm")
    qapp.processEvents()

    assert w.config["watermark_presets"][0]["items"][0]["asset"] == "wm2"
    assert (w.config["batch_pipelines"][0]["actions"][0]["params"]["asset"]
            == "wm2")
    assert w.tool_panel._pipe_pending[0]["params"]["asset"] == "wm2"

    # ... and the preset now actually applies instead of refusing
    w.tool_panel.wm_preset.addItem("P1", "P1")
    w.tool_panel.wm_preset.setCurrentIndex(
        w.tool_panel.wm_preset.findData("P1"))
    w.apply_watermark_preset()
    qapp.processEvents()
    assert w.canvas.stickers, "the renamed preset applied nothing"


def test_rename_same_or_missing_name_is_safe(qapp, tmp_path, monkeypatch,
                                            dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    w.rename_sticker_asset("does-not-exist")      # must not raise
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("logo", True)))
    w.rename_sticker_asset("logo")                # same name -> no-op
    assert "logo" in w.sticker_library


# ================================================== bug 2: delete vs ghosts

def test_delete_purges_ghost_placements(qapp, tmp_path, monkeypatch, dialogs):
    """Deleting a referenced asset must not leave invisible stickers behind."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, name="trash")
    w.canvas.add_sticker("trash", x=100, y=75, width=60, height=30)
    _drag_once(w)
    assert w.canvas.stickers

    w.delete_sticker_asset("trash")
    qapp.processEvents()
    assert w.canvas.stickers == [], "live ghost sticker left behind"
    for state in w.image_states.values():
        assert not (state or {}).get("stickers")
    for stack in (w.canvas._undo_stack, w.canvas._manual_undo_stack):
        for snap in stack:
            assert not [s for s in snap[1]
                        if getattr(s, "name", None) == "trash"], \
                "a history snapshot still holds the deleted asset"
    # and a later undo cannot resurrect it
    w.canvas.undo()
    qapp.processEvents()
    assert not w.canvas.stickers


def test_delete_warning_names_every_reference(qapp, tmp_path, monkeypatch,
                                             dialogs):
    """The confirm dialog must mention history + presets, not just live use."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, name="trash")
    w.canvas.add_sticker("trash", x=100, y=75, width=60, height=30)
    _drag_once(w)
    w.config["watermark_presets"] = [{
        "name": "P1",
        "items": [{"asset": "trash", "anchor": "center",
                   "size_frac": 0.2, "opacity": 1.0}]}]
    w.delete_sticker_asset("trash")
    text = dialogs.get("question", "")
    assert "编辑历史" in text, text
    assert "水印方案" in text or "方案" in text, text
    assert "trash" in text


def test_asset_name_refs_counts_history_and_config(qapp, tmp_path, monkeypatch,
                                                   dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, name="logo")
    w.canvas.add_sticker("logo", x=100, y=75, width=60, height=30)
    _drag_once(w)
    refs = w._asset_name_refs("logo")
    assert refs["live"] >= 1
    assert refs["history"] >= 1
    assert w._asset_name_refs("nope") == {
        "live": 0, "history": 0, "presets": 0, "pipelines": 0, "pending": 0}
