# -*- coding: utf-8 -*-
"""v1.12.0 UI enhancements borrowed from nEO iMAGING:

- Sticker library as a thumbnail wall (icon mode with previews).
- Visual 3x3 anchor picker replacing the anchor dropdown (watermark tab and
  pipeline action dialog).
- Hold-to-compare: canvas can show the pristine original (button hold or
  SPACE key), overlays/handles hidden while held.
- Pipeline pre-run preview renders through a deep-copied state so the
  workspace is never mutated by previewing.
- Common size presets persisted in config, usable from the resize tab and
  the pipeline resize action dialog.
"""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from types import SimpleNamespace


@pytest.fixture()
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))


def _mk_window(qapp, tmp_path, monkeypatch, n=3):
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow
    from app.core.config import default_config

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(_StubDetector(), default_config())
    paths = []
    for i in range(n):
        arr = np.full((600, 400, 3), 60 + i * 30, dtype=np.uint8)
        p = tmp_path / f"photo{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    w.thumbnail_list.add_paths(paths)
    w._load_path(paths[0], save_current=False)
    return w, paths


class _StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        h, w = image_bgr.shape[:2]
        m = np.zeros((h, w), dtype=bool)
        m[10:40, 10:40] = True
        return [SimpleNamespace(mask=m)]


# ------------------------------------------------------------ anchor picker
def test_anchor_picker_roundtrip(qapp):
    from app.ui.widgets import AnchorPicker
    picker = AnchorPicker()
    assert picker.anchor() == "bottom-right"
    picker.set_anchor("top-left")
    assert picker.anchor() == "top-left"
    picker.set_anchor("center")
    assert picker.anchor() == "center"


def test_watermark_tab_uses_anchor_picker(qapp, tmp_path, no_dialogs,
                                          monkeypatch):
    from PySide6.QtWidgets import QListWidget
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    # watermark tab config flows from the visual picker
    w.tool_panel.wm_anchor_picker.set_anchor("top-left")
    assert w.tool_panel.current_watermark_config()["anchor"] == "top-left"
    # sticker library is a thumbnail wall with icons
    w.create_text_watermark("署名", color=(255, 255, 255, 255))
    w.tool_panel.refresh_sticker_list()
    assert w.tool_panel.sticker_list.count() >= 1
    assert w.tool_panel.sticker_list.viewMode() == QListWidget.IconMode
    assert not w.tool_panel.sticker_list.item(0).icon().isNull()
    w.close()


# ---------------------------------------------------------- hold-to-compare
def test_show_original_toggles_and_restores(qapp, tmp_path, no_dialogs,
                                            monkeypatch):
    from PySide6.QtCore import QPointF
    from app.core.selection import RectSelection
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    # add real censor content so original vs composed differ
    w.canvas.selections.append(
        RectSelection(10, 10, 60, 60, effect="black_lines", strength=10))
    w.canvas._rebuild()
    assert w.canvas.is_showing_original() is False
    w.canvas.set_show_original(True)
    assert w.canvas.is_showing_original() is True
    # scene rebuilt with the pristine original: no censor black lines
    # toggle back and forth without exceptions
    w.canvas.set_show_original(False)
    w.canvas.set_show_original(True)
    w.canvas.set_show_original(False)
    assert w.canvas.is_showing_original() is False
    w.close()


def test_space_key_holds_original(qapp, tmp_path, no_dialogs, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)

    def key(type_):
        return QKeyEvent(type_, Qt.Key_Space, Qt.KeyboardModifier.NoModifier)

    w.canvas.keyPressEvent(key(QEvent.KeyPress))
    assert w.canvas.is_showing_original() is True
    w.canvas.keyReleaseEvent(key(QEvent.KeyRelease))
    assert w.canvas.is_showing_original() is False
    w.close()


# ------------------------------------------------------ pipeline preview
def test_pipeline_preview_render_leaves_workspace_untouched(
        qapp, tmp_path, no_dialogs, monkeypatch):
    import copy as _copy
    from app.core.export import run_pipeline_on_image
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.create_text_watermark("署名", color=(255, 255, 255, 255))
    w.tool_panel.refresh_watermark_assets()
    actions = [
        {"type": "auto_censor", "params": {"conf": 0.3, "iou": 0.7,
                                           "classes": []}},
        {"type": "watermark",
         "params": {"asset": "署名", "anchor": "bottom-right",
                    "size_frac": 0.15, "opacity": 1.0}},
        {"type": "resize", "params": {"mode": "percent", "percent": 50}},
        {"type": "convert", "params": {"fmt": "PNG", "quality": 100,
                                       "strip": False}},
    ]

    def render(p):
        single = {p: _copy.deepcopy(w.image_states.get(p, {}))}
        conv, _, _, _, _ = run_pipeline_on_image(
            p, single, actions, w.sticker_library,
            w.tool_panel.auto_settings(), {}, w.detector, index=0)
        return conv

    # preview an image that was NEVER opened in the canvas
    img = render(paths[1])
    assert (img.width, img.height) == (200, 300)  # 50% of 400x600
    # the preview must not leak into the workspace state
    st = w.image_states.get(paths[1], {})
    assert len(st.get("stickers", [])) == 0
    assert not (st.get("auto_mask") is not None and st["auto_mask"].any())
    w.close()


def test_pipeline_preview_dialog_flips(qapp, tmp_path, no_dialogs):
    from PIL import Image as PILImage
    from PySide6.QtWidgets import QDialog
    from app.ui.widgets import PipelinePreviewDialog
    paths = []
    for i in range(2):
        p = tmp_path / f"p{i}.png"
        PILImage.new("RGB", (80, 60), (i * 40, 0, 0)).save(p)
        paths.append(str(p))
    calls = []

    def render(p):
        calls.append(p)
        return PILImage.new("RGB", (80, 60), (255, 0, 0))

    dlg = PipelinePreviewDialog(paths, render)
    n0 = len(calls)
    dlg._step(1)
    assert calls[-1] == paths[1]  # next renders the new image
    dlg._step(1)  # wraps to index 0 -> served from cache, no re-render
    assert len(calls) == n0 + 1  # cached: no second render of paths[0]
    assert dlg._index == 0
    dlg.close()


# --------------------------------------------------------- size presets
def test_resize_size_presets_roundtrip(qapp, tmp_path, no_dialogs,
                                       monkeypatch):
    w, paths = _mk_window(qapp, tmp_path, monkeypatch)
    w.tool_panel.resize_width.setValue(1280)
    w.tool_panel.resize_height.setValue(720)
    w.tool_panel._save_resize_preset()
    assert w.config["resize_size_presets"] == ["1280×720"]
    # duplicate save does not duplicate the entry
    w.tool_panel._save_resize_preset()
    assert w.config["resize_size_presets"] == ["1280×720"]
    # selecting applies to the spinboxes
    w.tool_panel.resize_width.setValue(999)
    w.tool_panel.resize_presets.setCurrentIndex(0)  # force change
    w.tool_panel.resize_presets.setCurrentIndex(1)
    assert w.tool_panel.resize_width.value() == 1280
    assert w.tool_panel.resize_height.value() == 720
    # delete removes it
    w.tool_panel._delete_resize_preset()
    assert w.config["resize_size_presets"] == []
    w.close()


def test_pipeline_resize_dialog_uses_presets(qapp, tmp_path, no_dialogs,
                                             monkeypatch):
    from app.ui.tool_panel import ActionConfigDialog
    import app.ui.main_window as mw_mod
    from app.core.config import default_config
    monkeypatch.setattr(mw_mod, "resource_path",
                        lambda rel: tmp_path / rel)
    from app.ui.main_window import MainWindow
    w = MainWindow(_StubDetector(), default_config())
    w.config["resize_size_presets"] = ["800×600"]
    dlg = ActionConfigDialog("resize", None, w)
    dlg.size_preset.setCurrentIndex(0)
    assert dlg.width.value() == 800
    assert dlg.height.value() == 600
    params = dlg.result_params()
    assert (params["width"], params["height"]) == (800, 600)
    w.close()


# ------------------------------------------------- sticker folder import
def test_sticker_folder_import_button_path(qapp, tmp_path, no_dialogs,
                                           monkeypatch):
    import app.ui.main_window as mw_mod
    w, paths = _mk_window(qapp, tmp_path, monkeypatch, n=1)
    folder = tmp_path / "stickers_src"
    folder.mkdir()
    Image.new("RGBA", (40, 40), (255, 0, 0, 255)).save(folder / "a.png")
    Image.new("RGBA", (40, 40), (0, 255, 0, 255)).save(folder / "b.png")
    added = w.import_sticker_files([str(folder)])
    assert sorted(added) == ["a.png", "b.png"]
    assert "a" in w.sticker_library and "b" in w.sticker_library
    w.close()
