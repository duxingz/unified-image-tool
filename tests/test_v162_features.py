# -*- coding: utf-8 -*-
"""v1.6.2: overlay/compose consistency, eraser on rect/lasso, splitter drag
range, config migration v2, compact long-page panel, unsaved-close warning."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.image_ops import compose_image
from app.core.selection import LassoSelection, MaskSelection, RectSelection


def _noise(size=(96, 96), seed=17):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(size[1], size[0], 3),
                                        dtype=np.uint8))


# ------------------------------------------------------------------ ① overlay consistency

def test_overlay_mask_matches_compose(qapp):
    from app.core.effects import expand_mask
    from app.ui.editor_canvas import EditorCanvas

    c = EditorCanvas()
    c.set_image(np.array(_noise()))
    h, w = c.image_np.shape[:2]
    c.auto_mask = np.zeros((h, w), dtype=bool)
    c.auto_mask[20:60, 20:60] = True
    c.erase_mask = np.zeros((h, w), dtype=bool)
    c.erase_mask[38:46, 38:46] = True
    c.auto_params["auto_mask_dilate"] = 8
    c._rebuild()

    assert c._overlay_mask is not None
    expected = expand_mask(c.auto_mask, 8) & ~c.erase_mask
    assert np.array_equal(c._overlay_mask, expected)

    # erased band must not appear in the overlay
    assert not c._overlay_mask[42:44, 42:44].any()
    # dilated ring area (along the axis, outside the raw mask) IS covered
    assert c._overlay_mask[12:14, 20:24].any()


def test_eraser_fully_removes_rect_and_lasso():
    img = _noise(seed=23)
    erase = np.zeros((96, 96), dtype=bool)
    erase[10:60, 10:60] = True  # covers the whole selection below

    rect = RectSelection(20, 20, 45, 45, effect="blur", strength=8)
    lasso = LassoSelection([(20, 20), (45, 20), (45, 45), (20, 45)],
                           effect="mosaic", strength=8)
    for sel in (rect, lasso):
        erased = compose_image(img, [sel], [], [], {}, params={},
                               erase_mask=erase)
        assert np.array_equal(np.array(erased), np.array(img))  # fully gone

    # half-erase: the un-erased remainder still shows the effect
    erase_half = np.zeros((96, 96), dtype=bool)
    erase_half[10:35, 10:60] = True
    half = compose_image(img, [rect], [], [], {}, params={},
                         erase_mask=erase_half)
    arr = np.array(half)
    base = np.array(img)
    assert np.array_equal(arr[10:32, 10:32], base[10:32, 10:32])  # erased
    assert not np.array_equal(arr[38:43, 30:40], base[38:43, 30:40])  # censored


# ------------------------------------------------------------------ ② splitter drag range

def test_splitter_sizes_respect_requests(qapp, tmp_path):
    from PySide6.QtWidgets import QMessageBox
    for n in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.Yes))

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    p = tmp_path / "a.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(p)
    w._add_paths([str(p)])

    sp = w.findChildren(type(w), "QSplitter")
    splitter = None
    from PySide6.QtWidgets import QSplitter
    for c in w.findChildren(QSplitter):
        splitter = c
        break
    assert splitter is not None
    splitter.setSizes([220, 760, 340])
    sizes = splitter.sizes()
    assert sizes[0] <= 260, f"left panel stuck at {sizes[0]} (was 338)"


# ------------------------------------------------------------------ ③ migration v2

def test_migration_v2_flips_overlay_only(tmp_path, monkeypatch):
    import json

    from app.core import config as config_mod
    from app.core.config import UI_VERSION

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")
    # a 1.5.1-migrated config with the user's own tuned values
    legacy = {
        "ui_version": "1.5.1",
        "conf_threshold": 0.55,       # tuned by the user
        "auto_strength": 4,
        "show_mask_overlay": False,
    }
    config_mod.CONFIG_PATH.write_text(json.dumps(legacy), encoding="utf-8")

    cfg = config_mod.load_config()
    assert cfg["ui_version"] == UI_VERSION
    assert cfg["show_mask_overlay"] is True  # flipped by v2 migration
    assert cfg["conf_threshold"] == 0.55     # user values untouched
    assert cfg["auto_strength"] == 4

    # a config already at UI_VERSION is not touched again
    current = dict(legacy, ui_version=UI_VERSION, show_mask_overlay=False)
    config_mod.CONFIG_PATH.write_text(json.dumps(current), encoding="utf-8")
    cfg2 = config_mod.load_config()
    assert cfg2["show_mask_overlay"] is False  # user's choice preserved


# ------------------------------------------------------------------ ④ compact long page

def test_tool_panel_is_tabbed(qapp):
    """The right panel is the tabbed layout (edit/auto/stickers/watermark/
    batch)."""
    from PySide6.QtWidgets import QTabWidget

    from app.ui.tool_panel import ToolPanel
    from conftest import AutoStub

    cfg = default_config()
    canvas = AutoStub(current_strength=7, current_effect="blur", auto_mask=None,
                      state_changed=SimpleNamespace(emit=lambda *a, **k: None))
    panel = ToolPanel(AutoStub(config=cfg, sticker_library={}, canvas=canvas))
    assert isinstance(panel, QTabWidget)
    assert panel.count() == 5
    titles = [panel.tabText(i) for i in range(panel.count())]
    assert titles == ["编辑", "自动打码", "贴纸", "水印", "批量"]
    for attr in ("effect_combo", "auto_effect_combo", "convert_btn",
                 "resize_btn", "rename_btn", "sticker_list"):
        assert hasattr(panel, attr), attr


# ------------------------------------------------------------------ ⑤ unsaved close warning

def test_close_warns_on_unsaved_edits(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from app.ui.main_window import MainWindow

    calls = {"n": 0}

    def fake_question(*a, **k):
        calls["n"] += 1
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    p = tmp_path / "a.png"
    Image.new("RGB", (32, 32), (9, 9, 9)).save(p)
    w._add_paths([str(p)])
    w.canvas.selections = [RectSelection(2, 2, 10, 10, effect="blur", strength=5)]
    w.canvas.state_changed.emit()  # marks dirty

    # "No" keeps the window open
    assert w.close() is False
    assert calls["n"] == 1
    # "Yes" closes
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    assert w.close() is True

    # after export (dirty cleared) or no edits: no warning
    w2 = MainWindow(StubDet(), default_config())
    p2 = tmp_path / "b.png"
    Image.new("RGB", (32, 32), (9, 9, 9)).save(p2)
    w2._add_paths([str(p2)])
    w2._dirty_paths = {str(p2)}
    calls["n"] = 0
    w2._dirty_paths.clear()  # simulated successful export
    assert w2.close() is True
    assert calls["n"] == 0