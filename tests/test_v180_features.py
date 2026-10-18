# -*- coding: utf-8 -*-
"""v1.8.0: rename inline edit fix, display-name purge, display-name export,
fidelity byte-copy for untouched images, auto-detect tool button."""
import hashlib
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.core.config import default_config


def _noise(size=(48, 48), seed=4):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(size[1], size[0], 3),
                                        dtype=np.uint8))


# ------------------------------------------------------------------ ① inline rename fix

def test_rename_item_starts_editor(qapp, tmp_path):
    from app.ui.thumbnail_list import ThumbnailList

    p = tmp_path / "a.png"
    _noise().save(p)
    tl = ThumbnailList()
    tl.resize(300, 400)
    tl.show()
    tl.add_paths([str(p)])

    tl.rename_item(str(p))  # must exist and not raise (was AttributeError)
    # editItem focuses the item's editor; verify the state machine is armed
    assert tl.current_path() == str(p)

    # commit like the editor does
    item = tl.item(0)
    tl._suspend_item_changed = False
    item.setText("新名字")
    tl._on_item_changed(item)
    assert tl.display_name(str(p)) == "新名字"


def test_rename_menu_action_targets_existing_method(qapp, tmp_path):
    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    assert hasattr(w.thumbnail_list, "rename_item")


# ------------------------------------------------------------------ ② display-name purge

def test_remove_purges_display_name(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod
    from app.ui.main_window import MainWindow

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "cfg" / "config.json")
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    for n in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, n, staticmethod(lambda *a, **k: None))

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), config_mod.default_config())
    p = tmp_path / "a.png"
    _noise().save(p)
    w._add_paths([str(p)])
    w.thumbnail_list.set_display_name(str(p), "AA")
    w._save_display_names()
    assert str(p) in w.config["display_names"]

    w._remove_selected_images([str(p)])
    assert str(p) not in w.config["display_names"]


# ------------------------------------------------------------------ ③ display-name export

def test_export_display_name_mode(tmp_path):
    from app.core.export import export_batch

    a = tmp_path / "sub_a"
    b = tmp_path / "sub_b"
    a.mkdir()
    b.mkdir()
    p1 = a / "037.png"
    p2 = b / "037.png"
    _noise(seed=1).save(p1)
    _noise(seed=2).save(p2)
    paths = [str(p1), str(p2)]

    cfg = {
        "out_dir": str(tmp_path / "out"), "name_mode": "display",
        "prefix": "X", "start": 1, "digits": 3, "fmt": "PNG",
        "quality": 100, "scale_mode": "none", "percent": 100,
        "width": 1000, "height": 1000, "strip_metadata": False,
        "on_name_conflict": "dedupe",
    }
    display_names = {str(p1): "第一张", str(p2): "037"}
    res = export_batch(paths, {}, default_config(), {
        "conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
        "auto_strength": 10, "auto_mask_dilate": 0, "input_size": 1024,
    }, cfg, {}, display_names=display_names)
    assert res["processed"] == 2
    names = {f.name for f in (tmp_path / "out").glob("*.png")}
    assert names == {"第一张.png", "037.png"}  # display name used, dedupe no-op here


# ------------------------------------------------------------------ ④ fidelity byte-copy

def test_untouched_png_exported_byte_identical(tmp_path):
    """No edits -> PNG export is a byte-exact copy: size, pixels, metadata
    and encoding are 100% preserved (no re-encode)."""
    from app.core.export import export_batch

    p = tmp_path / "photo.png"
    img = _noise(seed=7)
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=PngImagePlugin.PngInfo())  # with metadata
    p.write_bytes(buf.getvalue())

    # a state with NO edits at all
    states = {str(p): {
        "pil": img, "selections": [], "stickers": [],
        "auto_mask": np.zeros((48, 48), dtype=bool),
        "erase_mask": np.zeros((48, 48), dtype=bool),
        "orig_size": (48, 48),
    }}
    cfg = {
        "out_dir": str(tmp_path / "out"), "name_mode": "preserve",
        "prefix": "X", "start": 1, "digits": 3, "fmt": "PNG",
        "quality": 100, "scale_mode": "none", "percent": 100,
        "width": 1000, "height": 1000, "strip_metadata": False,
    }
    res = export_batch([str(p)], states, default_config(), {
        "conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
        "auto_strength": 10, "auto_mask_dilate": 0, "input_size": 1024,
    }, cfg, {})
    assert res["processed"] == 1
    out = tmp_path / "out" / "photo.png"
    assert out.read_bytes() == p.read_bytes()  # byte-identical


def test_edited_image_still_composed(tmp_path):
    """With an edit the image is composed as before (no byte copy)."""
    from app.core.export import export_batch
    from app.core.selection import RectSelection

    p = tmp_path / "photo.png"
    _noise(seed=8).save(p)
    states = {str(p): {
        "pil": Image.open(p).convert("RGB"), "selections": [
            RectSelection(5, 5, 20, 20, effect="blur", strength=6)],
        "stickers": [],
        "auto_mask": np.zeros((48, 48), dtype=bool),
        "erase_mask": np.zeros((48, 48), dtype=bool),
        "orig_size": (48, 48),
    }}
    cfg = {
        "out_dir": str(tmp_path / "out"), "name_mode": "preserve",
        "prefix": "X", "start": 1, "digits": 3, "fmt": "PNG",
        "quality": 100, "scale_mode": "none", "percent": 100,
        "width": 1000, "height": 1000, "strip_metadata": False,
    }
    export_batch([str(p)], states, default_config(), {
        "conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
        "auto_strength": 10, "auto_mask_dilate": 0, "input_size": 1024,
    }, cfg, {})
    out = tmp_path / "out" / "photo.png"
    assert out.read_bytes() != p.read_bytes()  # re-encoded (blur applied)


# ------------------------------------------------------------------ ⑤⑥ tool button + toolbar

def test_auto_detect_has_one_entry_only(qapp, tmp_path):
    """v1.22.0：编辑页工具格里那个「自动打码」按钮已删（同一功能两个入口），
    单张检测只剩自动打码页的主按钮一个入口。"""
    from app.ui.tool_panel import ToolPanel
    from conftest import AutoStub
    from types import SimpleNamespace

    cfg = default_config()
    calls = []

    class MainStub(AutoStub):
        def auto_detect_current(self):
            calls.append(1)

    canvas = AutoStub(current_strength=10, current_effect="blur", auto_mask=None,
                      state_changed=SimpleNamespace(emit=lambda *a, **k: None))
    panel = ToolPanel(MainStub(config=cfg, sticker_library={}, canvas=canvas))
    assert "auto_detect" not in panel.tool_buttons
    panel.btn_detect.click()
    assert calls == [1]  # 唯一入口：自动打码页的主按钮


def test_batch_rename_dialog_opens_and_applies(qapp, tmp_path, monkeypatch):
    """Regression: multi-select 批量重命名 used to raise NameError (missing
    QFormLayout import), swallowed by the Qt slot -> 'no reaction'."""
    from PySide6.QtWidgets import QDialog

    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    tmp = Path(tempfile.mkdtemp("uit_multi_rn"))
    paths = []
    for n in ("a.png", "b.png", "c.png"):
        p = tmp / n
        _noise().save(p)
        paths.append(str(p))
    w._add_paths(paths)

    # auto-accept the dialog, then apply as if the user clicked 应用
    accepted = {"n": 0}

    def fake_exec(self):
        accepted["n"] += 1
        return 1

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    w._batch_rename_dialog(paths)  # must not raise

    assert accepted["n"] == 1
    assert w.thumbnail_list.display_name(paths[0]) == "IMG001"
    assert w.thumbnail_list.display_name(paths[2]) == "IMG003"


def test_tutorial_html_kept_but_no_entry(qapp, tmp_path):
    """v1.8.1 state: tutorial.html exists (pure text, no images) but there
    is NO help button / tutorial entry in the app."""
    from app.core.paths import resource_path
    from app.ui.main_window import MainWindow

    html = resource_path("app/assets/简单教程.html")
    assert html.exists()
    text = html.read_text(encoding="utf-8")
    assert "data:image" not in text          # pure text, no embedded images
    assert "快速上手" in text and "保存 / 导出" in text
    assert "同文件夹里相同名字的图片会被直接覆盖" in text
    # no PNG long image anymore
    assert not resource_path("app/assets/tutorial.png").exists()

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    assert not any("帮助" in b.text() for b in w.top_buttons.values())
    assert not hasattr(w, "open_tutorial")


def test_batch_rename_untouched_jpg_byte_identical(qapp, tmp_path, monkeypatch):
    """Regression (user report): 批量重命名 on an unedited JPG used to
    re-encode (72473 -> 33571 bytes, quality loss). Now it byte-copies."""
    import io

    from PySide6.QtWidgets import QMessageBox

    from app.ui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    for n in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, n, staticmethod(lambda *a, **k: None))

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(11)
    arr = np.clip(np.stack([rng.integers(0, 255, (200, 300)),
                            rng.integers(0, 255, (200, 300)),
                            rng.integers(0, 255, (200, 300))], axis=-1),
                  0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=95)
    src.joinpath("photo.jpg").write_bytes(buf.getvalue())

    w = MainWindow(StubDet(), default_config())
    w._add_paths([str(src / "photo.jpg")])
    w.tool_panel.rename_out.setText(str(tmp_path / "out"))
    w.tool_panel.rename_prefix.setText("R")
    w.tool_panel.rename_start.setValue(1)
    w.tool_panel.rename_digits.setValue(3)
    w.batch_rename()
    # wait for the background job worker to finish
    import time
    deadline = time.monotonic() + 10
    while w.job_worker and w.job_worker.isRunning():
        QApplication.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("batch rename worker did not finish")
        time.sleep(0.01)

    out_file = tmp_path / "out" / "R001.jpg"
    assert out_file.exists()
    assert out_file.read_bytes() == (src / "photo.jpg").read_bytes()  # byte-exact


def test_toolbar_has_batch_but_not_single_auto(qapp, tmp_path):
    from app.ui.main_window import MainWindow

    class StubDet:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []

    w = MainWindow(StubDet(), default_config())
    texts = [btn.text() for btn in w.top_buttons.values()]
    assert "🤖 批量自动打码" in texts
    assert not any("自动打码" == t or t == "🤖 自动打码" for t in texts)