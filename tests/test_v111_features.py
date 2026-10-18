# -*- coding: utf-8 -*-
"""v1.11.0 feature tests (borrowed from nEO iMAGING analysis):

A1 EXIF orientation: loads are upright everywhere and exports carry no
   orientation tag (no double rotation).
A3 Rename templates: {原名} {日期} {时间} {序号} + illegal-char sanitize,
   legacy prefix behavior preserved.
A4 Rename preview: build_rename_plan rows.
B5 Batch pipeline: run_pipeline_on_image executes content + output actions;
   describe_pipeline_action labels.
A2/B6 config-backed templates & presets persist via save_config.
"""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from app.core.config import default_config
from app.core.numbering import (
    build_rename_plan,
    expand_rename_template,
    sanitize_filename,
)
from app.core.export import (
    describe_pipeline_action,
    export_batch,
    run_pipeline_on_image,
)
from app.core.image_ops import load_original, load_resized, original_size


@pytest.fixture()
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))


def _auto():
    return {"auto_effect": "blur", "auto_strength": 8, "auto_mask_dilate": 0}


# ------------------------------------------------------------- A1 EXIF
def _oriented_source(tmp_path, orientation=6, size=(400, 200)):
    im = Image.new("RGB", size, (100, 150, 200))
    exif = Image.Exif()
    exif[274] = orientation
    p = tmp_path / f"rot{orientation}.jpg"
    im.save(p, exif=exif)
    return str(p)


def test_exif_orientation_upright_everywhere(tmp_path):
    src = _oriented_source(tmp_path)
    assert Image.open(src).size == (400, 200)  # raw pixels are landscape
    r, _, _ = load_resized(src)
    assert r.size == (200, 400)  # upright
    assert load_original(src).size == (200, 400)
    assert original_size(src) == (200, 400)


def test_exif_export_no_double_rotation(tmp_path):
    src = _oriented_source(tmp_path)
    cfg = {"out_dir": str(tmp_path / "out"), "name_mode": "preserve",
           "prefix": "", "start": 1, "digits": 3, "fmt": "PNG",
           "quality": 100, "scale_mode": "none", "percent": 100,
           "width": 1000, "height": 1000, "strip_metadata": False}
    res = export_batch([src], {}, default_config(), _auto(), cfg, {})
    assert res["processed"] == 1
    out = Image.open(tmp_path / "out" / "rot6.png")
    assert out.size == (200, 400)
    # re-encoded export must NOT keep the orientation tag, or viewers would
    # rotate the already-upright pixels a second time
    assert out.getexif().get(274) is None


def test_unedited_jpg_fastpath_keeps_bytes(tmp_path):
    """Without edits the byte-copy fast path keeps the original EXIF intact
    (viewers still show it upright via the tag)."""
    src = _oriented_source(tmp_path)
    cfg = {"out_dir": str(tmp_path / "out"), "name_mode": "preserve",
           "prefix": "", "start": 1, "digits": 3, "fmt": "JPG",
           "quality": 100, "scale_mode": "none", "percent": 100,
           "width": 1000, "height": 1000, "strip_metadata": False}
    export_batch([src], {}, default_config(), _auto(), cfg, {})
    with open(src, "rb") as f1:
        with open(tmp_path / "out" / "rot6.jpg", "rb") as f2:
            assert f1.read() == f2.read()


# ------------------------------------------------------ A3 rename templates
def test_rename_template_variables(tmp_path):
    p = tmp_path / "IMG_1234.png"
    p.write_bytes(b"x")
    # legacy: no variables -> prefix + zero-padded number
    assert expand_rename_template("IMG", str(p), 0, 1, 3) == "IMG001"
    assert expand_rename_template("IMG", str(p), 4, 10, 4) == "IMG0014"
    # variables
    out = expand_rename_template("{原名}_{序号}", str(p), 2, 5, 2)
    assert out == "IMG_1234_07"
    out = expand_rename_template("c_{日期}_{原名}", str(p), 0, 1, 3,
                                 date_info={"date": "20260909",
                                            "time": "20260909-101010"})
    assert out == "c_20260909_IMG_1234"
    out = expand_rename_template("{时间}", str(p), 0, 1, 3,
                                 date_info={"date": "20260909",
                                            "time": "20260909-101010"})
    assert out == "20260909-101010"


def test_rename_template_sanitizes_windows_illegal_chars():
    assert sanitize_filename('a<b>c:d"e/f\\g|h?i*j') == "a_b_c_d_e_f_g_h_i_j"


def test_rename_plan_preview_rows(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"x")
    p2.write_bytes(b"x")
    rows = build_rename_plan([str(p1), str(p2)], "序_{序号}", 1, 3)
    assert rows == [("a.png", "序_001.png"), ("b.png", "序_002.png")]


# ---------------------------------------------------------- B5 pipeline
def _stub_detector():
    class Stub:
        def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
                   allowed_classes=None, input_size=None):
            h, w = image_bgr.shape[:2]
            m = np.zeros((h, w), dtype=bool)
            m[10:40, 10:40] = True
            return [type("D", (), {"mask": m})()]
    return Stub()


def test_describe_pipeline_action():
    assert describe_pipeline_action({"type": "auto_censor"}) == "自动打码"
    assert "logo" in describe_pipeline_action(
        {"type": "watermark",
         "params": {"asset": "logo", "anchor": "top-left"}})
    assert describe_pipeline_action(
        {"type": "resize", "params": {"mode": "percent", "percent": 50}}
    ) == "缩放: 50%"
    assert describe_pipeline_action(
        {"type": "convert", "params": {"fmt": "JPG", "quality": 95}}
    ).startswith("转格式: JPG")


def test_pipeline_resize_convert_rename(tmp_path):
    src = tmp_path / "photo.png"
    Image.fromarray(np.full((600, 400, 3), 80, dtype=np.uint8)).save(src)
    states = {}
    actions = [
        {"type": "resize", "params": {"mode": "percent", "percent": 50}},
        {"type": "convert", "params": {"fmt": "JPG", "quality": 95}},
        {"type": "rename", "params": {"template": "小_{序号}", "start": 1,
                                      "digits": 3}},
    ]
    conv, pil_fmt, kwargs, ext, stem = run_pipeline_on_image(
        str(src), states, actions, {}, _auto(), {}, index=0)
    assert (conv.width, conv.height) == (200, 300)
    assert pil_fmt == "JPEG" and ext == "jpg"
    assert stem == "小_001"
    # output stage did NOT touch the workspace state (no watermark action)
    assert states[str(src)]["pil"].size == (400, 600)


def test_pipeline_auto_censor_and_watermark(tmp_path):
    from PIL import Image as PILImage
    lib = {"wm": PILImage.new("RGBA", (200, 100), (255, 0, 0, 255))}
    src = tmp_path / "pic.png"
    Image.fromarray(np.full((400, 300, 3), 60, dtype=np.uint8)).save(src)
    states = {}
    actions = [
        {"type": "auto_censor",
         "params": {"conf": 0.3, "iou": 0.7, "classes": []}},
        {"type": "watermark",
         "params": {"asset": "wm", "anchor": "bottom-right",
                    "size_frac": 0.2, "opacity": 1.0}},
    ]
    conv, pil_fmt, kwargs, ext, stem = run_pipeline_on_image(
        str(src), states, actions, lib, _auto(), {},
        detector=_stub_detector(), index=0)
    assert stem == "pic" and ext == "png"
    # the censor + watermark mutated the workspace state (visible in UI)
    assert states[str(src)]["auto_mask"].any()
    assert len(states[str(src)]["stickers"]) == 1
    assert states[str(src)]["stickers"][0].watermark
    # and the composed output actually carries the red watermark at BR
    a = np.array(conv.convert("RGB"))
    m = (a[..., 0] > 200) & (a[..., 1] < 80)
    assert m.any()


def test_pipeline_missing_watermark_asset_raises(tmp_path):
    src = tmp_path / "x.png"
    Image.new("RGB", (50, 50), (0, 0, 0)).save(src)
    actions = [{"type": "watermark",
                "params": {"asset": "gone", "anchor": "bottom-right"}}]
    with pytest.raises(ValueError):
        run_pipeline_on_image(str(src), {}, actions, {}, _auto(), {})


def test_pipeline_state_loaded_for_unopened_image(tmp_path):
    """Running the pipeline on an image never opened in the canvas must load
    it (the old _load_path KeyError path)."""
    src = tmp_path / "never.png"
    Image.fromarray(np.full((120, 80, 3), 40, dtype=np.uint8)).save(src)
    states = {}
    conv, _, _, _, _ = run_pipeline_on_image(
        str(src), states, [{"type": "convert", "params": {"fmt": "PNG"}}],
        {}, _auto(), {}, index=0)
    assert (conv.width, conv.height) == (80, 120)
    assert "pil" in states[str(src)] and "orig_size" in states[str(src)]


# ------------------------------------ A2/B6 config-backed templates/presets
def test_text_wm_template_and_preset_config_roundtrip(qapp, tmp_path,
                                                      no_dialogs,
                                                      monkeypatch):
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(_mk_stub(), default_config())
    paths = []
    for i in range(2):
        arr = np.full((240, 320, 3), 60, dtype=np.uint8)
        p = tmp_path / f"i{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    w.thumbnail_list.add_paths(paths)
    w._load_path(paths[0], save_current=False)

    # A2: text watermark template save -> config, selection fills inputs
    w.create_text_watermark("签名", color=(255, 255, 0, 255), font_size=48)
    w.tool_panel.refresh_watermark_assets()  # surface the new asset in UI
    w.config["text_wm_templates"] = [{
        "name": "我的签名", "text": "签名", "color": [255, 255, 0, 255],
        "font_size": 48, "opacity": 1.0,
    }]
    w.tool_panel.refresh_text_wm_templates()
    ix = w.tool_panel.wm_tpl.findData("我的签名")
    w.tool_panel.wm_tpl.setCurrentIndex(ix)
    assert w.tool_panel.wm_text.text() == "签名"
    assert w.tool_panel.wm_font.value() == 48

    # B6: preset save + apply (uses QInputDialog + QMessageBox, both stubbed)
    w.tool_panel.wm_asset.setCurrentIndex(
        w.tool_panel.wm_asset.findData("签名"))
    from PySide6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("我的方案", True)))
    w.save_watermark_preset()
    assert any(p["name"] == "我的方案"
               for p in w.config["watermark_presets"])
    w.apply_watermark_preset()
    for p in paths:
        sts = w.image_states[p].get("stickers", [])
        assert len(sts) == 1 and sts[0].watermark
    w.close()


def _mk_stub():
    from types import SimpleNamespace

    class StubDetector:
        model_path = "stub"
        providers = ["CPUExecutionProvider"]

        def detect(self, *a, **k):
            return []
    return StubDetector()
