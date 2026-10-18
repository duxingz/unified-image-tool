# -*- coding: utf-8 -*-
"""v1.5.1 default values, mask dilation, natural ordering and overlay tests."""
import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.image_ops import compose_image
from app.core.numbering import natural_key
from app.core.selection import MaskSelection


def _gradient_img(size=(64, 64)):
    grad = np.tile(np.linspace(0, 200, size[0], dtype=np.uint8), (size[1], 1))
    arr = np.stack([grad, grad // 2, grad // 4], axis=-1)
    return Image.fromarray(arr)


# ------------------------------------------------------------------ A: defaults

def test_default_config_values():
    cfg = default_config()
    assert cfg["conf_threshold"] == pytest.approx(0.3)
    assert cfg["iou_threshold"] == pytest.approx(0.7)
    assert cfg["auto_strength"] == 15
    assert cfg["edit_defaults"]["strength"] == 15
    assert cfg["auto_mask_dilate"] == 5
    assert cfg["show_mask_overlay"] is True
    assert cfg["theme"] == "xiaodu"


def test_tool_panel_defaults_follow_config(qapp):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QLabel

    from app.ui.tool_panel import ToolPanel
    from conftest import AutoStub

    cfg = default_config()
    canvas = AutoStub(
        current_strength=7, current_effect="blur", auto_mask=None,
        state_changed=SimpleNamespace(emit=lambda *a, **k: None),
    )
    fake = AutoStub(config=cfg, sticker_library={}, canvas=canvas)
    panel = ToolPanel(fake)
    assert panel.conf_spin.value() == pytest.approx(0.3)
    assert panel.iou_spin.value() == pytest.approx(0.7)
    assert panel.auto_strength_spin.value() == 15
    assert panel.strength_spin.value() == 15
    assert panel.mask_dilate_spin.value() == 5
    assert panel.auto_settings()["auto_mask_dilate"] == 5
    assert panel.auto_settings()["auto_strength"] == 15

    # _apply_auto_defaults reads conf/iou from config (used to be hardcoded)
    cfg["conf_threshold"] = 0.5
    cfg["auto_mask_dilate"] = 3
    panel._apply_auto_defaults()
    assert panel.conf_spin.value() == pytest.approx(0.5)
    assert panel.mask_dilate_spin.value() == 3

    # usage hints present (conf / IoU / mask dilation) -- scoped to the
    # auto-censor tab: other tabs legitimately carry hint-styled labels too.
    # v1.22.0 起页内说明压到一行（≤26 字），完整解释挪进 SettingRow 的 tooltip：
    # 因此这里既断页内那一行，也断 tooltip 里还留着完整口径。
    auto_tab = panel.widget(1)  # tab 1 = 自动打码
    hints = [lbl for lbl in auto_tab.findChildren(QLabel)
             if lbl.objectName() == "hint"]
    assert len(hints) >= 3
    joined = " ".join(h.text() for h in hints)
    assert "调低更灵敏" in joined      # confidence note（单行）
    assert "NMS" in joined            # IoU note
    assert "向外扩充" in joined        # dilation note
    assert "可信度" in panel.conf_spin.parentWidget().toolTip()   # 完整解释仍在
    assert "输入尺寸" in panel.input_size_combo.parentWidget().toolTip()


# ------------------------------------------------------------------ B: dilation

def test_mask_dilate_expands_censor_region():
    img = _gradient_img()
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    auto = dict(auto_effect="blur", auto_strength=8)

    out0 = compose_image(img, [], [MaskSelection(mask)], [], {},
                         auto_params={**auto, "auto_mask_dilate": 0})
    out10 = compose_image(img, [], [MaskSelection(mask)], [], {},
                          auto_params={**auto, "auto_mask_dilate": 10})

    arr0, arr10 = np.array(out0), np.array(out10)
    assert not np.array_equal(arr0, arr10)
    # 6px outside the raw mask: unaffected at 0, censored at 10
    probe = (slice(21, 25), slice(21, 25))
    assert np.array_equal(arr0[probe], np.array(img)[probe])
    assert not np.array_equal(arr10[probe], np.array(img)[probe])


def test_mask_dilate_zero_is_identity():
    img = _gradient_img()
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    out = compose_image(img, [], [MaskSelection(mask)], [], {},
                        auto_params=dict(auto_effect="blur", auto_strength=8,
                                         auto_mask_dilate=0))
    arr = np.array(out)
    assert not np.array_equal(arr, np.array(img))  # effect still applied


# ------------------------------------------------------------------ E: ordering

def test_natural_key_orders_numbers():
    names = ["img_10.png", "img_2.png", "img_1.png"]
    assert sorted(names, key=natural_key) == [
        "img_1.png", "img_2.png", "img_10.png"
    ]


def test_export_rename_follows_input_order(tmp_path):
    from app.core.export import export_batch

    p_a = tmp_path / "b.png"
    p_b = tmp_path / "a.png"
    Image.new("RGB", (32, 32), (250, 0, 0)).save(p_a)
    Image.new("RGB", (32, 32), (0, 0, 250)).save(p_b)
    # import order: a.png first (user's view), despite filename order
    order = [str(p_b), str(p_a)]
    cfg = {
        "out_dir": str(tmp_path / "out"), "name_mode": "rename",
        "prefix": "OUT", "start": 1, "digits": 3, "fmt": "PNG",
        "quality": 100, "scale_mode": "none", "percent": 100,
        "width": 1000, "height": 1000, "strip_metadata": False,
    }
    result = export_batch(order, {}, default_config(), {
        "conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
        "auto_strength": 7, "auto_mask_dilate": 0, "input_size": 1024,
    }, cfg, {})
    assert result["processed"] == 2
    img1 = np.array(Image.open(tmp_path / "out" / "OUT001.png"))
    assert img1[0, 0].tolist() == [0, 0, 250]  # a.png (blue) came first


# ------------------------------------------------------------------ F: overlay

def test_canvas_overlay_default_on(qapp):
    from app.ui.editor_canvas import EditorCanvas

    assert EditorCanvas().show_mask_overlay is False  # config applies it, canvas default stays explicit


def test_settings_dialog_has_overlay_controls(qapp, tmp_path, monkeypatch):
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")

    from app.ui.settings_dialog import SettingsDialog

    dlg = SettingsDialog(default_config())
    assert dlg.mask_overlay_check.isChecked() is True
    assert dlg.auto_dilate_spin.value() == 5
    assert dlg.auto_strength.value() == 15