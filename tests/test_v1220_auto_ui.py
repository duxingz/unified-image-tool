# -*- coding: utf-8 -*-
"""v1.22.0：自动打码页重排（动作区上移 / 页内结果条 / 两列类别 / 高级折叠）。

用户 2026-10-18 逐条裁决的 1–7 项，全部落在这组用例里：
1 动作区上移到页首、「清除自动掩码」降级为 ghost；
2 结果条四态（idle/none/ok/stale）+ 每类命中数；参数改动即作废；
3 无图时主按钮禁用、提示走页内（不再弹窗）；
4 顺序 类别 → 效果 → 参数（参数默认折叠）；
5 类别两列 + 全选/默认；
6 控件统一 132×30；
7 组间距 20。
"""
import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace

from app.core.config import CLASS_KEYS, default_config
from app.ui.main_window import MainWindow
from app.ui.tool_panel import _DEFAULT_CLASSES, ToolPanel
from app.ui.widgets import SectionCard


class StubDetector:
    """No detections (the shipped stubs' behaviour)."""
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, *a, **k):
        return []


class DetStub(StubDetector):
    """Returns one detection per requested class, masks inside (10..40, 10..40)."""

    def __init__(self, classes=()):
        self.classes = list(classes)

    def detect(self, image_bgr, *a, **k):
        h, w = image_bgr.shape[:2]
        out = []
        for i, key in enumerate(self.classes):
            m = np.zeros((h, w), dtype=bool)
            m[10:40, 10 + i * 20:30 + i * 20] = True
            out.append(SimpleNamespace(mask=m, class_name=key, bbox=(10, 10, 40, 40),
                                       class_id=0, confidence=0.9))
        return out


def _mk_window(monkeypatch, tmp_path, detector=None, images=1):
    import app.ui.main_window as mw_mod

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(detector or StubDetector(), default_config())
    paths = []
    for i in range(images):
        arr = np.full((120, 160, 3), 50 + i * 40, dtype=np.uint8)
        p = tmp_path / f"img{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    if paths:
        w.thumbnail_list.add_paths(paths)
        w._load_path(paths[0], save_current=False)
    return w, paths


# --------------------------------------------------------------- 出厂默认值同源
def test_default_classes_match_config():
    """面板的「默认」勾选与 core.config 的出厂值必须同源，别各写一份。"""
    assert tuple(default_config()["selected_regions"]) == _DEFAULT_CLASSES


# --------------------------------------------------------------- 1) 动作区层级
def test_primary_action_is_first_and_clear_is_ghost(qapp, monkeypatch, tmp_path):
    w, _ = _mk_window(monkeypatch, tmp_path)
    w.show()
    qapp.processEvents()
    tp = w.tool_panel
    tp.setCurrentIndex(1)
    qapp.processEvents()
    card_cls = [c for c in tp.widget(1).findChildren(SectionCard)
                if not c.isHidden()][0]
    # 主按钮排在所有卡片之前（旧版压在 3 张卡之后、页面 76% 处）
    assert tp.btn_detect.y() < card_cls.y()
    assert tp.btn_detect.property("variant") == "primary"
    assert tp.btn_clear_auto.property("variant") == "ghost"
    # 破坏性动作不再和主按钮同宽同高相贴
    assert tp.btn_clear_auto.width() < tp.btn_detect.width()
    assert tp.btn_detect.height() > tp.btn_clear_auto.height()
    w.close()


def test_tab_order_is_class_effect_params(qapp, monkeypatch, tmp_path):
    """4) 类别 → 效果 → 参数（参数卡默认收起）。"""
    from PySide6.QtWidgets import QLabel

    w, _ = _mk_window(monkeypatch, tmp_path)
    tp = w.tool_panel
    page = tp.widget(1)
    cards = [c for c in page.findChildren(SectionCard) if not c.isHidden()]
    names = [c.findChild(QLabel, "cardTitle").text() for c in cards]
    assert names == ["检测类别", "打码效果"]
    assert tp.plate_card.isHidden() is True       # 高级参数默认收起
    assert "展开" in tp.btn_advanced.text()
    tp.btn_advanced.setChecked(True)
    assert tp.plate_card.isHidden() is False
    assert "收起" in tp.btn_advanced.text()
    w.close()


# --------------------------------------------------------------- 2) 结果条
def test_detect_fills_status_and_per_class_counts(qapp, monkeypatch, tmp_path):
    w, _ = _mk_window(monkeypatch, tmp_path,
                      detector=DetStub(("vagina", "vagina", "penis")))
    tp = w.tool_panel
    assert tp.auto_status.tone() == "idle"
    w.auto_detect_current()
    assert tp.auto_status.tone() == "ok"
    txt = tp.auto_status.text()
    assert "3 处" in txt and "阴部×2" in txt and "阴茎×1" in txt
    # 每类命中数直接贴在勾选框上
    assert "×2" in tp.class_checks["vagina"].text()
    w.close()


def test_empty_detect_keeps_existing_mask(qapp, monkeypatch, tmp_path):
    """检测不到 ≠ 用户想擦掉已有的码：旧掩码必须原样留着。"""
    w, _ = _mk_window(monkeypatch, tmp_path,
                      detector=DetStub(("vagina",)))
    w.auto_detect_current()
    assert w.canvas.auto_mask.any()
    w.detector = StubDetector()          # 下一次啥也检测不到
    w.auto_detect_current()
    assert w.canvas.auto_mask.any()      # 没被清空
    assert w.tool_panel.auto_status.tone() == "none"
    assert "旧的自动掩码仍在" in w.tool_panel.auto_status.text()
    w.close()


def test_param_change_invalidates_result(qapp, monkeypatch, tmp_path):
    """2) 改置信度/类别之后旧结果作废，必须重新检测。"""
    w, _ = _mk_window(monkeypatch, tmp_path,
                      detector=DetStub(("vagina",)))
    tp = w.tool_panel
    tp.btn_advanced.setChecked(True)
    w.auto_detect_current()
    assert tp.auto_status.tone() == "ok"
    tp.conf_spin.setValue(0.55)
    assert tp.auto_status.tone() == "stale"
    assert "需重新检测" in tp.auto_status.text()
    # 命中数同时清掉，别拿旧数字骗人
    assert "×" not in tp.class_checks["vagina"].text()
    w.close()


def test_effect_change_does_not_invalidate(qapp, monkeypatch, tmp_path):
    """效果/强度是实时重绘，不需要重新检测。"""
    w, _ = _mk_window(monkeypatch, tmp_path,
                      detector=DetStub(("vagina",)))
    tp = w.tool_panel
    w.auto_detect_current()
    tp.auto_strength_spin.setValue(42)
    assert tp.auto_status.tone() == "ok"
    w.close()


# --------------------------------------------------------------- 3) 无图态
def test_no_image_is_inline_and_disables_primary(qapp, monkeypatch, tmp_path):
    w, _ = _mk_window(monkeypatch, tmp_path, images=0)
    tp = w.tool_panel
    assert tp.btn_detect.isEnabled() is False
    assert tp.btn_clear_auto.isEnabled() is False
    assert tp.auto_status.tone() == "noimg"
    # 点一下也不会弹窗（弹窗在测试环境会被 conftest 拦掉，这里断的是状态）
    w.auto_detect_current()
    assert tp.auto_status.tone() == "noimg"
    assert "先导入" in tp.auto_status.text()
    w.close()


# --------------------------------------------------------------- 5) 类别
def test_class_checkboxes_are_two_columns(qapp, monkeypatch, tmp_path):
    w, _ = _mk_window(monkeypatch, tmp_path)
    w.show()
    w.tool_panel.setCurrentIndex(1)
    qapp.processEvents()
    tp = w.tool_panel
    xs = {round(tp.class_checks[k].x()) for k in CLASS_KEYS}
    assert len(xs) == 2, f"应为两列，实测列 x={sorted(xs)}"
    # 短标签（完整双语名进 tooltip）
    assert tp.class_checks["nipple"].text() == "乳头"
    assert "nipple" in tp.class_checks["nipple"].toolTip()
    w.close()


def test_check_all_and_default_links(qapp, monkeypatch, tmp_path):
    w, _ = _mk_window(monkeypatch, tmp_path)
    tp = w.tool_panel
    tp._set_class_checks(set(CLASS_KEYS))
    assert all(cb.isChecked() for cb in tp.class_checks.values())
    tp._set_class_checks(set(_DEFAULT_CLASSES))
    assert [k for k, cb in tp.class_checks.items() if cb.isChecked()] == list(_DEFAULT_CLASSES)
    w.close()


# --------------------------------------------------------------- 7) 清除按钮
def test_clear_button_tracks_mask(qapp, monkeypatch, tmp_path):
    w, _ = _mk_window(monkeypatch, tmp_path,
                      detector=DetStub(("vagina",)))
    tp = w.tool_panel
    assert tp.btn_clear_auto.isEnabled() is False     # 图已载入但还没有掩码
    w.auto_detect_current()
    assert tp.btn_clear_auto.isEnabled() is True
    w.canvas.clear_auto_mask()
    assert tp.btn_clear_auto.isEnabled() is False
    w.close()


def test_switching_image_resets_result(qapp, monkeypatch, tmp_path):
    """换图不能把上一张的命中数留在这一张上。"""
    w, paths = _mk_window(monkeypatch, tmp_path,
                          detector=DetStub(("vagina",)), images=2)
    tp = w.tool_panel
    w.auto_detect_current()
    assert tp.auto_status.tone() == "ok"
    w._load_path(paths[1], save_current=True)
    assert tp.auto_status.tone() == "idle"
    assert "×" not in tp.class_checks["vagina"].text()
    w.close()


# --------------------------------------------------------------- 6) 控件度量
def test_auto_controls_share_one_size(qapp, monkeypatch, tmp_path):
    """6) 数字框与下拉框统一 132×30（旧版 93/90/82×31 与 217/230×28）。

    控件总高 = 内容 20 + 主题内边距 8 + 边框 2，所以必须先把主题挂到
    QApplication 上，否则量到的是裸 Qt 的 22/26。
    """
    from app.ui.theme import apply_theme
    from app.ui.tool_panel import _CTRL_W

    w, _ = _mk_window(monkeypatch, tmp_path)
    apply_theme(qapp, "xiaodu")
    try:
        w.show()
        qapp.processEvents()
        tp = w.tool_panel
        tp.setCurrentIndex(1)
        tp.btn_advanced.setChecked(True)
        qapp.processEvents()
        controls = [tp.conf_spin, tp.iou_spin, tp.input_size_combo,
                    tp.auto_effect_combo, tp.auto_strength_spin, tp.mask_dilate_spin]
        sizes = {(c.width(), c.height()) for c in controls}
        assert sizes == {(_CTRL_W, 30)}, f"控件尺寸应统一 132×30，实测 {sizes}"
        # 主按钮 36 高（36 = 内容 26 + 内边距 8 + 边框 2）
        assert tp.btn_detect.height() == 36
    finally:
        qapp.setStyleSheet("")      # 别把主题漏给后续用例
        w.close()


# --------------------------------------------------------------- 10) 去重复入口
def test_edit_tab_has_no_duplicate_detect_button(qapp, monkeypatch, tmp_path):
    """「自动打码」只留在自动打码页，编辑页工具格不再重复一个。"""
    w, _ = _mk_window(monkeypatch, tmp_path)
    assert "auto_detect" not in w.tool_panel.tool_buttons
    assert set(w.tool_panel.tool_buttons) == {"pointer", "brush", "eraser",
                                              "rect", "lasso"}
    w.close()
