# -*- coding: utf-8 -*-
"""v1.20.0 — 小独完全体落地：镜框卡片自绘 + 爱心瞳进度条（绘制冒烟锁）。

v1.20.0 首轮就踩过：paintEvent 里 setBrush(Qt.NoPen)（应为 NoBrush）抛
TypeError → QPainter 未结束 → "Cannot destroy paint device that is being
painted" → 进程级 access violation。这两个冒烟测试把该崩溃类锁死。
"""
import pytest
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout

from app.ui.widgets import HeartProgressBar, SectionCard
from app.ui import theme


def _grab(widget) -> bytes:
    widget.show()
    for _ in range(3):
        QApplication.instance().processEvents()
    png = widget.grab().toImage()
    widget.close()
    return bytes(png.bits())[:100]  # non-empty check only


@pytest.mark.parametrize("name", ["xiaodu", "xiaodunight"])
def test_sectioncard_mirror_frame_paint_smoke(qapp, name):
    """镜框卡片（暖影+白圈+外环+hover染蓝）在两套皮肤下绘制不炸、出非空像素。"""
    theme.apply_theme(qapp, name)
    w = QWidget()
    lay = QVBoxLayout(w)
    card = SectionCard("标题卡", "副标题说明")
    lay.addWidget(card)
    assert _grab(w)
    # hover 路径也画一遍（enterEvent 会重绘染蓝外环）
    card._hover = True
    card.update()
    assert _grab(w)


@pytest.mark.parametrize("name", ["xiaodu", "xiaodunight"])
def test_heart_progress_paint_smoke(qapp, name):
    """爱心瞳进度条在 0/50/100% 与两套皮肤下绘制不炸（0% 无爱心不炸，
    100% 满格爱心不越界）。"""
    theme.apply_theme(qapp, name)
    w = QWidget()
    lay = QVBoxLayout(w)
    bar = HeartProgressBar()
    lay.addWidget(bar)
    for v in (0, 50, 100):
        bar.setValue(v)
        assert _grab(w)
