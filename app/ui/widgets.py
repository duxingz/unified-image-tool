# -*- coding: utf-8 -*-
"""Custom widgets for Automatic Coding 2."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRect, QRectF, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRegion,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import SPACE, current_palette  # noqa: F401  (shared tokens)


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert RGB ndarray to QImage."""
    h, w, ch = arr.shape
    bytes_per_line = 3 * w
    return QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


class MaskGraphicsView(QGraphicsView):
    """Image view with zoom/pan and brush mask editing."""

    mask_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self._image_item: QGraphicsPixmapItem | None = None
        self._overlay_item: QGraphicsPixmapItem | None = None

        self._rgb: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

        self.tool = "view"  # view | add | erase
        self.brush_size = 20
        self.edited = False

        self._panning = False
        self._last_scene_pos: QPointF | None = None

    def set_image_with_mask(self, rgb: np.ndarray, mask: np.ndarray | None = None):
        self._rgb = rgb.copy()
        if mask is None:
            self._mask = np.zeros(rgb.shape[:2], dtype=bool)
        else:
            self._mask = mask.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.edited = False
        self._refresh()

    def set_mask(self, mask: np.ndarray):
        self._mask = mask.copy()
        self._refresh(fit=False)

    def get_mask(self) -> np.ndarray:
        if self._mask is None:
            return np.zeros((1, 1), dtype=bool)
        return self._mask.copy()

    def has_image(self) -> bool:
        return self._rgb is not None

    def clear(self):
        self._rgb = None
        self._mask = None
        self._scene.clear()
        self._image_item = None
        self._overlay_item = None

    def undo(self):
        if not self._undo_stack:
            return
        if self._mask is not None:
            self._redo_stack.append(self._mask.copy())
        self._mask = self._undo_stack.pop().copy()
        self.edited = True
        self._refresh(fit=False)
        self.mask_changed.emit()

    def redo(self):
        if not self._redo_stack:
            return
        if self._mask is not None:
            self._undo_stack.append(self._mask.copy())
        self._mask = self._redo_stack.pop().copy()
        self.edited = True
        self._refresh(fit=False)
        self.mask_changed.emit()

    def expand_mask(self, pixels: int = 5):
        if self._mask is None:
            return
        from ..core.effects import expand_mask as _expand

        self._push_undo()
        self._mask = _expand(self._mask, pixels)
        self.edited = True
        self._refresh(fit=False)
        self.mask_changed.emit()

    def shrink_mask(self, pixels: int = 5):
        if self._mask is None:
            return
        from ..core.effects import shrink_mask as _shrink

        self._push_undo()
        self._mask = _shrink(self._mask, pixels)
        self.edited = True
        self._refresh(fit=False)
        self.mask_changed.emit()

    def _push_undo(self):
        if self._mask is not None:
            self._undo_stack.append(self._mask.copy())
            self._redo_stack.clear()

    def _refresh(self, fit: bool = True):
        self._scene.clear()
        self._image_item = None
        self._overlay_item = None
        if self._rgb is None:
            return

        qimg = numpy_to_qimage(self._rgb)
        pix = QPixmap.fromImage(qimg)
        self._image_item = self._scene.addPixmap(pix)
        self._image_item.setZValue(0)

        if self._mask is not None:
            overlay = self._build_overlay_pixmap()
            self._overlay_item = self._scene.addPixmap(overlay)
            self._overlay_item.setZValue(1)

        self._scene.setSceneRect(QRectF(pix.rect()))
        if fit:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def _build_overlay_pixmap(self) -> QPixmap:
        h, w = self._mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        # red translucent overlay on masked pixels
        overlay[self._mask] = (255, 80, 80, 120)
        qimg = QImage(
            overlay.data, w, h, 4 * w, QImage.Format_RGBA8888
        ).copy()
        return QPixmap.fromImage(qimg)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self._panning = True
            self._last_scene_pos = self.mapToScene(event.position().toPoint())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.tool in ("add", "erase") and self._rgb is not None:
            self._push_undo()
            self._paint_at(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            scene_pos = self.mapToScene(event.position().toPoint())
            delta = scene_pos - self._last_scene_pos
            self._last_scene_pos = scene_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        if self.tool in ("add", "erase") and self._rgb is not None and event.buttons() & Qt.LeftButton:
            self._paint_at(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self._last_scene_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _paint_at(self, view_pos):
        scene_pos = self.mapToScene(view_pos.toPoint())
        if self._rgb is None or self._mask is None:
            return
        h, w = self._mask.shape
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        r = max(1, self.brush_size // 2)

        changed = False
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                yy = y + dy
                xx = x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    if self.tool == "add" and not self._mask[yy, xx]:
                        self._mask[yy, xx] = True
                        changed = True
                    elif self.tool == "erase" and self._mask[yy, xx]:
                        self._mask[yy, xx] = False
                        changed = True
        if changed:
            self.edited = True
            self._refresh(fit=False)
            self.mask_changed.emit()


class NoWheelComboBox(QComboBox):
    """QComboBox that ignores mouse wheel changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """QSpinBox that ignores mouse wheel changes."""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores mouse wheel changes."""

    def wheelEvent(self, event):
        event.ignore()


class CollapsibleBox(QWidget):
    """A simple collapsible section with a title button."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._toggle_button = QToolButton()
        self._toggle_button.setText(title)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(True)
        self._toggle_button.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; font-size: 14px;"
            " padding: 6px; text-align: left; }"
            "QToolButton::menu-indicator { image: none; }"
        )
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle_button.setArrowType(Qt.DownArrow)
        self._toggle_button.toggled.connect(self._on_toggled)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 8, 8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle_button)
        layout.addWidget(self._content)

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)
        self._toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def set_content_layout(self, layout):
        """Move an existing layout into the collapsible content area."""
        # Remove any existing items
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._content_layout.addLayout(layout)

    def set_collapsed(self, collapsed: bool):
        self._toggle_button.setChecked(not collapsed)

    def is_collapsed(self) -> bool:
        return not self._toggle_button.isChecked()


class RenamePreviewDialog(QDialog):
    """Batch-rename preview: shows original -> new filename rows before any
    file is touched. 确认 = run, 取消 = no side effects."""

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("重命名预览")
        self.resize(560, 420)
        lay = QVBoxLayout(self)
        info = QLabel(f"共 {len(rows)} 项。确认后才执行重命名。")
        lay.addWidget(info)
        table = QTableWidget(len(rows), 2, self)
        table.setHorizontalHeaderLabels(["原文件名", "新文件名"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setColumnWidth(0, 240)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        for r, (old, new) in enumerate(rows):
            table.setItem(r, 0, QTableWidgetItem(old))
            item = QTableWidgetItem(new)
            item.setForeground(QColor("#1E88E5"))
            table.setItem(r, 1, item)
        lay.addWidget(table, 1)
        btns = QHBoxLayout()
        btns.addStretch()
        ok = QPushButton("确认执行")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay.addLayout(btns)


# ----------------------------------------------------------------------
# Reusable widgets (nEO iMAGING-inspired additions, v1.12)
# ----------------------------------------------------------------------
def pil_to_icon(img, size=56):
    """PIL image -> QIcon scaled to fit ``size`` px (aspect preserved),
    for thumbnail-wall list entries."""
    from PIL import Image as PILImage
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    im = img
    if im.mode not in ("RGBA", "RGB"):
        im = im.convert("RGBA")
    thumb = im.copy()
    thumb.thumbnail((size, size))
    qimg = QImage(
        thumb.tobytes("raw", "RGBA") if thumb.mode == "RGBA"
        else thumb.tobytes("raw", "RGB"),
        thumb.width, thumb.height,
        4 * thumb.width if thumb.mode == "RGBA" else 3 * thumb.width,
        QImage.Format_RGBA8888 if thumb.mode == "RGBA" else QImage.Format_RGB888,
    )
    pmw = QPixmap.fromImage(qimg)
    painter = QPainter(pm)
    x = (size - pmw.width()) // 2
    y = (size - pmw.height()) // 2
    painter.drawPixmap(x, y, pmw)
    painter.end()
    return QIcon(pm)


# 3x3 anchor grid -> anchor keys used by Sticker.place()
ANCHOR_GRID = [
    ["top-left", "top-center", "top-right"],
    ["left-center", "center", "right-center"],
    ["bottom-left", "bottom-center", "bottom-right"],
]
ANCHOR_LABELS = {
    "top-left": "左上", "top-center": "上中", "top-right": "右上",
    "left-center": "左中", "center": "居中", "right-center": "右中",
    "bottom-left": "左下", "bottom-center": "下中", "bottom-right": "右下",
}


class AnchorPicker(QWidget):
    """Visual 3x3 grid replacing the anchor dropdown: click a cell to pick
    where the watermark sticks (values match Sticker.place anchors)."""

    anchorChanged = Signal(str)

    def __init__(self, anchor="bottom-right", parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons = {}
        for r, row in enumerate(ANCHOR_GRID):
            for c, key in enumerate(row):
                b = QPushButton(ANCHOR_LABELS[key])
                b.setCheckable(True)
                b.setFixedSize(52, 26)
                b.setToolTip(key)
                self._group.addButton(b)
                b.clicked.connect(lambda _=False, k=key: self.anchorChanged.emit(k))
                self._buttons[key] = b
                lay.addWidget(b, r, c)
        self.set_anchor(anchor)

    def set_anchor(self, anchor: str):
        for key, b in self._buttons.items():
            b.setChecked(key == anchor)

    def anchor(self) -> str:
        for key, b in self._buttons.items():
            if b.isChecked():
                return key
        return "bottom-right"


class PipelinePreviewDialog(QDialog):
    """Pre-run preview for the batch pipeline: renders ONE image through the
    full action chain (lazily, on demand) and lets the user flip through the
    batch before committing. Rendering is delegated to a callback so the
    dialog stays Qt-only."""

    def __init__(self, paths, render_fn, parent=None):
        super().__init__(parent)
        self.setWindowTitle("流水线预览（执行前请确认效果）")
        self.resize(760, 640)
        self._paths = list(paths)
        self._render = render_fn  # callable(path) -> PIL.Image | None
        self._index = 0
        self._cache = {}

        lay = QVBoxLayout(self)
        self._image_label = QLabel("正在渲染预览…")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(640, 440)
        self._image_label.setStyleSheet("background: #202020;")
        lay.addWidget(self._image_label, 1)

        nav = QHBoxLayout()
        btn_prev = QToolButton()
        btn_prev.setText("◀ 上一张")
        btn_next = QToolButton()
        btn_next.setText("下一张 ▶")
        btn_prev.clicked.connect(lambda: self._step(-1))
        btn_next.clicked.connect(lambda: self._step(1))
        nav.addWidget(btn_prev)
        self._pos_label = QLabel("")
        self._pos_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self._pos_label, 1)
        nav.addWidget(btn_next)
        lay.addLayout(nav)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("确认并执行")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._show()

    def _step(self, delta):
        if not self._paths:
            return
        self._index = (self._index + delta) % len(self._paths)
        self._show()

    def _show(self):
        if not self._paths:
            self._image_label.setText("列表为空")
            return
        path = self._paths[self._index]
        if path not in self._cache:
            img = self._render(path)
            self._cache[path] = img
        img = self._cache[path]
        self._pos_label.setText(
            f"第 {self._index + 1}/{len(self._paths)} 张  {Path(path).name}")
        if img is None:
            self._image_label.setText("预览渲染失败")
            return
        pm = QPixmap.fromImage(numpy_to_qimage(np.array(img.convert("RGB"))))
        scaled = pm.scaled(self._image_label.size(),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)


# ---------------------------------------------------------------------------
# v1.14.0 layout components (Fluent-flavoured, compact)
#
# These exist because the old right panel expressed grouping as
# ``QLabel("—— 文字水印（…）——")`` and put every control in an ad-hoc
# QHBoxLayout/QFormLayout mix. Two consequences: nothing looked like a group,
# and "字号" (clarity) sat next to "大小(短边%)" (on-image size) as two
# identical spin boxes with no way to tell them apart.
# ---------------------------------------------------------------------------

def hint_label(text: str, parent=None) -> QLabel:
    """Muted one-line explanation (styled by the ``#hint`` QSS rule)."""
    lbl = QLabel(text, parent)
    lbl.setObjectName("hint")
    lbl.setWordWrap(True)
    return lbl


def set_variant(widget, variant: str):
    """Tag a widget with a QSS variant ("primary" / "danger" / "ghost").

    Re-polishes immediately because the dynamic property is usually set after
    the app stylesheet has already been applied."""
    widget.setProperty("variant", variant)
    try:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
    except Exception:
        pass
    return widget


class AutoStatusBar(QWidget):
    """自动打码结果条：状态点 + 一行说明。

    v1.22.0：这一页以前只在状态栏（窗口最底端）报「检测到 N 个目标」，页内
    完全没有结果的痕迹。这里把四态压在同一行里——切态时高度不变，布局不跳。

    tone 决定点的颜色（全部读 current_palette，不手写色值）：
    ``idle``/``noimg`` 灰、``ok`` 天蓝、``none``/``stale`` 蜜橙、``error`` 危险橙。
    """

    TONES = {
        "idle": ("text_faint", "text_muted"),
        "noimg": ("text_faint", "text_muted"),
        "ok": ("accent", "text"),
        "none": ("energy_text", "text"),
        "stale": ("energy_text", "text"),
        "error": ("danger_soft_text", "text"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tone = "idle"
        self._text = ""
        self.setMinimumHeight(18)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_state(self, tone: str, text: str):
        self._tone = tone if tone in self.TONES else "idle"
        self._text = text or ""
        self.setToolTip(self._text)
        self.update()

    def tone(self) -> str:
        return self._tone

    def text(self) -> str:
        return self._text

    def paintEvent(self, event):
        pal = current_palette()
        dot_key, text_key = self.TONES[self._tone]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        d = 7.0
        cy = self.height() / 2.0
        # v1.22.1 四色配额：常驻蜜橙左色条（套件 §11.3「左色条」）。
        # 旧版橙色只在 none/stale 两个瞬态出现，静态正常态数不到橙，
        # check_usage.py 判「不合格」（套件 §11.2 对照表点名的就是本页）。
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal["energy_text"]))
        p.drawRoundedRect(QRectF(0.0, 0.0, 3.0, float(self.height())), 1.5, 1.5)
        p.setBrush(QColor(pal[dot_key]))
        p.drawEllipse(QPointF(d / 2.0 + 5.0, cy), d / 2.0, d / 2.0)
        p.setPen(QColor(pal[text_key]))
        rect = self.rect().adjusted(int(d) + 12, 0, 0, 0)
        txt = p.fontMetrics().elidedText(self._text, Qt.ElideRight, rect.width())
        p.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, txt)


class HeartBadge(QWidget):
    """小独爱心「徽章」：状态栏常驻的一枚心粉爱心（v1.22.1 四色配额）。

    填充用 ``sel_border``（套件的心粉色），外圈用 ``heart_ring`` —— 后者是
    套件 ``check_usage.py``「心粉」角色认的色值之一，保证**静态正常态截图**
    里也能数到粉（旧版粉只在批量进度条上，平时整条进度条是隐藏的）。
    形状走「爱心 + 徽章环」，属套件允许的心粉形状。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 18)
        self.setToolTip("小独爱心")

    def paintEvent(self, event):
        pal = current_palette()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        s = min(self.width(), self.height()) * 0.80
        p.setPen(QPen(QColor(pal["heart_ring"]), 2.0))
        p.setBrush(QColor(pal["sel_border"]))
        p.drawPath(HeartProgressBar._heart_path(cx, cy, s))


class SectionCard(QFrame):
    """A bordered card that groups related controls under a real title.

    Put rows into ``self.body`` (a QVBoxLayout). ``title=None`` gives a bare
    surface panel; ``subtitle`` renders as a muted second line under the
    title.

    v1.20.0 小独爱心「镜框卡片」母题：本类接管自身绘制——暖棕软影 + 白圈 +
    浅暖外环（QSS 画不出双层环与影子），hover 时外环染蓝。卡片本体向内缩
    RING=5px 给环带留位，色值全部读 current_palette()。"""

    RING = 5          # white ring (4px) + outer frame line (1px)

    def __init__(self, title: str | None = None, subtitle: str | None = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._hover = False
        ring = self.RING
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE["md"] + ring, SPACE["sm"] + ring,
                                 SPACE["md"] + ring, SPACE["md"] + ring)
        outer.setSpacing(SPACE["sm"])
        if title:
            lbl = QLabel(title, self)
            lbl.setObjectName("cardTitle")
            lbl.setWordWrap(True)
            outer.addWidget(lbl)
        if subtitle:
            outer.addWidget(hint_label(subtitle, self))
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(SPACE["sm"])
        outer.addLayout(self.body)

    # -- 镜框绘制 ------------------------------------------------------
    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def _qcolor(self, raw: str) -> QColor:
        """palette 值 → QColor；rgba(...) 字符串也能解析。"""
        c = QColor(raw)
        if not c.isValid():
            m = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)", raw)
            if m:
                c = QColor(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           int(float(m.group(4)) * 255))
        return c

    def paintEvent(self, event):
        pal = current_palette()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        R = self.RING
        body = self.rect().adjusted(R, R, -R, -R)
        rad = 16.0
        frame = QColor(pal["frame_hover"] if self._hover else pal["frame"])
        shadow = self._qcolor(pal["shadow"])
        # 暖棕软影：向下一侧的分层近似（纯 paintEvent，不挂 effect）
        for i in range(4):
            sh = QColor(shadow)
            sh.setAlpha(int(sh.alpha() * (0.55 - i * 0.12)))
            p.setPen(Qt.NoPen)
            p.setBrush(sh)
            p.drawRoundedRect(body.adjusted(-i, -i + 2, i, i + 3),
                              rad + i, rad + i)
        # 外环 1px + 白圈 4px + 卡片本体
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(frame, 1))
        p.drawRoundedRect(body.adjusted(-1, -1, 1, 1), rad + 5, rad + 5)
        p.setPen(QPen(QColor(pal["surface"]), 4))
        p.drawRoundedRect(body.adjusted(-2, -2, 2, 2), rad + 3.5, rad + 3.5)
        p.setBrush(QColor(pal["surface"]))
        p.setPen(QPen(QColor(pal["border"]), 1))
        p.drawRoundedRect(body, rad, rad)

    def add(self, item):
        """Add a widget or a sub-layout to the card body."""
        if hasattr(item, "addWidget"):
            self.body.addLayout(item)
        else:
            self.body.addWidget(item)
        return item

    def add_row(self, title, control, note=None, stacked=False):
        row = SettingRow(title, control, note=note, stacked=stacked, parent=self)
        self.body.addWidget(row)
        return row


class HeartProgressBar(QProgressBar):
    """小独爱心「爱心瞳进度条」：米色胶囊轨 + 天蓝胶囊填充，填充尽头骑一枚
    心粉爱心瞳（左上白高光点）。全站唯一图腾落到桌面。纯自绘，色值全部
    读 current_palette()（track/accent/sel_border/text）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(False)
        self.setRange(0, 100)
        self.setMinimumHeight(16)

    @staticmethod
    def _heart_path(cx: float, cy: float, s: float) -> QPainterPath:
        path = QPainterPath()
        path.moveTo(cx, cy + s * 0.32)
        path.cubicTo(cx - s * 0.58, cy - s * 0.04, cx - s * 0.26,
                     cy - s * 0.48, cx, cy - s * 0.14)
        path.cubicTo(cx + s * 0.26, cy - s * 0.48, cx + s * 0.58,
                     cy - s * 0.04, cx, cy + s * 0.32)
        path.closeSubpath()
        return path

    def paintEvent(self, event):
        pal = current_palette()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        rad = r.height() / 2

        # 米轨胶囊
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(pal["track"]))
        p.drawRoundedRect(r, rad, rad)

        # 天蓝填充胶囊
        rng = max(1, self.maximum() - self.minimum())
        frac = 0.0 if rng == 0 else (self.value() - self.minimum()) / rng
        frac = min(1.0, max(0.0, frac))
        cw = int(round(r.width() * frac))
        chunk = QRect(r.x(), r.y(), max(0, cw), r.height())
        if cw > rad:
            p.setBrush(QColor(pal["accent"]))
            p.drawRoundedRect(chunk, rad, rad)

        # 爱心瞳骑在填充尽头（有进度才出现）
        if cw > r.height() * 0.8:
            cx = chunk.right() - r.height() * 0.38
            cy = r.center().y() + r.height() * 0.04
            s = r.height() * 1.25
            p.setPen(QPen(QColor(pal["surface"]), 1.4))
            p.setBrush(QColor(pal["sel_border"]))
            p.drawPath(self._heart_path(cx, cy, s))
            # 左上白高光点（瞳孔图腾的高光）
            p.setPen(Qt.NoPen)
            hl = QColor(255, 255, 255, 230)
            p.setBrush(hl)
            p.drawEllipse(QPointF(cx - s * 0.17, cy - s * 0.20),
                          s * 0.09, s * 0.09)

        # 百分比文字：分别以轨道区/填充区裁剪，两段底色各自配可读字色
        txt = f"{int(round(frac * 100))}%"
        f = QFont(self.font())
        f.setBold(True)
        p.setFont(f)
        chunk_region = QRegion(chunk)
        rest = QRegion(r) - chunk_region
        p.save()
        p.setClipRegion(chunk_region)
        p.setPen(QColor(pal["accent_text"]))
        p.drawText(r, Qt.AlignCenter, txt)
        p.restore()
        p.save()
        p.setClipRegion(rest)
        p.setPen(QColor(pal["text"]))
        p.drawText(r, Qt.AlignCenter, txt)
        p.restore()


class SettingRow(QWidget):
    """One labelled setting: title (+ optional note) with the control beside it.

    ``control`` is a QWidget or a layout. With ``stacked=True`` the control
    goes on its own line, which is what wide controls (combos, lists, the
    anchor grid) need when the panel is dragged narrow."""

    def __init__(self, title: str, control, note: str | None = None,
                 stacked: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("settingRow")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(1)

        self.title_label = QLabel(title, self)
        self.title_label.setWordWrap(True)

        def _add_control(container):
            if hasattr(control, "addWidget"):
                container.addLayout(control)
            else:
                container.addWidget(control)

        if stacked:
            outer.addWidget(self.title_label)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACE["sm"])
            _add_control(row)
            outer.addLayout(row)
        else:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACE["sm"])
            row.addWidget(self.title_label)
            # A control with an Ignored horizontal policy (the combos that must
            # be able to shrink inside a narrow panel) is the one that has to
            # take the leftover width. Putting a stretch BEFORE it collapses it
            # to 0 px -- exactly how the 效果 / 检测分辨率 / 自动效果 combos
            # became invisible in this version's first cut.
            expand = False
            try:
                expand = (control.sizePolicy().horizontalPolicy()
                          == QSizePolicy.Ignored)
            except AttributeError:
                expand = False
            if expand:
                _add_control(row)
                row.setStretch(row.count() - 1, 1)
            else:
                row.addStretch(1)
                _add_control(row)
            outer.addLayout(row)

        self.note_label = None
        if note:
            self.note_label = hint_label(note, self)
            outer.addWidget(self.note_label)


_PHOTO_TONE = "#7F7F7F"      # mid grey: readable with white AND dark watermarks


class WatermarkPreview(QWidget):
    """Mock-photo preview of where/how big a watermark lands.

    Added in v1.14.0 to kill the "字号 vs 大小(短边%)" confusion at the
    source: the two are separate knobs (clarity vs on-image size) but looked
    like identical spin boxes. Here the watermark is drawn at the CURRENT
    ``size_frac`` inside a stand-in photo, so moving 大小 visibly resizes it
    while moving 字号 does not.

    Purely informational: it never touches composition or export.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("wmPreview")
        # kept small on purpose: the right panel is a QTabWidget with NO scroll
        # area (v1.6.2 rollback), so every pixel here is taken from the rows
        # below it and an oversized preview clips the bottom of the page
        self.setMinimumSize(120, 62)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._asset = None          # PIL image (or None -> placeholder)
        self._size_frac = 0.15
        self._anchor = "bottom-right"
        self._opacity = 1.0
        self._angle = 0.0

    def set_state(self, asset=None, size_frac=0.15, anchor="bottom-right",
                  opacity=1.0, angle=0.0):
        self._asset = asset
        self._size_frac = max(0.0, float(size_frac))
        self._anchor = anchor or "bottom-right"
        self._opacity = max(0.0, min(1.0, float(opacity)))
        self._angle = float(angle or 0.0)
        self.update()

    # ---------------------------------------------------------------- paint
    def _photo_rect(self) -> QRectF:
        """The stand-in photo: full-bleed inside the widget.

        Deliberately NOT letterboxed to 3:2. The first cut fitted a 3:2 frame
        into a short wide box, which shrank the mock photo to ~80px wide and
        left the watermark a few pixels across; combined with the theme's
        `sunken` fill a WHITE watermark measured ~1.13:1 on it, i.e. invisible
        in the light theme. A mid-grey full-bleed panel keeps both light and
        dark watermarks visible and makes the 大小 difference obvious."""
        # 6px inset, not 3: an anchored watermark sits flush against the photo's
        # edge, and at 3px its outline visually merged with the panel border
        # (flagged in review as a doubled line).
        m = 6
        return QRectF(m, m, max(1, self.width() - 2 * m),
                      max(1, self.height() - 2 * m))

    def paintEvent(self, event):
        pal = current_palette()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        photo = self._photo_rect()
        p.setPen(QPen(QColor(pal["border_strong"]), 1))
        p.setBrush(QBrush(QColor(_PHOTO_TONE)))
        p.drawRoundedRect(photo, 10, 10)

        if self._asset is None:
            p.setPen(QColor(pal["text_faint"]))
            p.drawText(photo, Qt.AlignCenter, "（先选素材 / 生成文字）")
            return

        # watermark width = size_frac * the photo's SHORTER side (same rule as
        # Sticker.place), aspect taken from the asset itself
        short = min(photo.width(), photo.height())
        w = max(2.0, short * self._size_frac)
        ar = self._asset.width / max(1, self._asset.height)
        h = max(2.0, w / ar)

        # anchor -> centre inside the photo (mirrors Sticker.place)
        a = self._anchor
        cx = photo.left() + w / 2 if "left" in a else (
            photo.right() - w / 2 if "right" in a else photo.center().x())
        cy = photo.top() + h / 2 if "top" in a else (
            photo.bottom() - h / 2 if "bottom" in a else photo.center().y())

        img = self._asset.convert("RGBA")
        pm = QPixmap.fromImage(numpy_to_qimage(np.array(img)))
        target = QRectF(cx - w / 2, cy - h / 2, w, h)
        p.setOpacity(self._opacity)
        if self._angle:
            p.translate(target.center())
            p.rotate(self._angle)
            p.translate(-target.center())
        try:
            p.drawPixmap(target, pm, QRectF(pm.rect()))
        except Exception:
            pass
        p.setOpacity(1.0)

        # A 15% watermark is only ~9 px wide in a panel this size, which read as
        # an illegible speck in review. Outline it (in the theme's text colour)
        # so it always reads as a deliberate mark, and print the percentage so
        # the value is unambiguous even when the mark is tiny.
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(pal["text"]), 1))
        p.drawRect(target)
        # near-black on the mid-grey photo (~6:1); the previous light grey was
        # only ~2.6:1 and read as a smudge
        p.setPen(QColor("#111111"))
        p.drawText(photo.adjusted(8, 5, -8, 0), Qt.AlignLeft | Qt.AlignTop,
                   f"{round(self._size_frac * 100)}%")

