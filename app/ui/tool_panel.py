# -*- coding: utf-8 -*-
"""Right tool panel with tabs: Edit, Auto Censor, Stickers, Watermark, Batch."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.config import CLASS_DISPLAY, CLASS_KEYS, CLASS_LABELS, save_config
from ..core.detector import DETECT_SIZES
from ..core.export import describe_pipeline_action
from .widgets import (
    AnchorPicker,
    AutoStatusBar,
    SectionCard,
    WatermarkPreview,
    hint_label,
    pil_to_icon,
    set_variant,
)
from .theme import SPACE
from ..core.selection import AUTO_EFFECT_CHOICES, EFFECTS, normalize_effect

EFFECT_LABELS = {
    "blur": "模糊",
    "mosaic": "马赛克",
    "black_lines": "黑色线条",
    "white_mist": "白色雾气",
    "light": "光效",
    "pixel": "像素块",
    "sticker": "贴纸",
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# 自动打码页的控件度量（v1.22.0）：数字框与下拉框统一 132×30、右缘成列。
# 旧版数字框 93/90/82×31、下拉 217/230×28——同一条竖列里换行会跳 3px。
# ⚠️ 高度不能靠 setFixedHeight/setMinimumHeight：主题 QSS 里的 min-height 会把它
# 盖掉（实测 setFixedSize(132,30) 后高度仍是 28/31），只能写到控件自己的样式表。
# 控件总高 30 = 内容 20 + 上下内边距 8 + 边框 2；主按钮 36 = 内容 26 + 同两项。
_CTRL_W = 132
_CTRL_CSS = "min-height: 20px; max-height: 20px;"
_PRIMARY_CSS = "min-height: 26px;"
# 出厂勾选的类别，与 core.config.default_config()["selected_regions"] 保持一致
# （tests/test_v1220_auto_ui.py 有一条断言守着两者永不分叉）
_DEFAULT_CLASSES = ("vagina", "penis")


def _pin(control, css=_CTRL_CSS, width=_CTRL_W):
    """统一控件度量：宽度写死、高度写进控件样式表（见上面的注释）。"""
    control.setFixedWidth(width)
    control.setStyleSheet(css)
    return control


def _dropped_image_files(mime):
    """Local image files from a drag mime data, or None when not acceptable."""
    if not mime.hasUrls():
        return None
    files = []
    for u in mime.urls():
        local = u.toLocalFile()
        if local:
            p = Path(local)
            if p.is_dir() or p.suffix.lower() in _IMAGE_EXTS:
                files.append(local)
    return files or None


class StickerDropList(QListWidget):
    """Sticker library list. Dropping local image files imports them into
    the sticker library instead of treating them as images to censor."""

    imagesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if _dropped_image_files(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if _dropped_image_files(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        files = _dropped_image_files(event.mimeData())
        if files:
            self.imagesDropped.emit(files)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class StickerDropZone(QWidget):
    """Empty area of the Stickers tab: same image-drop behaviour as the list
    so dropping anywhere on the tab imports stickers (and never falls through
    to the main window's "import as images to censor" handler)."""

    imagesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if _dropped_image_files(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if _dropped_image_files(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        files = _dropped_image_files(event.mimeData())
        if files:
            self.imagesDropped.emit(files)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class ActionConfigDialog(QDialog):
    """Configure ONE pipeline action. Field set depends on the action type.

    auto_censor has no fields of its own: it runs with whatever the 自动打码
    tab is configured to at execution time."""

    def __init__(self, action_type, params, main, parent=None):
        super().__init__(parent)
        self._type = action_type
        self._main = main
        params = params or {}
        names = {"auto_censor": "自动打码", "watermark": "加水印",
                 "resize": "缩放", "convert": "转格式", "rename": "重命名"}
        self.setWindowTitle("配置动作: " + names.get(action_type, action_type))
        lay = QVBoxLayout(self)
        form = QFormLayout()

        if action_type == "auto_censor":
            hint = QLabel("执行时使用「自动打码」页当前的置信度 / IoU / 类别 /\n"
                          "效果与检测分辨率设置，对每张图重新检测。")
            hint.setObjectName("hint")
            hint.setWordWrap(True)
            lay.addWidget(hint)

        elif action_type == "watermark":
            self.asset = QComboBox()
            for name in sorted(main.sticker_library.keys()):
                self.asset.addItem(name, name)
                ix = self.asset.count() - 1
                self.asset.setItemIcon(ix, pil_to_icon(
                    main.sticker_library[name], 40))
            cur = params.get("asset") or main.tool_panel.wm_asset.currentData()
            if cur:
                ix = self.asset.findData(cur)
                if ix >= 0:
                    self.asset.setCurrentIndex(ix)
            form.addRow("素材", self.asset)
            self.anchor = AnchorPicker(params.get("anchor", "bottom-right"))
            form.addRow("位置", self.anchor)
            self.size_frac = QDoubleSpinBox()
            self.size_frac.setRange(2, 90)
            self.size_frac.setSuffix(" %")
            self.size_frac.setValue((params.get("size_frac", 0.15)) * 100)
            form.addRow("大小(短边%)", self.size_frac)
            self.opacity = QDoubleSpinBox()
            self.opacity.setRange(5, 100)
            self.opacity.setSuffix(" %")
            self.opacity.setValue((params.get("opacity", 1.0)) * 100)
            form.addRow("透明度", self.opacity)
            lay.addLayout(form)

        elif action_type == "resize":
            self.mode = QComboBox()
            self.mode.addItem("按百分比", "percent")
            self.mode.addItem("固定尺寸", "size")
            self.mode.setCurrentIndex(1 if params.get("mode") == "size" else 0)
            form.addRow("模式", self.mode)
            self.percent = QDoubleSpinBox()
            self.percent.setRange(1, 1000)
            self.percent.setValue(params.get("percent", 100))
            form.addRow("百分比", self.percent)
            self.width = QSpinBox()
            self.width.setRange(1, 10000)
            self.width.setValue(params.get("width", 1920))
            form.addRow("宽度", self.width)
            self.height = QSpinBox()
            self.height.setRange(1, 10000)
            self.height.setValue(params.get("height", 1080))
            form.addRow("高度", self.height)
            # 常用尺寸预设（与缩放 tab 共用 config 存储）
            self.size_preset = QComboBox()
            self.size_preset.currentIndexChanged.connect(
                self._on_dialog_size_preset)
            for entry in (main.config.get("resize_size_presets") or []):
                self.size_preset.addItem(entry, entry)
            form.addRow("常用尺寸", self.size_preset)
            lay.addLayout(form)

        elif action_type == "convert":
            self.fmt = QComboBox()
            self.fmt.addItems(["PNG", "JPG", "WEBP", "BMP", "GIF"])
            cur = params.get("fmt", "PNG")
            ix = self.fmt.findText(cur)
            self.fmt.setCurrentIndex(ix if ix >= 0 else 0)
            form.addRow("格式", self.fmt)
            self.quality = QSpinBox()
            self.quality.setRange(30, 100)
            self.quality.setValue(int(params.get("quality", 100)))
            form.addRow("质量", self.quality)
            self.strip = QCheckBox("去除元数据")
            self.strip.setChecked(bool(params.get("strip", False)))
            form.addRow("", self.strip)
            lay.addLayout(form)

        elif action_type == "rename":
            self.template = QLineEdit(params.get("template", "{原名}"))
            form.addRow("命名模板", self.template)
            hint = QLabel("变量: {原名} {日期} {时间} {序号}\n"
                          "例: 水印_{日期}_{序号} → 水印_20260909_001")
            hint.setObjectName("hint")
            form.addRow("", hint)
            self.start = QSpinBox()
            self.start.setRange(0, 99999)
            self.start.setValue(int(params.get("start", 1)))
            form.addRow("起始编号", self.start)
            self.digits = QSpinBox()
            self.digits.setRange(1, 10)
            self.digits.setValue(int(params.get("digits", 3)))
            form.addRow("编号位数", self.digits)
            lay.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _on_dialog_size_preset(self):
        entry = self.size_preset.currentData()
        if not entry or "×" not in entry:
            return
        try:
            w_str, h_str = entry.split("×", 1)
            self.width.setValue(int(w_str))
            self.height.setValue(int(h_str))
        except ValueError:
            pass

    def result_params(self) -> dict:
        if self._type == "watermark":
            return {
                "asset": self.asset.currentData(),
                "anchor": self.anchor.anchor(),
                "size_frac": self.size_frac.value() / 100.0,
                "opacity": self.opacity.value() / 100.0,
            }
        if self._type == "resize":
            return {
                "mode": self.mode.currentData() or "percent",
                "percent": self.percent.value(),
                "width": self.width.value(),
                "height": self.height.value(),
            }
        if self._type == "convert":
            return {
                "fmt": self.fmt.currentText(),
                "quality": self.quality.value(),
                "strip": self.strip.isChecked(),
            }
        if self._type == "rename":
            return {
                "template": self.template.text().strip() or "{原名}",
                "start": self.start.value(),
                "digits": self.digits.value(),
            }
        return {}


class ToolPanel(QTabWidget):
    def __init__(self, main, parent=None):
        super().__init__(parent)
        self.main = main
        # 自动打码页的结果状态（_build_auto_tab 里会立刻读到）
        self._result_tone = "idle"
        self._last_counts = {}
        # horizontal Ignored unlocks the splitter: the panel can be dragged
        # as narrow as the user wants (content clips instead of blocking)
        sp = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        sp.setHorizontalStretch(0)
        self.setSizePolicy(sp)
        self.addTab(self._build_edit_tab(), "编辑")
        self.addTab(self._build_auto_tab(), "自动打码")
        self.addTab(self._build_sticker_tab(), "贴纸")
        self.addTab(self._build_watermark_tab(), "水印")
        self.addTab(self._build_batch_tab(), "批量")

    # ---------------- Edit ----------------
    def _build_edit_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["md"])

        tools = [
            ("pointer", "指针"), ("brush", "画笔"), ("eraser", "橡皮擦"),
            ("rect", "矩形"), ("lasso", "套索"),
        ]
        self.tool_group = QButtonGroup(self)
        # 2x3 grid: keeps the row's minimum width ~40% lower so the panel
        # can be dragged narrow
        grid = QGridLayout()
        grid.setHorizontalSpacing(SPACE["sm"])
        grid.setVerticalSpacing(SPACE["sm"])
        self.tool_buttons = {}
        for idx, (value, text) in enumerate(tools):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(value == "pointer")
            btn.clicked.connect(lambda _, v=value: self._set_tool(v))
            self.tool_group.addButton(btn)
            self.tool_buttons[value] = btn
            grid.addWidget(btn, idx // 3, idx % 3)
        # 三列等分：删掉「自动打码」后按钮按 sizeHint 排会短出卡片右缘 34px
        for _c in range(3):
            grid.setColumnStretch(_c, 1)

        card_tools = SectionCard("工具", "在画布上涂抹或框选，效果所见即所得")
        card_tools.add(grid)
        lay.addWidget(card_tools)

        self.effect_combo = QComboBox()
        for eff in EFFECTS:
            self.effect_combo.addItem(EFFECT_LABELS.get(eff, eff), eff)
        self.effect_combo.currentIndexChanged.connect(self._on_effect_changed)
        self.effect_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(1, 100)
        self.strength_spin.setValue(15)
        self.strength_spin.valueChanged.connect(
            lambda v: setattr(self.main.canvas, "current_strength", v)
        )

        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(2, 100)
        self.brush_spin.setValue(30)
        self.brush_spin.valueChanged.connect(
            lambda v: self.main.canvas.set_brush_size(v)
        )

        card_eff = SectionCard("打码效果")
        card_eff.add_row("效果", self.effect_combo)
        card_eff.add_row("强度", self.strength_spin)
        card_eff.add_row("笔刷大小", self.brush_spin, note="画笔与橡皮擦的笔尖直径（像素）")
        lay.addWidget(card_eff)

        btn_clear = QPushButton("清除手动选区")
        btn_clear.clicked.connect(lambda: self.main.clear_manual_selections())
        btn_delete = QPushButton("删除最后一项")
        btn_delete.clicked.connect(lambda: self.main.canvas.delete_selected())
        btn_row = QHBoxLayout()
        btn_row.addWidget(btn_clear)
        btn_row.addWidget(btn_delete)
        card_sel = SectionCard("手动选区")
        card_sel.add(btn_row)
        card_sel.body.addWidget(hint_label(
            "「删除最后一项」按 手动选区 → 贴纸 → 自动打码 的顺序逐层删除；"
            "选中贴纸时也可直接按 Delete 键"))
        lay.addWidget(card_sel)
        lay.addStretch()
        return w

    def _set_tool(self, tool):
        self.main.canvas.set_tool(tool)
        for v, b in self.tool_buttons.items():
            b.setChecked(v == tool)

    def _on_effect_changed(self):
        effect = self.effect_combo.currentData() or "blur"
        self.main.canvas.current_effect = effect
        self.main.canvas.current_strength = self.strength_spin.value()

    # ---------------- Auto ----------------
    def _build_auto_tab(self):
        """v1.22.0 重排：动作区在页首、结果在页内、常用在前。

        旧版把「检测当前图」压在 3 张设置卡之后（页面 76% 处）、与「清除自动
        掩码」同宽同高相贴，检测完页面上看不到任何结果（只写窗口底部的状态
        栏）。现在的顺序：结果条 → 主按钮 →（右对齐）清除 → 检测类别 →
        打码效果 → 高级参数（默认收起）。
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["xl"])      # 组间 20 > 卡内 12，卡片才分得开

        # ---- 动作区（页首）----
        self.auto_status = AutoStatusBar()
        lay.addWidget(self.auto_status)

        act = QVBoxLayout()
        act.setContentsMargins(0, 0, 0, 0)
        act.setSpacing(SPACE["sm"])
        self.btn_detect = QPushButton("检测当前图")
        set_variant(self.btn_detect, "primary")
        self.btn_detect.setStyleSheet(_PRIMARY_CSS)   # 总高 36（见 _PRIMARY_CSS 注释）
        self.btn_detect.clicked.connect(lambda: self.main.auto_detect_current())
        self.btn_detect.setToolTip("按当前参数在这张图上跑一次检测")
        act.addWidget(self.btn_detect)

        clear_row = QHBoxLayout()
        clear_row.setContentsMargins(0, 0, 0, 0)
        clear_row.addStretch(1)
        self.btn_clear_auto = QPushButton("清除自动掩码")
        set_variant(self.btn_clear_auto, "ghost")
        self.btn_clear_auto.clicked.connect(
            lambda: self.main.canvas.clear_auto_mask())
        self.btn_clear_auto.setToolTip(
            "只清掉自动检测出来的掩码；手动选区与贴纸不受影响（可撤销）")
        clear_row.addWidget(self.btn_clear_auto)
        act.addLayout(clear_row)
        lay.addLayout(act)

        # ---- 检测类别（常用 → 排在效果前面）----
        self.class_checks = {}
        card_cls = SectionCard("检测类别")
        cls_head = QHBoxLayout()
        cls_head.setContentsMargins(0, 0, 0, 0)
        cls_head.setSpacing(SPACE["xs"])
        cls_head.addWidget(hint_label("只对勾选的类别打码"), 1)
        for text, handler, tip in (
            ("全选", lambda: self._set_class_checks(set(CLASS_KEYS)),
             "勾选全部类别（检测更全，误检也更多）"),
            ("默认", lambda: self._set_class_checks(set(_DEFAULT_CLASSES)),
             "恢复出厂勾选：阴部 + 阴茎"),
        ):
            link = QPushButton(text)
            set_variant(link, "ghost")
            link.clicked.connect(handler)
            link.setToolTip(tip)
            cls_head.addWidget(link)
        card_cls.body.addLayout(cls_head)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE["md"])
        grid.setVerticalSpacing(SPACE["xs"])
        for i, key in enumerate(CLASS_KEYS):
            cb = QCheckBox(CLASS_LABELS.get(key, key))
            cb.setChecked(key in _DEFAULT_CLASSES)
            cb.setToolTip(CLASS_DISPLAY.get(key, key))
            cb.toggled.connect(self._on_auto_param_changed)
            self.class_checks[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        card_cls.body.addLayout(grid)
        lay.addWidget(card_cls)

        # ---- 打码效果（改了就实时重绘，不用重检测）----
        self.auto_effect_combo = QComboBox()
        for eff, label in AUTO_EFFECT_CHOICES:
            self.auto_effect_combo.addItem(label, eff)
        self.auto_effect_combo.currentIndexChanged.connect(
            lambda _: self._update_auto_preview())
        _pin(self.auto_effect_combo)
        self.auto_strength_spin = QSpinBox()
        self.auto_strength_spin.setRange(1, 100)
        self.auto_strength_spin.setValue(15)
        _pin(self.auto_strength_spin)
        self.auto_strength_spin.valueChanged.connect(
            lambda _: self._update_auto_preview())
        self.mask_dilate_spin = QSpinBox()
        self.mask_dilate_spin.setRange(0, 50)
        self.mask_dilate_spin.setValue(5)
        _pin(self.mask_dilate_spin)
        self.mask_dilate_spin.valueChanged.connect(
            lambda _: self._update_auto_preview())

        card_eff = SectionCard("打码效果")
        card_eff.add_row("自动效果", self.auto_effect_combo)
        card_eff.add_row("强度", self.auto_strength_spin)
        dilate_row = card_eff.add_row("覆盖扩展", self.mask_dilate_spin,
                                      note="向外扩充掩码的像素数")
        dilate_row.setToolTip("防止检测掩码太小盖不住目标，向外扩充的像素数。0 = 不外扩")
        lay.addWidget(card_eff)

        # ---- 高级参数（专业旋钮，默认收起）----
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.30)
        _pin(self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.70)
        _pin(self.iou_spin)
        self.input_size_combo = QComboBox()
        for size in DETECT_SIZES:
            self.input_size_combo.addItem(f"{size}×{size}", size)
        self.input_size_combo.currentIndexChanged.connect(
            lambda _: self._on_auto_param_changed())
        _pin(self.input_size_combo)
        # 改了就作废旧结果：这两个只影响下一次检测，不会动已生成的掩码
        self.conf_spin.valueChanged.connect(lambda _: self._on_auto_param_changed())
        self.iou_spin.valueChanged.connect(lambda _: self._on_auto_param_changed())

        card_det = SectionCard("检测参数")
        conf_row = card_det.add_row("置信度", self.conf_spin, note="低于此值就忽略；调低更灵敏")
        conf_row.setToolTip("低于此可信度的检测结果会被忽略。调高减少误检，调低捕获更多目标。")
        iou_row = card_det.add_row("IoU", self.iou_spin, note="重叠框合并阈值（NMS）")
        iou_row.setToolTip("同一目标重叠框的合并阈值。调高要求更高重合度，重复框更少。")
        size_row = card_det.add_row("检测分辨率", self.input_size_combo,
                                    note="模型输入尺寸；CPU 可调低提速")
        size_row.setToolTip("模型推理时的输入尺寸。GPU 下 1280 也很快；CPU 下降低更快，小目标可能漏检。")
        self.plate_card = card_det

        self.btn_advanced = QPushButton()
        set_variant(self.btn_advanced, "ghost")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setToolTip("置信度 / IoU / 检测分辨率：一年调一次的东西")
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        self._set_advanced_text(False)
        lay.addWidget(self.btn_advanced)
        card_det.setVisible(False)
        lay.addWidget(card_det)

        lay.addStretch(1)
        self.refresh_auto_state()
        return w

    # ---- 自动打码页的状态机 -------------------------------------------
    def _set_advanced_text(self, opened: bool):
        # 不用 ▸/▾ 三角形：那类字形在缺字形回退的环境会变豆腐块，「展开/收起」
        # 是纯文字，任何字体都能渲染（v1.22.0 视觉验收提过这一点）
        self.btn_advanced.setText(
            ("收起高级参数：置信度 / IoU / 分辨率" if opened
             else "展开高级参数：置信度 / IoU / 分辨率"))

    def _toggle_advanced(self, opened: bool):
        self.plate_card.setVisible(opened)
        self._set_advanced_text(opened)

    def _set_status(self, tone: str, text: str):
        self._result_tone = tone
        self.auto_status.set_state(tone, text)

    def _refresh_class_counts(self):
        for key, cb in self.class_checks.items():
            base = CLASS_LABELS.get(key, key)
            n = self._last_counts.get(key, 0)
            cb.setText(f"{base} ×{n}" if n else base)

    def _set_class_checks(self, keys):
        """批量勾选（全选 / 默认）：合并成一次参数变更，不逐格触发。"""
        for key, cb in self.class_checks.items():
            cb.blockSignals(True)
            cb.setChecked(key in keys)
            cb.blockSignals(False)
        self._on_auto_param_changed()

    def _on_auto_param_changed(self, *_):
        """置信度 / IoU / 分辨率 / 类别变化：已生成的掩码不再是这个参数的产物。"""
        if self._result_tone in ("ok", "none"):
            self._last_counts = {}
            self._refresh_class_counts()
            self._set_status("stale", "参数已改动 · 需重新检测")
        self._update_auto_preview()

    def refresh_auto_state(self):
        """只同步按钮可用性（掩码增删时由画布 state_changed 触发）。"""
        has_img = getattr(self.main.canvas, "image_np", None) is not None
        mask = getattr(self.main.canvas, "auto_mask", None)
        # 测试/替身 main 上没有真掩码数组（AutoStub 的 __getattr__ 会返回一个
        # 函数），所以先确认它真的是个 ndarray 再说 any()
        has_mask = bool(mask is not None and hasattr(mask, "any") and mask.any())
        self.btn_detect.setEnabled(has_img)
        self.btn_clear_auto.setEnabled(has_img and has_mask)
        if not has_img:
            self._set_status("noimg", "先导入一张图片")
        return has_img, has_mask

    def on_image_changed(self):
        """切图（或图被删空）时重置结果：上一张的命中数不能留在这一张上。"""
        self._last_counts = {}
        self._refresh_class_counts()
        has_img, has_mask = self.refresh_auto_state()
        if not has_img:
            self._set_status("noimg", "先导入一张图片")
        elif has_mask:
            self._set_status("ok", "本图已有自动掩码")
        else:
            self._set_status("idle", "尚未检测")

    def show_detect_result(self, dets):
        """把主窗口的检测结果落到页内结果条 + 每类命中数。"""
        counts = {}
        for d in dets:
            key = getattr(d, "class_name", "") or ""
            counts[key] = counts.get(key, 0) + 1
        self._last_counts = counts
        self._refresh_class_counts()
        self.refresh_auto_state()
        if not dets:
            mask = getattr(self.main.canvas, "auto_mask", None)
            if mask is not None and hasattr(mask, "any") and bool(mask.any()):
                self._set_status("none", "未检测到目标 · 旧的自动掩码仍在画布上")
            else:
                self._set_status("none", "未检测到目标 · 可调低置信度再试")
            return
        parts = []
        for key in CLASS_KEYS:                      # 固定顺序，读数不看运气
            n = counts.get(key)
            if n:
                parts.append(f"{CLASS_LABELS.get(key, key)}×{n}")
        extra = sum(n for k, n in counts.items() if k not in CLASS_KEYS)
        if extra:
            parts.append(f"其他×{extra}")
        self._set_status("ok", f"本次命中 {len(dets)} 处 · " + " ".join(parts))

    def show_detect_error(self, msg: str):
        self._set_status("error", f"检测失败：{msg}")

    # ---------------- Stickers ----------------
    def _build_sticker_tab(self):
        w = StickerDropZone()
        w.imagesDropped.connect(self._on_sticker_drop)
        lay = QVBoxLayout(w)
        self.sticker_list = StickerDropList()
        self.sticker_list.setObjectName("stickerWall")  # skin: chip outline
        # thumbnail wall (nEO iMAGING style): icon mode with previews
        self.sticker_list.setViewMode(QListWidget.IconMode)
        self.sticker_list.setIconSize(QSize(64, 64))
        self.sticker_list.setGridSize(QSize(78, 96))
        self.sticker_list.setResizeMode(QListWidget.Adjust)
        self.sticker_list.setMovement(QListWidget.Static)
        self.sticker_list.setWordWrap(True)
        self.sticker_list.imagesDropped.connect(self._on_sticker_drop)
        self.sticker_list.itemDoubleClicked.connect(lambda _: self.main.add_current_sticker())
        # v1.13 library management: right-click to add / favourite / rename /
        # delete a LIBRARY ASSET (previously import-only, so duplicate
        # "name_2/name_3" copies piled up in app/stickers with no way out)
        self.sticker_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sticker_list.customContextMenuRequested.connect(
            self._on_sticker_list_menu)
        lay.setSpacing(SPACE["md"])
        lay.addWidget(self.sticker_list, 1)
        # empty-state line: the wall is the largest blank area in the UI when
        # the library has nothing in it
        self.sticker_empty_label = hint_label(
            "素材库还是空的 —— 把图片拖进本页，或点下面的「导入贴纸…」")
        lay.addWidget(self.sticker_empty_label)

        self.sticker_filter = QLineEdit()
        self.sticker_filter.setPlaceholderText("搜索素材…")
        self.sticker_filter.setClearButtonEnabled(True)
        self.sticker_filter.textChanged.connect(lambda _: self.refresh_sticker_list())
        self.sticker_fav_only = QCheckBox("只看收藏")
        self.sticker_fav_only.toggled.connect(lambda: self.refresh_sticker_list())
        find_row = QHBoxLayout()
        find_row.setSpacing(SPACE["sm"])
        find_row.addWidget(self.sticker_filter, 1)
        find_row.addWidget(self.sticker_fav_only)

        btn_add = QPushButton("添加到当前图")
        set_variant(btn_add, "primary")
        btn_add.clicked.connect(lambda: self.main.add_current_sticker())
        btn_import = QPushButton("导入贴纸…")
        btn_import.clicked.connect(self._pick_sticker_files)
        btn_folder = QPushButton("导入文件夹…")
        btn_folder.clicked.connect(self._pick_sticker_folder)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(lambda: self.main.refresh_stickers(self.sticker_list))
        # 2-column grid instead of one long row: at the panel's minimum width a
        # single row of four buttons gets clipped
        btn_grid = QGridLayout()
        btn_grid.setHorizontalSpacing(SPACE["sm"])
        btn_grid.setVerticalSpacing(SPACE["sm"])
        btn_grid.addWidget(btn_add, 0, 0, 1, 2)
        btn_grid.addWidget(btn_import, 1, 0)
        btn_grid.addWidget(btn_folder, 1, 1)
        btn_grid.addWidget(btn_refresh, 2, 0, 1, 2)

        card_lib = SectionCard(
            "素材库", "双击素材即添加到当前图；右键可收藏 / 重命名 / 删除")
        card_lib.add(find_row)
        card_lib.add(btn_grid)
        card_lib.body.addWidget(
            hint_label("图片或文件夹可直接拖进本页加入素材库"))
        lay.addWidget(card_lib)
        self.refresh_sticker_list()
        return w

    def _pick_sticker_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择贴纸文件夹", "")
        if folder:
            self._on_sticker_drop([folder])

    def _on_sticker_drop(self, files):
        try:
            added = self.main.import_sticker_files(files)
        except Exception as e:
            QMessageBox.warning(self, "导入贴纸失败", str(e))
            return
        if not added:
            QMessageBox.information(self, "导入贴纸", "未找到可导入的图片文件。")
            return
        self.main.statusBar().showMessage(f"已导入 {len(added)} 张贴纸")

    def _pick_sticker_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择贴纸图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.gif *.bmp);;所有文件 (*.*)"
        )
        if files:
            self._on_sticker_drop(files)

    def refresh_sticker_list(self):
        """Thumbnail wall: every library asset shows its own preview.

        The search box and the favourites toggle are applied HERE, so every
        refresh path (import / delete / rename / favourite) keeps the view
        consistent instead of resetting the filter."""
        needle = (self.sticker_filter.text().strip().lower()
                  if hasattr(self, "sticker_filter") else "")
        fav_only = (self.sticker_fav_only.isChecked()
                    if hasattr(self, "sticker_fav_only") else False)
        favs = set(self.main.config.get("sticker_favorites") or [])
        self.sticker_list.clear()
        for name in sorted(self.main.sticker_library.keys()):
            if needle and needle not in name.lower():
                continue
            starred = name in favs
            if fav_only and not starred:
                continue
            item = QListWidgetItem(f"★ {name}" if starred else name)
            img = self.main.sticker_library.get(name)
            if img is not None:
                item.setIcon(pil_to_icon(img, 64))
                item.setToolTip(
                    f"{name}\n双击添加到当前图\n右键：收藏 / 重命名 / 删除")
            item.setData(Qt.UserRole, name)
            self.sticker_list.addItem(item)
        if hasattr(self, "sticker_empty_label"):
            self.sticker_empty_label.setVisible(
                not self.main.sticker_library)

    def _on_sticker_list_menu(self, pos):
        """Right-click menu for one library asset."""
        item = self.sticker_list.itemAt(pos)
        if item is None:
            return
        name = item.data(Qt.UserRole)
        starred = name in set(self.main.config.get("sticker_favorites") or [])
        menu = QMenu(self)
        act_add = menu.addAction("添加到当前图")
        act_star = menu.addAction("取消收藏" if starred else "收藏")
        menu.addSeparator()
        act_rename = menu.addAction("重命名素材…")
        act_del = menu.addAction("从库中删除")
        chosen = menu.exec(self.sticker_list.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is act_add:
            self.sticker_list.setCurrentItem(item)
            self.main.add_current_sticker()
            return
        if chosen is act_star:
            self.main.toggle_sticker_favorite(name)
        elif chosen is act_rename:
            self.main.rename_sticker_asset(name)
        elif chosen is act_del:
            self.main.delete_sticker_asset(name)
        # the list content just changed: re-render keeping the current filter
        self.refresh_sticker_list()

    # ---------------- Watermark ----------------
    ANCHORS = [
        ("右下角", "bottom-right"), ("左下角", "bottom-left"),
        ("右上角", "top-right"), ("左上角", "top-left"),
        ("顶部居中", "top-center"), ("底部居中", "bottom-center"),
        ("居中", "center"),
    ]

    def _build_watermark_tab(self):
        from PySide6.QtGui import QColor
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["md"])

        # ---------------- 文字水印：生成素材 ----------------
        self.wm_text = QLineEdit()
        self.wm_text.setPlaceholderText("输入文字生成水印…")
        self.wm_gen_btn = QPushButton("生成文字")
        set_variant(self.wm_gen_btn, "primary")
        self.wm_gen_btn.clicked.connect(self.main.create_text_watermark_from_panel)
        gen = QHBoxLayout()
        gen.setSpacing(SPACE["sm"])
        gen.addWidget(self.wm_text, 1)
        gen.addWidget(self.wm_gen_btn)

        self.wm_color = QPushButton("文字颜色")
        self.wm_color.setObjectName("wmColorBtn")
        self.wm_color.clicked.connect(self._pick_wm_color)
        self._wm_color_val = (255, 255, 255, 255)
        self._show_wm_color(QColor(255, 255, 255))

        self.wm_font = QSpinBox()
        self.wm_font.setRange(12, 300)
        self.wm_font.setValue(64)
        self.wm_font.setToolTip("文字水印字号（像素）")

        self.wm_tpl = QComboBox()
        self.wm_tpl.currentIndexChanged.connect(self._wm_tpl_selected)
        self.wm_tpl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        btn_tpl_save = QPushButton("存为模板")
        btn_tpl_save.clicked.connect(self.main.save_text_wm_template)
        btn_tpl_del = QPushButton("删除")
        btn_tpl_del.clicked.connect(self.main.delete_text_wm_template)
        tpl_row = QHBoxLayout()
        tpl_row.setSpacing(SPACE["sm"])
        tpl_row.addWidget(self.wm_tpl, 1)
        tpl_row.addWidget(btn_tpl_save)
        tpl_row.addWidget(btn_tpl_del)

        card_text = SectionCard(
            "文字水印", "输入文字生成一个素材，之后像普通贴纸一样使用")
        card_text.add(gen)
        card_text.add_row("文字颜色", self.wm_color)
        card_text.add_row(
            "文字精度（字号）", self.wm_font,
            note="只影响清晰度，不改变水印在照片上的大小")
        card_text.add(tpl_row)
        lay.addWidget(card_text)

        # ---------------- 图片水印：选素材 + 定位 + 大小 ----------------
        # 素材 belongs HERE (it is what gets applied); it used to sit above the
        # text section, which made it look like part of the text generator.
        self.wm_asset = QComboBox()
        self.wm_asset.currentIndexChanged.connect(self._wm_asset_changed)

        self.wm_anchor_picker = AnchorPicker("bottom-right")
        anchor_row = QHBoxLayout()
        anchor_row.setSpacing(SPACE["sm"])
        anchor_row.addWidget(self.wm_anchor_picker)
        anchor_row.addStretch(1)

        self.wm_size = QDoubleSpinBox()
        self.wm_size.setRange(2, 90)
        self.wm_size.setValue(15)
        self.wm_size.setSuffix(" %")
        self.wm_size.setToolTip("水印宽度 = 图片短边的百分比")

        self.wm_opacity = QDoubleSpinBox()
        self.wm_opacity.setRange(5, 100)
        self.wm_opacity.setValue(100)
        self.wm_opacity.setSuffix(" %")

        self.wm_preview = WatermarkPreview()

        card_img = SectionCard(
            "图片水印", "选素材 → 选位置 → 定大小，然后加到当前图或整批")
        card_img.add_row("素材", self.wm_asset)
        card_img.add_row("位置", anchor_row, stacked=True,
                         note="水印钉在图片的哪个角（竖图横图都贴同一个角）")
        self.wm_size_row = card_img.add_row("图上大小（占短边）", self.wm_size)
        # v1.22.0：这行 note 曾经写「水印宽度 = 图片短边的百分比」——与行标题、
        # 控件自身的 " %" 后缀三次重复，还把这一页顶到溢出（九宫格被压到 24px）。
        # 说明挪进 tooltip，省下的 15px 还给九宫格。
        self.wm_size_row.setToolTip("水印宽度 = 图片短边的百分比（2–90%）")
        card_img.add_row("透明度", self.wm_opacity)
        card_img.body.addWidget(self.wm_preview)

        btn_cur = QPushButton("加到当前图")
        set_variant(btn_cur, "primary")
        btn_cur.clicked.connect(lambda: self._apply_watermark("current"))
        btn_all = QPushButton("批量加到全部")
        btn_all.clicked.connect(lambda: self._apply_watermark("all"))
        wm_row = QHBoxLayout()
        wm_row.setSpacing(SPACE["sm"])
        wm_row.addWidget(btn_cur)
        wm_row.addWidget(btn_all)
        card_img.add(wm_row)

        # v1.13: removal used to require clicking each watermark on the canvas
        btn_clear_wm = QPushButton("移除当前图全部水印")
        btn_clear_wm.setToolTip("只移除水印，普通贴纸保留（可撤销）")
        btn_clear_wm.clicked.connect(self.main.remove_watermarks_current)
        card_img.add(btn_clear_wm)
        lay.addWidget(card_img)

        # ---------------- 水印方案 ----------------
        self.wm_preset = QComboBox()
        self.wm_preset.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        btn_apply = QPushButton("应用")
        btn_apply.clicked.connect(self.main.apply_watermark_preset)
        pre_row = QHBoxLayout()
        pre_row.setSpacing(SPACE["sm"])
        pre_row.addWidget(self.wm_preset, 1)
        pre_row.addWidget(btn_apply)
        btn_save = QPushButton("存为方案")
        btn_save.clicked.connect(self.main.save_watermark_preset)
        btn_append = QPushButton("追加到方案")
        btn_append.clicked.connect(self.main.append_watermark_preset)
        btn_del = QPushButton("删除方案")
        btn_del.clicked.connect(self.main.delete_watermark_preset)
        pre_row2 = QHBoxLayout()
        pre_row2.setSpacing(SPACE["sm"])
        pre_row2.addWidget(btn_save)
        pre_row2.addWidget(btn_append)
        pre_row2.addWidget(btn_del)

        card_pre = SectionCard("水印方案")
        card_pre.add(pre_row)
        card_pre.add(pre_row2)
        lay.addWidget(card_pre)

        # keep the preview in sync with every knob that affects it
        self.wm_asset.currentIndexChanged.connect(lambda _: self._refresh_wm_preview())
        self.wm_anchor_picker.anchorChanged.connect(
            lambda _: self._refresh_wm_preview())
        self.wm_size.valueChanged.connect(lambda _: self._refresh_wm_preview())
        self.wm_opacity.valueChanged.connect(lambda _: self._refresh_wm_preview())

        lay.addStretch()

        self.refresh_watermark_assets()
        self.refresh_text_wm_templates()
        self.refresh_watermark_presets()
        self._refresh_wm_preview()
        return w

    def _refresh_wm_preview(self):
        """Feed the mock-photo preview the current watermark settings."""
        if not hasattr(self, "wm_preview"):
            return
        asset = None
        try:
            name = self.wm_asset.currentData()
            asset = self.main.sticker_library.get(name) if name else None
        except Exception:
            asset = None
        self.wm_preview.set_state(
            asset=asset,
            size_frac=self.wm_size.value() / 100.0,
            anchor=self.wm_anchor_picker.anchor(),
            opacity=self.wm_opacity.value() / 100.0,
        )

    def refresh_text_wm_templates(self):
        """(Re)fill the text-watermark template combo from config."""
        cfg = self.main.config
        templates = cfg.get("text_wm_templates") or []
        current = self.wm_tpl.currentData()
        self.wm_tpl.blockSignals(True)
        self.wm_tpl.clear()
        self.wm_tpl.addItem("(文字模板)", None)
        for t in templates:
            self.wm_tpl.addItem(t["name"], t["name"])
        if current is not None:
            ix = self.wm_tpl.findData(current)
            if ix >= 0:
                self.wm_tpl.setCurrentIndex(ix)
        self.wm_tpl.blockSignals(False)

    def _wm_tpl_selected(self):
        """Apply the chosen template to the generator inputs."""
        name = self.wm_tpl.currentData()
        if not name:
            return
        for t in (self.main.config.get("text_wm_templates") or []):
            if t["name"] == name:
                self.wm_text.setText(t.get("text", ""))
                self.wm_opacity.setValue(float(t.get("opacity", 1.0)) * 100)
                self.wm_font.setValue(int(t.get("font_size", 64)))
                c = t.get("color") or (255, 255, 255, 255)
                self._wm_color_val = tuple(c)
                self._show_wm_color(QColor(*c[:3]))
                return

    def refresh_watermark_presets(self):
        """(Re)fill the watermark preset combo from config."""
        cfg = self.main.config
        presets = cfg.get("watermark_presets") or []
        current = self.wm_preset.currentData()
        self.wm_preset.blockSignals(True)
        self.wm_preset.clear()
        self.wm_preset.addItem("(选择方案)", None)
        for p in presets:
            n_items = len(p.get("items", []))
            self.wm_preset.addItem(f"{p['name']} ({n_items}项)", p["name"])
        if current is not None:
            ix = self.wm_preset.findData(current)
            if ix >= 0:
                self.wm_preset.setCurrentIndex(ix)
        self.wm_preset.blockSignals(False)

    def current_watermark_config(self) -> dict:
        """Current page config as one preset item."""
        return {
            "asset": self.wm_asset.currentData(),
            "anchor": self.wm_anchor_picker.anchor(),
            "size_frac": self.wm_size.value() / 100.0,
            "opacity": self.wm_opacity.value() / 100.0,
        }

    def _wm_color_changed(self, rgba):
        self._wm_color_val = tuple(rgba)

    def refresh_watermark_assets(self):
        """(Re)fill the watermark asset combo from the shared sticker library.
        Only image assets make sense as watermarks, so skip nothing - all
        library entries are images."""
        current = self.wm_asset.currentData()
        self.wm_asset.blockSignals(True)
        self.wm_asset.clear()
        for name in sorted(self.main.sticker_library.keys()):
            self.wm_asset.addItem(name, name)
        if current is not None:
            ix = self.wm_asset.findData(current)
            if ix >= 0:
                self.wm_asset.setCurrentIndex(ix)
        self.wm_asset.blockSignals(False)
        self._wm_asset_changed()
        self._refresh_wm_preview()

    def _wm_asset_changed(self):
        # a watermark is any library asset; no special state needed
        return

    def _pick_wm_color(self):
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog
        col = QColorDialog.getColor(QColor(255, 255, 255), self, "文字颜色")
        if col.isValid():
            self._wm_color_val = tuple(col.getRgb())
            self._show_wm_color(col)

    def _show_wm_color(self, col):
        """Swatch + the current value as the button text.

        Before v1.14 the button just said "文字颜色", which repeated the row
        label and never showed the颜色 actually in use."""
        self.wm_color.setText(col.name().upper())
        self.wm_color.setToolTip(f"当前文字颜色 {col.name().upper()}（点击更改）")
        self.wm_color.setStyleSheet(
            f"background-color: {col.name()};"
            f" color: {'#000' if col.lightness() > 140 else '#fff'};"
        )

    def _wm_color_rgba(self):
        return getattr(self, "_wm_color_val", (255, 255, 255, 255))

    def _apply_watermark(self, scope):
        name = self.wm_asset.currentData()
        if not name:
            QMessageBox.warning(self, "提示", "请先生成或选择一个水印素材。")
            return
        anchor = self.wm_anchor_picker.anchor()
        frac = self.wm_size.value() / 100.0
        opacity = self.wm_opacity.value() / 100.0
        try:
            self.main.apply_watermark(
                scope=scope, name=name, anchor=anchor,
                size_frac=frac, opacity=opacity,
            )
        except ValueError as e:
            QMessageBox.warning(self, "提示", str(e))
            return
        if scope == "all":
            QMessageBox.information(
                self, "水印", f"已为全部图片添加水印。\n可切换到「指针」工具逐张拖动/缩放调整。"
            )

    # ---------------- Batch ----------------
    def _build_batch_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["md"])

        btn_save = QPushButton("打开批量导出对话框")
        set_variant(btn_save, "primary")
        btn_save.clicked.connect(lambda: self.main.open_save_dialog())
        lay.addWidget(btn_save)

        sub = QTabWidget()
        sub.setObjectName("subTabs")   # skin: stronger selected capsule
        sub.addTab(self._build_convert_tab(), "转格式")
        sub.addTab(self._build_resize_tab(), "缩放")
        sub.addTab(self._build_rename_tab(), "重命名")
        sub.addTab(self._build_pipeline_tab(), "流水线")
        lay.addWidget(sub, 1)
        return w

    def _build_convert_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["md"])
        self.convert_fmt = QComboBox()
        self.convert_fmt.addItems(["PNG", "JPG", "WEBP", "BMP", "GIF"])
        self.convert_quality = QSpinBox()
        self.convert_quality.setRange(30, 100)
        self.convert_quality.setValue(100)

        card_fmt = SectionCard("输出格式")
        card_fmt.add_row("格式", self.convert_fmt)
        card_fmt.add_row("质量", self.convert_quality,
                         note="仅 JPG / WEBP 有效；100 = 最高质量")
        lay.addWidget(card_fmt)

        self.convert_out = QLineEdit()
        self.convert_out.setPlaceholderText("输出目录")
        browse = QPushButton("浏览")
        browse.clicked.connect(lambda: self._browse(self.convert_out))
        row = QHBoxLayout()
        row.setSpacing(SPACE["sm"])
        row.addWidget(self.convert_out, 1)
        row.addWidget(browse)
        card_out = SectionCard("输出")
        card_out.add(row)
        lay.addWidget(card_out)

        self.convert_btn = QPushButton("开始转换")
        set_variant(self.convert_btn, "primary")
        self.convert_btn.clicked.connect(self.main.batch_convert)
        lay.addWidget(self.convert_btn)
        lay.addStretch()
        return w

    def _build_resize_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["md"])
        self.resize_mode = QComboBox()
        self.resize_mode.addItem("按百分比", "percent")
        self.resize_mode.addItem("固定尺寸", "size")
        self.resize_percent = QDoubleSpinBox()
        self.resize_percent.setRange(1, 1000)
        self.resize_percent.setValue(100)
        self.resize_width = QSpinBox()
        self.resize_width.setRange(1, 10000)
        self.resize_width.setValue(1920)
        self.resize_height = QSpinBox()
        self.resize_height.setRange(1, 10000)
        self.resize_height.setValue(1080)

        card_mode = SectionCard("缩放方式")
        card_mode.add_row("模式", self.resize_mode)
        card_mode.add_row("百分比", self.resize_percent)
        card_mode.add_row("宽度", self.resize_width)
        card_mode.add_row("高度", self.resize_height)
        lay.addWidget(card_mode)

        # 常用尺寸预设（保存/选用/删除，config 持久化）
        self.resize_presets = QComboBox()
        self.resize_presets.currentIndexChanged.connect(
            self._on_resize_preset_selected)
        pre_row = QHBoxLayout()
        pre_row.setSpacing(SPACE["sm"])
        pre_row.addWidget(self.resize_presets, 1)
        btn_pre_save = QPushButton("存")
        btn_pre_save.setFixedWidth(38)
        btn_pre_save.setToolTip("把当前宽高存为常用尺寸")
        btn_pre_save.clicked.connect(self._save_resize_preset)
        pre_row.addWidget(btn_pre_save)
        btn_pre_del = QPushButton("删")
        btn_pre_del.setFixedWidth(38)
        btn_pre_del.setToolTip("删除选中的常用尺寸")
        btn_pre_del.clicked.connect(self._delete_resize_preset)
        pre_row.addWidget(btn_pre_del)
        card_pre = SectionCard("常用尺寸", "存好后可在缩放与流水线里直接选用")
        card_pre.add(pre_row)
        lay.addWidget(card_pre)

        self.resize_out = QLineEdit()
        self.resize_out.setPlaceholderText("输出目录")
        browse = QPushButton("浏览")
        browse.clicked.connect(lambda: self._browse(self.resize_out))
        row = QHBoxLayout()
        row.setSpacing(SPACE["sm"])
        row.addWidget(self.resize_out, 1)
        row.addWidget(browse)
        card_out = SectionCard("输出")
        card_out.add(row)
        lay.addWidget(card_out)

        self.resize_btn = QPushButton("开始缩放")
        set_variant(self.resize_btn, "primary")
        self.resize_btn.clicked.connect(self.main.batch_resize)
        lay.addWidget(self.resize_btn)
        lay.addStretch()
        self.refresh_resize_presets()
        return w

    # ---- common size presets (nEO iMAGING AddCommonSize equivalent) ----
    def _size_preset_list(self) -> list:
        return list(self.main.config.get("resize_size_presets") or [])

    def refresh_resize_presets(self):
        combo = self.resize_presets
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(常用尺寸)", None)
        for entry in self._size_preset_list():
            combo.addItem(entry, entry)
        combo.blockSignals(False)

    def _on_resize_preset_selected(self):
        entry = self.resize_presets.currentData()
        if not entry or "×" not in entry:
            return
        try:
            w_str, h_str = entry.split("×", 1)
            self.resize_width.setValue(int(w_str))
            self.resize_height.setValue(int(h_str))
        except ValueError:
            pass

    def _save_resize_preset(self):
        entry = f"{self.resize_width.value()}×{self.resize_height.value()}"
        presets = self._size_preset_list()
        if entry not in presets:
            presets.append(entry)
            self.main.config["resize_size_presets"] = presets
            save_config(self.main.config)
        self.refresh_resize_presets()
        ix = self.resize_presets.findData(entry)
        if ix >= 0:
            self.resize_presets.setCurrentIndex(ix)

    def _delete_resize_preset(self):
        entry = self.resize_presets.currentData()
        if not entry:
            return
        presets = [p for p in self._size_preset_list() if p != entry]
        self.main.config["resize_size_presets"] = presets
        save_config(self.main.config)
        self.refresh_resize_presets()

    def _build_rename_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["md"])
        self.rename_prefix = QLineEdit("IMG")
        self.rename_start = QSpinBox()
        self.rename_start.setRange(0, 99999)
        self.rename_start.setValue(1)
        self.rename_digits = QSpinBox()
        self.rename_digits.setRange(1, 10)
        self.rename_digits.setValue(3)

        card_name = SectionCard("命名规则")
        card_name.add_row("前缀", self.rename_prefix)
        card_name.add_row("起始编号", self.rename_start)
        card_name.add_row("编号位数", self.rename_digits,
                          note="3 → IMG001；编号严格跟随左侧列表顺序")
        lay.addWidget(card_name)

        self.rename_out = QLineEdit()
        self.rename_out.setPlaceholderText("输出目录")
        browse = QPushButton("浏览")
        browse.clicked.connect(lambda: self._browse(self.rename_out))
        row = QHBoxLayout()
        row.setSpacing(SPACE["sm"])
        row.addWidget(self.rename_out, 1)
        row.addWidget(browse)
        card_out = SectionCard("输出")
        card_out.add(row)
        lay.addWidget(card_out)

        self.rename_btn = QPushButton("开始重命名")
        set_variant(self.rename_btn, "primary")
        self.rename_btn.clicked.connect(self.main.batch_rename)
        lay.addWidget(self.rename_btn)
        lay.addStretch()
        return w

    # ---------------- Pipeline (SuperBatch-style) ----------------
    PIPELINE_ACTION_TYPES = [
        ("auto_censor", "自动打码"),
        ("watermark", "加水印"),
        ("resize", "缩放"),
        ("convert", "转格式"),
        ("rename", "重命名"),
    ]

    def _build_pipeline_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(SPACE["md"])

        self.pipe_preset = QComboBox()
        self.pipe_preset.currentIndexChanged.connect(
            lambda _: self.refresh_pipeline_from_preset())
        self.pipe_preset.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        btn_psave = QPushButton("存方案")
        btn_psave.clicked.connect(self.main.save_batch_pipeline)
        btn_pdel = QPushButton("删方案")
        btn_pdel.clicked.connect(self.main.delete_batch_pipeline)
        preset_row = QHBoxLayout()
        preset_row.setSpacing(SPACE["sm"])
        preset_row.addWidget(self.pipe_preset, 1)
        preset_row.addWidget(btn_psave)
        preset_row.addWidget(btn_pdel)
        card_pre = SectionCard("方案", "把整套动作序列存下来，下次一键复用")
        card_pre.add(preset_row)
        lay.addWidget(card_pre)

        self.pipe_actions = QListWidget()
        self.pipe_actions.itemDoubleClicked.connect(
            lambda _: self._edit_pipeline_action())
        # right-click toggles the parameter lock (光影 SuperBatch parity)
        self.pipe_actions.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pipe_actions.customContextMenuRequested.connect(
            self._pipeline_action_menu)

        btn_up = QPushButton("上移")
        btn_up.clicked.connect(lambda: self._move_pipeline_action(-1))
        btn_down = QPushButton("下移")
        btn_down.clicked.connect(lambda: self._move_pipeline_action(1))
        btn_del = QPushButton("删除动作")
        btn_del.clicked.connect(self._remove_pipeline_action)
        nav_row = QHBoxLayout()
        nav_row.setSpacing(SPACE["sm"])
        nav_row.addWidget(btn_up)
        nav_row.addWidget(btn_down)
        nav_row.addWidget(btn_del)

        card_act = SectionCard("动作序列", "按从上到下的顺序执行；双击一行可改参数，右键一行可锁定参数")
        card_act.body.addWidget(self.pipe_actions, 1)
        # small minimum: the outer batch sub-tab has a fixed height and the list
        # is the only element that can give space back
        self.pipe_actions.setMinimumHeight(72)
        self.pipe_empty_label = hint_label(
            "还没有动作 —— 点下面的「添加动作」选自动打码 / 加水印 / 缩放 / 转格式 / 重命名")
        card_act.body.addWidget(self.pipe_empty_label)
        card_act.add(nav_row)
        lay.addWidget(card_act, 1)

        self.pipe_add_btn = QPushButton("添加动作")
        self.pipe_add_btn.clicked.connect(self._add_pipeline_action_menu)
        lay.addWidget(self.pipe_add_btn)

        self.pipeline_out = QLineEdit()
        self.pipeline_out.setPlaceholderText("输出目录")
        browse = QPushButton("浏览")
        browse.clicked.connect(lambda: self._browse(self.pipeline_out))
        out_row = QHBoxLayout()
        out_row.setSpacing(SPACE["sm"])
        out_row.addWidget(self.pipeline_out, 1)
        out_row.addWidget(browse)

        self.pipe_conflict = QComboBox()
        self.pipe_conflict.addItem("自动去重", "dedupe")
        self.pipe_conflict.addItem("直接覆盖", "overwrite")

        card_out = SectionCard("输出")
        card_out.add(out_row)
        card_out.add_row("重名文件", self.pipe_conflict,
                         note="自动去重 = 加序号不覆盖已有文件")
        lay.addWidget(card_out)

        self.pipeline_btn = QPushButton("开始执行流水线")
        set_variant(self.pipeline_btn, "primary")
        self.pipeline_btn.clicked.connect(self.main.run_batch_pipeline)
        lay.addWidget(self.pipeline_btn)

        self.refresh_pipeline_presets()
        self.refresh_pipeline_actions()
        return w

    def pipeline_actions_list(self) -> list:
        """Actions currently in the list widget (in order)."""
        return [self.pipe_actions.item(i).data(Qt.UserRole)
                for i in range(self.pipe_actions.count())]

    def refresh_pipeline_actions(self):
        """Reload the action list widget from the pending config list."""
        pending = getattr(self, "_pipe_pending", [])
        self.pipe_actions.clear()
        for i, act in enumerate(pending):
            item = QListWidgetItem(f"{i + 1}. {describe_pipeline_action(act)}")
            item.setData(Qt.UserRole, act)
            self.pipe_actions.addItem(item)
        if hasattr(self, "pipe_empty_label"):
            self.pipe_empty_label.setVisible(not pending)

    def refresh_pipeline_presets(self):
        cfg = self.main.config
        presets = cfg.get("batch_pipelines") or []
        current = self.pipe_preset.currentData()
        self.pipe_preset.blockSignals(True)
        self.pipe_preset.clear()
        self.pipe_preset.addItem("(选择方案)", None)
        for pr in presets:
            n = len(pr.get("actions", []))
            self.pipe_preset.addItem(f"{pr['name']} ({n}动作)", pr["name"])
        if current is not None:
            ix = self.pipe_preset.findData(current)
            if ix >= 0:
                self.pipe_preset.setCurrentIndex(ix)
        self.pipe_preset.blockSignals(False)

    def _add_pipeline_action_menu(self):
        menu = QMenu(self)
        for key, label in self.PIPELINE_ACTION_TYPES:
            act = menu.addAction(label)
            act.setData(key)
        chosen = menu.exec(self.pipe_add_btn.mapToGlobal(
            self.pipe_add_btn.rect().bottomLeft()))
        if chosen is None:
            return
        self._open_action_dialog(str(chosen.data()), None)

    def _open_action_dialog(self, action_type, existing):
        dlg = ActionConfigDialog(action_type, existing, self.main, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        params = dlg.result_params()
        if not hasattr(self, "_pipe_pending"):
            self._pipe_pending = []
        if existing is not None:
            idx = self.pipe_actions.currentRow()
            if 0 <= idx < len(self._pipe_pending):
                self._pipe_pending[idx] = {"type": action_type,
                                           "params": params}
        else:
            self._pipe_pending.append({"type": action_type, "params": params})
        self.refresh_pipeline_actions()

    def _edit_pipeline_action(self):
        idx = self.pipe_actions.currentRow()
        if not (0 <= idx < len(getattr(self, "_pipe_pending", []))):
            return
        act = self._pipe_pending[idx]
        if act.get("locked"):
            QMessageBox.information(
                self, "动作已锁定",
                "这个动作的参数已锁定，防止误改。\n在列表里右键它并选「解锁动作」后再双击编辑。")
            return
        self._open_action_dialog(act["type"], act.get("params", {}))

    def _pipeline_action_menu(self, pos):
        """Right-click menu on the action list: lock / unlock parameters."""
        item = self.pipe_actions.itemAt(pos)
        if item is None:
            return
        idx = self.pipe_actions.row(item)
        if not (0 <= idx < len(getattr(self, "_pipe_pending", []))):
            return
        act = self._pipe_pending[idx]
        menu = QMenu(self)
        if act.get("locked"):
            menu.addAction("解锁动作").triggered.connect(
                lambda: self._set_pipeline_lock(idx, False))
        else:
            menu.addAction("锁定动作").triggered.connect(
                lambda: self._set_pipeline_lock(idx, True))
        menu.exec(self.pipe_actions.mapToGlobal(pos))

    def _set_pipeline_lock(self, idx: int, locked: bool):
        if not (0 <= idx < len(getattr(self, "_pipe_pending", []))):
            return
        if locked:
            self._pipe_pending[idx]["locked"] = True
        else:
            self._pipe_pending[idx].pop("locked", None)
        self.refresh_pipeline_actions()
        self.pipe_actions.setCurrentRow(idx)

    def _move_pipeline_action(self, delta):
        idx = self.pipe_actions.currentRow()
        if not (0 <= idx < len(getattr(self, "_pipe_pending", []))):
            return
        j = idx + delta
        if not (0 <= j < len(self._pipe_pending)):
            return
        lst = self._pipe_pending
        lst[idx], lst[j] = lst[j], lst[idx]
        self.refresh_pipeline_actions()
        self.pipe_actions.setCurrentRow(j)

    def _remove_pipeline_action(self):
        idx = self.pipe_actions.currentRow()
        if not (0 <= idx < len(getattr(self, "_pipe_pending", []))):
            return
        del self._pipe_pending[idx]
        self.refresh_pipeline_actions()

    def pipeline_config(self) -> dict:
        """Full config for running the pipeline."""
        if not hasattr(self, "_pipe_pending"):
            self._pipe_pending = []
        return {
            "actions": list(self._pipe_pending),
            "out_dir": self.pipeline_out.text().strip(),
            "on_conflict": self.pipe_conflict.currentData() or "dedupe",
        }

    def refresh_pipeline_from_preset(self):
        """Load the selected preset into the pending action list."""
        name = self.pipe_preset.currentData()
        if not name:
            return
        for pr in (self.main.config.get("batch_pipelines") or []):
            if pr["name"] == name:
                self._pipe_pending = [dict(a) for a in pr.get("actions", [])]
                self.pipeline_out.setText(pr.get("out_dir", ""))
                ix = self.pipe_conflict.findData(
                    pr.get("on_conflict", "dedupe"))
                self.pipe_conflict.setCurrentIndex(ix if ix >= 0 else 0)
                self.refresh_pipeline_actions()
                return

    def _browse(self, edit):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", edit.text())
        if path:
            edit.setText(path)

    # ---------------- helpers ----------------
    def _update_auto_preview(self):
        if not hasattr(self.main, "canvas"):
            return
        # Push the panel's values even before an image is loaded (v1.14.5): the
        # panel is the source of truth, and the old guard skipped the push until
        # an image arrived, so the canvas kept its own default strength instead
        # of the user's saved one. Only the re-render needs an image.
        self.main.canvas.auto_params = self.auto_settings()
        if self.main.canvas.auto_mask is not None:
            self.main.canvas._rebuild()
            self.main.canvas.state_changed.emit()

    def auto_settings(self) -> dict:
        return {
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "classes": [k for k, cb in self.class_checks.items() if cb.isChecked()],
            "auto_effect": self.auto_effect_combo.currentData() or "blur",
            "auto_strength": self.auto_strength_spin.value(),
            "auto_mask_dilate": self.mask_dilate_spin.value(),
            "input_size": self.input_size_combo.currentData() or 1024,
        }

    def _apply_auto_defaults(self):
        """Read auto-censor defaults from config and apply to the auto tab."""
        cfg = self.main.config
        self.conf_spin.setValue(float(cfg.get("conf_threshold", 0.3)))
        self.iou_spin.setValue(float(cfg.get("iou_threshold", 0.7)))
        effect = normalize_effect(cfg.get("auto_effect", "blur"))
        idx = self.auto_effect_combo.findData(effect)
        if idx >= 0:
            self.auto_effect_combo.setCurrentIndex(idx)
        try:
            self.auto_strength_spin.setValue(int(cfg.get("auto_strength", 15)))
        except (TypeError, ValueError):
            pass
        try:
            self.mask_dilate_spin.setValue(int(cfg.get("auto_mask_dilate", 5)))
        except (TypeError, ValueError):
            pass
        idx = self.input_size_combo.findData(int(cfg.get("detect_size", 1024)))
        if idx >= 0:
            self.input_size_combo.setCurrentIndex(idx)
        # v1.14.5: the class checkboxes were never restored, so a saved
        # selection was silently ignored on every start
        saved = cfg.get("selected_regions")
        if saved:
            for key, cb in self.class_checks.items():
                cb.setChecked(key in saved)