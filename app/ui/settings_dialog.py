# -*- coding: utf-8 -*-
"""Settings dialog: model, auto-censor defaults, save defaults."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..core.config import default_config, save_config
from ..core.detector import DETECT_SIZES
from ..core.paths import resource_path
from ..core.selection import AUTO_EFFECT_CHOICES, normalize_effect
from .theme import SPACE
from .widgets import SectionCard, set_variant


class SettingsDialog(QDialog):
    clearListRequested = Signal()
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        # measured minimum width is ~518 (longest label + its control), so ask
        # for that instead of a number the layout would silently override
        self.resize(520, 660)
        self.config = config

        cards = QVBoxLayout()
        cards.setSpacing(SPACE["md"])

        # ---------------- 检测模型 ----------------
        self.model_combo = QComboBox()
        models_dir = resource_path("models")
        if models_dir.is_dir():
            for f in sorted(models_dir.glob("*.onnx")):
                self.model_combo.addItem(f.name, str(f))
        current = Path(config.get("model_path", "")).resolve()
        for i in range(self.model_combo.count()):
            if Path(self.model_combo.itemData(i)).resolve() == current:
                self.model_combo.setCurrentIndex(i)
                break
        btn_scan = QPushButton("重新扫描")
        btn_scan.clicked.connect(self._scan_models)
        model_row = QHBoxLayout()
        model_row.setSpacing(SPACE["sm"])
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(btn_scan)
        card_model = SectionCard("检测模型", "models 文件夹里的 .onnx 会自动被发现")
        card_model.add(model_row)
        cards.addWidget(card_model)

        # ---------------- 自动打码默认值 ----------------
        self.auto_effect_combo = QComboBox()
        for eff, label in AUTO_EFFECT_CHOICES:
            self.auto_effect_combo.addItem(label, eff)
        effect = normalize_effect(config.get("auto_effect", "blur"))
        idx = self.auto_effect_combo.findData(effect)
        if idx >= 0:
            self.auto_effect_combo.setCurrentIndex(idx)

        self.auto_strength = QSpinBox()
        self.auto_strength.setRange(1, 100)
        try:
            self.auto_strength.setValue(int(config.get("auto_strength", 15)))
        except (TypeError, ValueError):
            self.auto_strength.setValue(15)

        self.auto_dilate_spin = QSpinBox()
        self.auto_dilate_spin.setRange(0, 50)
        self.auto_dilate_spin.setValue(int(config.get("auto_mask_dilate", 5)))

        self.detect_size_combo = QComboBox()
        for size in DETECT_SIZES:
            self.detect_size_combo.addItem(f"{size}×{size}", size)
        idx = self.detect_size_combo.findData(int(config.get("detect_size", 1024)))
        self.detect_size_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.device_combo = QComboBox()
        self.device_combo.addItem("自动（优先 DirectML GPU）", "gpu")
        self.device_combo.addItem("仅使用 CPU", "cpu")
        self.device_combo.setCurrentIndex(
            0 if config.get("use_gpu", True) else 1
        )

        self.mask_overlay_check = QCheckBox("显示检测区域（红色蒙层）")
        self.mask_overlay_check.setChecked(bool(config.get("show_mask_overlay", True)))

        card_auto = SectionCard(
            "自动打码默认值", "这些是工具面板「自动打码」页的初值")
        card_auto.add_row("自动打码效果", self.auto_effect_combo)
        card_auto.add_row("打码强度", self.auto_strength)
        card_auto.add_row("覆盖扩展（像素）", self.auto_dilate_spin,
                          note="掩码向外扩充的像素数，防止盖不住目标")
        card_auto.add_row("检测分辨率", self.detect_size_combo)
        card_auto.add_row("推理设备", self.device_combo,
                          note="切换后会自动重建检测器；无 GPU 时自动回退 CPU")
        card_auto.body.addWidget(self.mask_overlay_check)
        cards.addWidget(card_auto)

        # ---------------- 保存默认值 ----------------
        self.default_fmt_combo = QComboBox()
        self.default_fmt_combo.addItems(["PNG", "JPG"])
        self.default_fmt_combo.setCurrentText(config.get("default_save_fmt", "PNG"))

        self.default_strip_meta = QComboBox()
        self.default_strip_meta.addItem("保留元数据", "preserve")
        self.default_strip_meta.addItem("去除元数据", "strip")
        self.default_strip_meta.setCurrentIndex(
            0 if config.get("default_strip_meta", "strip") == "preserve" else 1
        )

        card_save = SectionCard("导出默认值")
        card_save.add_row("默认保存格式", self.default_fmt_combo)
        card_save.add_row("默认元数据处理", self.default_strip_meta)
        cards.addWidget(card_save)

        # ---------------- 界面 ----------------
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("小独爱心（暖白）", "xiaodu")
        self.theme_combo.addItem("小独夜（暖黑）", "xiaodunight")
        # legacy values (light/dark/persona) resolve via findData miss ->
        # index 0 (小独爱心), and apply_theme aliases them on application
        self.theme_combo.setCurrentIndex(
            max(0, self.theme_combo.findData(config.get("theme", "xiaodu")))
        )
        self.dbl_click_fit_check = QCheckBox("画布双击空白处适配视图")
        self.dbl_click_fit_check.setChecked(bool(config.get("double_click_fit", True)))
        self.auto_open_check = QCheckBox("导出完成后自动打开输出文件夹")
        self.auto_open_check.setChecked(
            bool(config.get("auto_open_export_folder", False)))

        card_ui = SectionCard("界面")
        card_ui.add_row("主题", self.theme_combo)
        card_ui.body.addWidget(self.dbl_click_fit_check)
        card_ui.body.addWidget(self.auto_open_check)
        cards.addWidget(card_ui)

        # ---------------- 按钮 ----------------
        btn_clear_list = QPushButton("清空当前列表")
        btn_clear_list.clicked.connect(self.clearListRequested.emit)
        btn_clear_list.setToolTip("清空左侧图片列表（不影响磁盘文件）")
        restore = QPushButton("恢复默认设置")
        restore.clicked.connect(self._restore_defaults)
        ok = QPushButton("确定")
        set_variant(ok, "primary")
        ok.clicked.connect(self._apply)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addWidget(btn_clear_list)
        btns.addWidget(restore)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.setSpacing(SPACE["md"])
        layout.addLayout(cards)
        layout.addStretch()
        layout.addLayout(btns)

    def _scan_models(self):
        models_dir = resource_path("models")
        self.model_combo.clear()
        if models_dir.is_dir():
            for f in sorted(models_dir.glob("*.onnx")):
                self.model_combo.addItem(f.name, str(f))
        self.model_combo.setCurrentIndex(0)

    def _restore_defaults(self):
        ret = QMessageBox.question(
            self, "恢复默认设置",
            "恢复为出厂默认设置？\n（置信度 0.3 / IoU 0.7 / 强度 15 / 覆盖扩展 5 / 默认小独爱心主题等）",
        )
        if ret != QMessageBox.Yes:
            return
        # 只重置对话框控件、不立即写盘：之后点「取消」必须无任何副作用。
        # 落盘发生在「确定」(_apply / open_settings) 里。
        self.config = default_config()
        self._scan_models()
        self.auto_effect_combo.setCurrentIndex(0)
        self.auto_strength.setValue(15)
        self.auto_dilate_spin.setValue(5)
        self.mask_overlay_check.setChecked(True)
        idx = self.detect_size_combo.findData(1024)
        self.detect_size_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.device_combo.setCurrentIndex(0)
        self.default_fmt_combo.setCurrentText("PNG")
        self.default_strip_meta.setCurrentIndex(1)
        self.theme_combo.setCurrentIndex(0)
        self.dbl_click_fit_check.setChecked(True)
        self.auto_open_check.setChecked(False)
        QMessageBox.information(
            self, "恢复默认设置",
            "已恢复出厂默认设置，点击「确定」后生效。\n"
            "（工具面板中的置信度 / IoU / 强度 / 覆盖扩展也会同步为新默认值）",
        )

    def _apply(self):
        if self.model_combo.currentData():
            self.config["model_path"] = self.model_combo.currentData()
        self.config["auto_effect"] = self.auto_effect_combo.currentData() or "blur"
        self.config["auto_strength"] = self.auto_strength.value()
        self.config["auto_mask_dilate"] = self.auto_dilate_spin.value()
        self.config["show_mask_overlay"] = self.mask_overlay_check.isChecked()
        self.config["detect_size"] = self.detect_size_combo.currentData() or 1024
        self.config["use_gpu"] = self.device_combo.currentData() == "gpu"
        self.config["default_save_fmt"] = self.default_fmt_combo.currentText()
        self.config["default_strip_meta"] = self.default_strip_meta.currentData()
        self.config["theme"] = self.theme_combo.currentData() or "xiaodu"
        self.config["double_click_fit"] = self.dbl_click_fit_check.isChecked()
        self.config["auto_open_export_folder"] = self.auto_open_check.isChecked()
        save_config(self.config)
        self.accept()