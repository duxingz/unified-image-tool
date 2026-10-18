# -*- coding: utf-8 -*-
"""Unified save/batch export dialog.

v1.14.0 restyle: the controls are grouped into titled cards instead of one
flat QFormLayout, and the "prefix / start / digits" and "percent / width /
height" rows (previously crammed into single continuation rows with empty
labels) each get their own labelled row so they cannot be misread.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .theme import SPACE
from .widgets import SectionCard, hint_label, set_variant


class SaveDialog(QDialog):
    def __init__(self, parent=None, default_out="", default_prefix="IMG",
                 default_fmt="PNG", default_strip_meta="strip", last_rename=None,
                 last_export=None):
        super().__init__(parent)
        self.setWindowTitle("保存 / 批量导出")
        # height from the measured sizeHint (628) plus a little slack: the
        # dialog has no scroll area, so a short window would clip the buttons
        self.resize(470, 660)

        last_exp = last_export or {}
        last = last_rename or {}
        cards = QVBoxLayout()
        cards.setSpacing(SPACE["md"])

        # ---------------- 输出 ----------------
        self.out_edit = QLineEdit(str(last_exp.get("out_dir") or default_out))
        self.out_edit.setPlaceholderText("选择输出文件夹")
        btn_out = QPushButton("浏览")
        btn_out.clicked.connect(self._browse_out)
        out_row = QHBoxLayout()
        out_row.setSpacing(SPACE["sm"])
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(btn_out)
        card_out = SectionCard("输出目录")
        card_out.add(out_row)
        cards.addWidget(card_out)

        # ---------------- 命名 ----------------
        self.name_combo = QComboBox()
        self.name_combo.addItem("保留原文件名", "preserve")
        self.name_combo.addItem("前缀+编号", "rename")
        self.name_combo.addItem("按当前显示名", "display")
        self.name_combo.currentIndexChanged.connect(self._update_name_fields)
        mode = last.get("name_mode")
        if mode in ("preserve", "rename", "display"):
            self.name_combo.setCurrentIndex(
                {"preserve": 0, "rename": 1, "display": 2}[mode])

        self.prefix_edit = QLineEdit(str(last.get("prefix", default_prefix)))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 99999)
        try:
            self.start_spin.setValue(int(last.get("start", 1)))
        except (TypeError, ValueError):
            self.start_spin.setValue(1)
        self.digits_spin = QSpinBox()
        self.digits_spin.setRange(1, 10)
        try:
            self.digits_spin.setValue(int(last.get("digits", 3)))
        except (TypeError, ValueError):
            self.digits_spin.setValue(3)

        card_name = SectionCard("命名方式")
        card_name.add_row("方式", self.name_combo)
        card_name.add_row("前缀", self.prefix_edit)
        card_name.add_row("起始编号", self.start_spin)
        card_name.add_row("编号位数", self.digits_spin)
        self._name_fields = [self.prefix_edit, self.start_spin, self.digits_spin]
        cards.addWidget(card_name)

        # ---------------- 格式 ----------------
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["PNG", "JPG", "WEBP", "BMP", "GIF"])
        fmt = str(last_exp.get("fmt") or default_fmt)
        self.fmt_combo.setCurrentText(
            fmt if fmt in ["PNG", "JPG", "WEBP", "BMP", "GIF"] else "PNG")
        self.fmt_combo.currentTextChanged.connect(self._update_quality)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(30, 100)
        self.quality_spin.setValue(100)

        card_fmt = SectionCard("格式与质量")
        card_fmt.add_row("格式", self.fmt_combo)
        card_fmt.add_row("质量", self.quality_spin,
                         note="仅 JPG / WEBP 有效；100 = 最高质量")
        cards.addWidget(card_fmt)

        # ---------------- 缩放 ----------------
        self.scale_combo = QComboBox()
        self.scale_combo.addItem("不缩放", "none")
        self.scale_combo.addItem("按百分比", "percent")
        self.scale_combo.addItem("按固定尺寸", "size")
        self.scale_combo.currentIndexChanged.connect(self._update_scale_fields)
        self.percent_spin = QDoubleSpinBox()
        self.percent_spin.setRange(1, 1000)
        self.percent_spin.setValue(100)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(1920)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(1080)

        card_scale = SectionCard("缩放")
        card_scale.add_row("方式", self.scale_combo)
        card_scale.add_row("百分比", self.percent_spin)
        card_scale.add_row("宽度", self.width_spin)
        card_scale.add_row("高度", self.height_spin)
        self._scale_fields = [self.percent_spin, self.width_spin, self.height_spin]
        cards.addWidget(card_scale)

        # remember last used values (override the settings defaults)
        try:
            self.quality_spin.setValue(int(last_exp.get("quality",
                                       self.quality_spin.value())))
        except (TypeError, ValueError):
            pass
        scale = last_exp.get("scale_mode")
        if scale in ("none", "percent", "size"):
            self.scale_combo.setCurrentText({
                "none": "不缩放", "percent": "按百分比", "size": "按固定尺寸",
            }[scale])
        try:
            self.percent_spin.setValue(float(last_exp.get("percent", 100)))
        except (TypeError, ValueError):
            pass
        try:
            self.width_spin.setValue(int(last_exp.get("width", 1920)))
            self.height_spin.setValue(int(last_exp.get("height", 1080)))
        except (TypeError, ValueError):
            pass

        # ---------------- 元数据 ----------------
        self.strip_meta_check = QCheckBox("去除元数据（EXIF 等）")
        strip = last_exp.get("strip_metadata")
        if strip is None:
            self.strip_meta_check.setChecked(default_strip_meta == "strip")
        else:
            self.strip_meta_check.setChecked(bool(strip))
        card_meta = SectionCard("元数据")
        card_meta.body.addWidget(self.strip_meta_check)
        card_meta.body.addWidget(hint_label(
            "勾选后不写入 EXIF/GPS；导出的新文件也不会带方向标记"))
        cards.addWidget(card_meta)

        # ---------------- 按钮 ----------------
        ok = QPushButton("确认并导出")
        set_variant(ok, "primary")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.setSpacing(SPACE["md"])
        layout.addLayout(cards)
        layout.addStretch()
        layout.addLayout(btns)

        self._update_name_fields()
        self._update_scale_fields()
        self._update_quality()

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.out_edit.text())
        if path:
            self.out_edit.setText(path)

    def _update_name_fields(self):
        enabled = self.name_combo.currentData() == "rename"
        for w in self._name_fields:
            w.setEnabled(enabled)

    def _update_scale_fields(self):
        mode = self.scale_combo.currentData()
        self.percent_spin.setEnabled(mode == "percent")
        self.width_spin.setEnabled(mode == "size")
        self.height_spin.setEnabled(mode == "size")

    def _update_quality(self):
        fmt = self.fmt_combo.currentText()
        self.quality_spin.setEnabled(fmt in ("JPG", "WEBP"))

    def result_config(self) -> dict:
        return {
            "out_dir": self.out_edit.text().strip(),
            "name_mode": self.name_combo.currentData(),
            "prefix": self.prefix_edit.text().strip(),
            "start": self.start_spin.value(),
            "digits": self.digits_spin.value(),
            "fmt": self.fmt_combo.currentText(),
            "quality": self.quality_spin.value(),
            "scale_mode": self.scale_combo.currentData(),
            "percent": self.percent_spin.value(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "strip_metadata": self.strip_meta_check.isChecked(),
        }
