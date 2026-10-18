# -*- coding: utf-8 -*-
"""Mini window for quick drag-drop censoring.

v1.15.0: runs the SAME core compose chain as the main window
(``compose_image`` + ``MaskSelection``) and reads the SAME config keys the
main window's 自动打码 page uses (效果 / 强度 / 覆盖扩展 / 置信度 / IoU /
类别 / 检测分辨率). It used to run the separate processor.py pipeline with a
private mosaic-type combo, so results differed from the main window; that
old pipeline is gone."""
from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..core.detector import ONNXDetector
from ..core.image_ops import compose_image, load_original, load_resized
from ..core.selection import MaskSelection, normalize_effect
from .clipboard import copy_image_to_clipboard
from .theme import SPACE
from .widgets import SectionCard


class MiniWindow(QWidget):
    def __init__(self, detector: ONNXDetector, config: dict):
        super().__init__()
        self.detector = detector
        self.config = config
        self.setWindowTitle("Automatic Coding 2 - 迷你模式")
        self.setMinimumSize(360, 240)
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(SPACE["md"])
        self.drop_label = QLabel("拖放图片到此处\n\n支持：本地图片 / 图片链接 / Base64 / HTML 内嵌图片")
        self.drop_label.setAlignment(Qt.AlignCenter)
        # styled by the app-wide #dropZone rule so it follows the light/dark
        # theme instead of a hardcoded grey
        self.drop_label.setObjectName("dropZone")
        layout.addWidget(self.drop_label, 1)

        card = SectionCard(
            "打码参数",
            "与主窗口「自动打码」页共用同一套参数：效果 / 强度 / 覆盖扩展 /\n"
            "置信度 / IoU / 检测类别 / 检测分辨率，改了主窗口设置立即生效。")
        layout.addWidget(card)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("hint")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() or md.hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            for url in md.urls():
                path = url.toLocalFile()
                if path:
                    self.process_path(path)
                else:
                    u = url.toString()
                    if u.startswith(("http://", "https://")):
                        self.download_and_process(u)
            event.acceptProposedAction()
            return

        if md.hasText():
            text = md.text().strip()
            if text.startswith("data:image"):
                self.process_base64(text)
            elif text.startswith(("http://", "https://")):
                self.download_and_process(text)
            else:
                self.process_html_text(text)
            event.acceptProposedAction()
            return

        event.ignore()

    def _compose_censored(self, path: str):
        """Detect + censor one image with the main window's auto settings.

        Detection runs on the SAME input the main window uses: the edit-size
        image (``load_resized`` caps the long edge at 2048), so a large photo
        yields identical detections in both windows. The compose still
        happens at the FULL original resolution -- ``MaskSelection.to_mask``
        upscales the detection mask to the compose size, which is exactly
        how the main window's export path reconciles the two resolutions.

        Returns (processed_PIL, detection_count) or (None, 0) when nothing
        was detected; raises on load/detect errors (caller reports them)."""
        cfg = self.config
        edit_pil, _, _ = load_resized(path)
        rgb = np.array(edit_pil.convert("RGB"))
        orig = load_original(path)

        dets = self.detector.detect(
            rgb[:, :, ::-1],  # detector expects BGR
            conf_threshold=float(cfg.get("conf_threshold", 0.3)),
            iou_threshold=float(cfg.get("iou_threshold", 0.7)),
            allowed_classes=cfg.get("selected_regions") or [],
            input_size=int(cfg.get("detect_size", 1024)),
        )
        union = np.zeros(rgb.shape[:2], dtype=bool)
        for d in dets or []:
            union |= d.mask
        if not union.any():
            return None, 0
        effect = normalize_effect(cfg.get("auto_effect", "blur"))
        strength = int(cfg.get("auto_strength", 15))
        sel = MaskSelection(union, effect=effect, strength=strength)
        out = compose_image(
            orig, [], [sel], [], {},
            params=cfg.get("params", {}),
            auto_params={
                "auto_effect": effect,
                "auto_strength": strength,
                "auto_mask_dilate": int(cfg.get("auto_mask_dilate", 5)),
            },
        )
        return out, len(dets or [])

    def process_path(self, path: str):
        if not Path(path).is_file():
            self.status_label.setText(f"文件不存在: {path}")
            return
        self.status_label.setText("正在处理...")
        try:
            processed, _n = self._compose_censored(path)
            if processed is None:
                self.status_label.setText("未检测到需要打码的区域。")
                return
            if copy_image_to_clipboard(processed):
                self.status_label.setText("已打码并复制到剪贴板！")
            else:
                self._save_temp_and_report(processed, path)
        except Exception as e:
            self.status_label.setText(f"处理失败: {e}")

    def _save_temp_and_report(self, pil_img: Image.Image, source_path: str):
        try:
            suffix = Path(source_path).suffix or ".png"
            tmp_dir = Path(tempfile.gettempdir()) / "AutomaticCoding2_mini"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out = tmp_dir / (Path(source_path).stem + "_processed" + suffix)
            if suffix.lower() in (".jpg", ".jpeg"):
                pil_img.convert("RGB").save(out, quality=100)
            else:
                pil_img.save(out)
            self.status_label.setText(f"已保存到: {out}")
        except Exception as e:
            self.status_label.setText(f"处理完成，但复制/保存失败: {e}")

    def download_and_process(self, url: str):
        self.status_label.setText("正在下载图片...")
        try:
            import requests

            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            if "image" not in resp.headers.get("Content-Type", ""):
                self.status_label.setText("链接内容似乎不是图片。")
                return
            tmp = Path(tempfile.gettempdir()) / "AutomaticCoding2_mini"
            tmp.mkdir(parents=True, exist_ok=True)
            ext = ".png"
            ctype = resp.headers.get("Content-Type", "")
            if "jpeg" in ctype or "jpg" in ctype:
                ext = ".jpg"
            elif "webp" in ctype:
                ext = ".webp"
            elif "gif" in ctype:
                ext = ".gif"
            path = tmp / ("downloaded" + ext)
            path.write_bytes(resp.content)
            self.process_path(str(path))
        except Exception as e:
            self.status_label.setText(f"下载图片错误: {e}")

    def process_base64(self, data_uri: str):
        self.status_label.setText("处理内嵌图片...")
        try:
            header, encoded = data_uri.split(",", 1)
            raw = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(raw))
            tmp = Path(tempfile.gettempdir()) / "AutomaticCoding2_mini"
            tmp.mkdir(parents=True, exist_ok=True)
            path = tmp / "base64_image.png"
            img.save(path)
            self.process_path(str(path))
        except Exception as e:
            self.status_label.setText(f"处理Base64图片错误: {e}")

    def process_html_text(self, text: str):
        self.status_label.setText("解析HTML拖放数据...")
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            img = soup.find("img")
            src = img.get("src") if img else None
            if not src:
                self.status_label.setText("无法识别拖放内容。")
                return
            if src.startswith("data:image"):
                self.process_base64(src)
            elif src.startswith(("http://", "https://")):
                self.download_and_process(src)
            else:
                self.status_label.setText("无法识别的图片链接。")
        except Exception as e:
            self.status_label.setText(f"解析HTML拖放数据时出错: {e}")
