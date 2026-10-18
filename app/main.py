# -*- coding: utf-8 -*-
"""Unified Image Tool entry point."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make project root importable when running `python -m app.main`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt, QLockFile
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

from app.core.config import CONFIG_DIR, load_config, save_config
from app.core.detector import ONNXDetector, pick_providers
from app.core.paths import resource_path
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme
from app.version import APP_NAME, __version__


def acquire_single_instance_lock(config_dir) -> QLockFile | None:
    """Prevent a second instance (two windows would race on config.json and
    the shared settings). Returns the lock the caller MUST hold for the
    whole process lifetime, or None when another instance is running.

    QLockFile detects stale locks from crashed processes (PID check), so a
    hard kill never locks the user out permanently."""
    lock = QLockFile(str(Path(config_dir) / "app.lock"))
    if not lock.tryLock(0):
        return None
    return lock


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("UnifiedImageTool")
    icon_path = resource_path("app/assets/icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # single instance: must run before anything touches config or spawns UI
    lock = acquire_single_instance_lock(CONFIG_DIR)
    if lock is None:
        QMessageBox.information(
            None, "已经在运行",
            "自动打码工具已经在运行了。\n请使用已打开的那个窗口。")
        return 0
    # keep the lock alive for the whole process (releasing it would allow
    # a second instance)
    app._single_instance_lock = lock

    # 模型加载状态提示
    splash_pix = QPixmap(520, 220)
    splash_pix.fill(QColor(35, 35, 40))
    painter = QPainter(splash_pix)
    painter.setPen(QColor(255, 255, 255))
    painter.setFont(QFont("Microsoft YaHei", 16))
    painter.drawText(splash_pix.rect(), Qt.AlignCenter, f"{APP_NAME} v{__version__}\n模型加载中...")
    painter.end()
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()

    config = load_config()
    apply_theme(app, config.get("theme", "xiaodu"))

    # Resolve model path; fallback to first available ONNX in models/
    model_path = config.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        models_dir = resource_path("models")
        candidates = sorted(models_dir.glob("*.onnx"))
        if candidates:
            model_path = str(candidates[0])
            config["model_path"] = model_path
        else:
            QMessageBox.critical(
                None, "模型缺失",
                f"未找到 ONNX 模型文件，请放到:\n{models_dir}",
            )
            return 1

    try:
        providers = pick_providers(config.get("use_gpu", True))
        detector = ONNXDetector(model_path, providers=providers)
    except Exception as e:
        splash.close()
        QMessageBox.critical(None, "模型加载失败", str(e))
        return 1

    splash.close()
    window = MainWindow(detector, config)
    if "Dml" in detector.session.get_providers()[0]:
        window.statusBar().showMessage("就绪（DirectML GPU 加速）")
    window.show()

    exit_code = app.exec()
    save_config(config)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
