# -*- coding: utf-8 -*-
"""Stress test: import 300 images at once -> batch auto-censor all -> export all.

Runs the real MainWindow offscreen (QT_QPA_PLATFORM=offscreen). Modal
QMessageBoxes are no-op'd so the headless run cannot block on user input.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from app.core.config import default_config
from app.core.detector import ONNXDetector, pick_providers
from app.version import __version__

COUNT = 300
W, H = 512, 768
DETECT_SIZE = 1024

# --- disable modal dialogs for headless run -------------------------------
from PySide6.QtWidgets import QMessageBox

for name in ("information", "warning", "critical"):
    setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))

from PySide6.QtWidgets import QApplication  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


def mem_mb() -> float:
    if psutil is not None:
        return psutil.Process().memory_info().rss / 1048576
    return float("nan")


def main() -> int:
    rng = np.random.default_rng(7)
    src = Path(tempfile.mkdtemp(prefix="uit_stress_"))
    out = src / "导出"
    print(f"[环境] Python {sys.version.split()[0]} | app v{__version__} | 图片 {COUNT} 张 {W}x{H} | detect_size={DETECT_SIZE}")

    print("[准备] 生成测试图片 ...")
    t0 = time.perf_counter()
    paths: list[str] = []
    for i in range(COUNT):
        arr = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        p = src / f"img_{i:04d}.jpg"
        Image.fromarray(arr).save(p, quality=85)
        paths.append(str(p))
    print(f"       生成耗时 {time.perf_counter() - t0:.1f}s")

    app = QApplication([])
    cfg = default_config()
    cfg["use_gpu"] = True
    cfg["detect_size"] = DETECT_SIZE

    print("[环境] 加载模型 ...")
    t0 = time.perf_counter()
    detector = ONNXDetector(cfg["model_path"], providers=pick_providers(True))
    provider = detector.session.get_providers()[0]
    print(f"       模型加载 {time.perf_counter() - t0:.1f}s | provider: {provider}")
    if not provider.startswith("Cpu") and "Dml" not in provider:
        provider = detector.session.get_providers()[0]

    window = None
    from app.ui.main_window import MainWindow  # noqa: E402
    window = MainWindow(detector, cfg)

    # --- 1) 同时拉取 300 张 ------------------------------------------------
    print("[阶段1] 一次导入 300 张（含缩略图生成）...")
    t0 = time.perf_counter()
    window._add_paths(paths)
    t_import = time.perf_counter() - t0
    loaded = window.thumbnail_list.count()
    thumb_failed = sum(
        1 for i in range(loaded)
        if "无法加载" in (window.thumbnail_list.item(i).text() or "")
    )
    print(f"       导入 {loaded} 张 | 耗时 {t_import:.1f}s | 缩略图失败 {thumb_failed} | 内存 {mem_mb():.0f}MB")

    # --- 2) 批量自动打码 300 张 --------------------------------------------
    print("[阶段2] 批量自动打码 300 张（后台线程）...")
    t0 = time.perf_counter()
    window.batch_auto_censor()
    deadline = time.monotonic() + 900
    while window.auto_worker and window.auto_worker.isRunning():
        app.processEvents()
        if time.monotonic() > deadline:
            print("!! 超时：批量打码 900s 未完成，退出")
            return 1
        time.sleep(0.05)
    t_detect = time.perf_counter() - t0

    detected = sum(
        1 for st in window.image_states.values()
        if st.get("auto_mask") is not None and st["auto_mask"].any()
    )
    states_ok = len(window.image_states)
    print(f"       耗时 {t_detect:.1f}s ({t_detect / COUNT * 1000:.0f}ms/张) | "
          f"检出区域 {detected}/{states_ok} | 内存 {mem_mb():.0f}MB")

    # --- 3) 全部导出 -------------------------------------------------------
    print("[阶段3] 批量导出 300 张 ...")
    save_cfg = {
        "out_dir": str(out),
        "name_mode": "preserve",
        "prefix": "OUT",
        "start": 1,
        "digits": 3,
        "fmt": "PNG",
        "quality": 95,
        "scale_mode": "none",
        "percent": 100,
        "width": 1000,
        "height": 1000,
        "strip_metadata": True,
    }
    from app.ui.main_window import BatchExportWorker  # noqa: E402
    result_holder = {}

    worker = BatchExportWorker(
        window.thumbnail_list.item_paths(),
        window.image_states,
        window.config,
        window.tool_panel.auto_settings(),
        save_cfg,
        window.sticker_library,
        window,
    )
    worker.done.connect(lambda r: result_holder.update(r))
    t0 = time.perf_counter()
    worker.start()
    deadline = time.monotonic() + 900
    while worker.isRunning():
        app.processEvents()
        if time.monotonic() > deadline:
            print("!! 超时：导出 900s 未完成，退出")
            return 1
        time.sleep(0.05)
    # drain the queued cross-thread delivery of the done signal
    while not result_holder:
        app.processEvents()
        if time.monotonic() > deadline:
            print("!! 导出结果信号未收到")
            return 1
        time.sleep(0.02)
    t_export = time.perf_counter() - t0
    result = result_holder
    print(f"       耗时 {t_export:.1f}s ({t_export / COUNT * 1000:.0f}ms/张)")

    # --- 核验 ---------------------------------------------------------------
    print("[核验] 导出产物 ...")
    files = sorted(out.glob("*.png")) if out.is_dir() else []
    ok = result.get("processed", -1) == COUNT and result.get("failed", -1) == 0
    size_ok = len(files) == COUNT
    spot = []
    for f in files[:3]:
        try:
            with Image.open(f) as im:
                spot.append((f.name, im.size, im.format))
        except Exception as e:
            spot.append((f.name, "损坏", str(e)))
    print(f"       worker 统计: 成功 {result.get('processed')} / 失败 {result.get('failed')} / 总数 {result.get('total')}")
    print(f"       输出文件数 {len(files)} | 抽查 {spot}")

    print("=" * 60)
    print(f"结论: 300 张同时导入 ✓ | 批量打码 √{states_ok}/300 | 导出 √{result.get('processed')}/300 → {'通过' if ok and size_ok else '失败'}")
    print(f"耗时汇总: 导入 {t_import:.1f}s | 打码 {t_detect:.1f}s | 导出 {t_export:.1f}s | 峰值内存约 {mem_mb():.0f}MB")
    if ok and size_ok:
        try:
            import shutil
            shutil.rmtree(src, ignore_errors=True)
            print("测试图片已清理")
        except Exception:
            print(f"测试图片目录: {src}")
    else:
        print(f"测试图片目录: {src}")
    if not ok or not size_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())