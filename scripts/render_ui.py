# -*- coding: utf-8 -*-
"""Render every page/window in both themes to PNG for visual review.

Run:  C:\\Users\\qu\\miniconda3\\python.exe scripts\\_render_ui.py <outdir>
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                       # noqa: E402
from PIL import Image, ImageDraw          # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_ui_shots")
OUT.mkdir(parents=True, exist_ok=True)


class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        return []


app = QApplication.instance() or QApplication([])

# ---------------------------------------------------------------- fidelity
# Two capture-fidelity traps found by the visual reviewer:
#  1. the offscreen platform plugin reports ZERO font families here, so every
#     glyph renders as a tofu box -> no text QA is possible without loading a
#     real font explicitly;
#  2. offscreen defaults to the Fusion base style, while the shipped app gets
#     windows11 on Windows -- and the base style is what draws the sub-controls
#     the QSS deliberately does not touch (spin arrows, combo arrow, checkbox).
#  3. the windows11 base style paints the combo chevron / check glyphs from
#     Segoe Fluent Icons, which is NOT in the default offscreen family list:
#     without it every combo renders an EMPTY box and looks like a defect.
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QStyleFactory  # noqa: E402
for _p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
           r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf",
           r"C:\Windows\Fonts\SegoeIcons.ttf", r"C:\Windows\Fonts\segmdl2.ttf",
           r"C:\Windows\Fonts\seguisym.ttf"):
    if os.path.exists(_p):
        QFontDatabase.addApplicationFont(_p)
app.setFont(QFont("Microsoft YaHei UI", 9))
print(f"font families available: {len(QFontDatabase.families())}", flush=True)
for _style in ("windows11", "windowsvista", "Fusion"):
    if _style in QStyleFactory.keys():
        app.setStyle(_style)
        print(f"base style: {_style}", flush=True)
        break

# This harness closes windows that hold edits, which would pop the "unsaved
# changes" modal and block forever (conftest does the same job for pytest).
from PySide6.QtWidgets import QDialog, QMessageBox  # noqa: E402
QDialog.exec = lambda self: 1
for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: QMessageBox.Yes))


def check_overlaps(root_widget, label):
    """Report hint labels that overlap a sibling control.

    A visual pass flagged the settings dialog's 推理设备 caption as clipped by
    its combo. That turned out to be THIS harness forcing a too-small dialog
    size; the check stays so a real overlap can never slip through again.

    v1.22.0：检查范围从「同级控件」扩到「任意后代」——卡片标题变大后，水印页的
    「位置」说明曾压到九宫格按钮上（同级不同级都查不到，因为一个是 SettingRow
    里的 QLabel、另一个是跨两层的 QPushButton 子控件）。"""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtWidgets import QLabel, QWidget
    bad = []
    hints = [lbl for lbl in root_widget.findChildren(QLabel)
             if lbl.objectName() == "hint" and lbl.isVisible()]
    others = [w for w in root_widget.findChildren(QWidget)
              if w.isVisible()
              and w not in hints
              and w.__class__.__name__ in (
                  "QComboBox", "QPushButton", "QSpinBox", "QDoubleSpinBox",
                  "QCheckBox", "QLineEdit", "AnchorPicker", "WatermarkPreview",
                  "StickerDropList", "AutoStatusBar", "SectionCard")]
    for lbl in hints:
        lr = QRect(lbl.mapTo(root_widget, QPoint(0, 0)), lbl.size())
        for sib in others:
            if sib.isAncestorOf(lbl) or lbl.isAncestorOf(sib):
                continue
            sr = QRect(sib.mapTo(root_widget, QPoint(0, 0)), sib.size())
            if lr.intersects(sr):
                bad.append((lbl.text()[:20], sib.__class__.__name__))
    print(f"     overlap [{label}]: {'OK' if not bad else 'OVERLAP ' + str(bad)}",
          flush=True)
    return not bad


def shoot(widget, name, size=None):
    if size:
        widget.resize(*size)
    widget.show()
    for _ in range(4):
        app.processEvents()
    pm = widget.grab()
    path = OUT / f"{name}.png"
    pm.save(str(path))
    print(f"  {path.name}  {pm.width()}x{pm.height()}", flush=True)
    widget.hide()


def make_assets(tmp: Path):
    """A couple of stickers + images so the pages are not empty."""
    sdir = tmp / "app" / "stickers"
    sdir.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (160, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, 159, 63), 10, fill=(255, 255, 255, 230))
    d.text((14, 20), "WATERMARK 2026", fill=(20, 20, 20, 255))
    im.save(sdir / "watermark_sample.png")
    im.save(sdir / "watermark_sample_2.png")
    ids = []
    for i in range(4):
        arr = np.full((300, 400, 3), 40 + i * 40, dtype=np.uint8)
        arr[40:120, 40:200] = 200
        p = tmp / f"photo_{i}.png"
        Image.fromarray(arr).save(p)
        ids.append(str(p))
    return ids


def run_theme(theme_name: str):
    import tempfile
    import app.ui.main_window as mw_mod
    from app.core.config import default_config
    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme

    tmp = Path(tempfile.mkdtemp(prefix=f"uit_shot_{theme_name}_"))
    mw_mod.resource_path = lambda rel: tmp / rel
    # CRITICAL: MainWindow.closeEvent() calls save_config(). Without this the
    # harness overwrites the user's real %APPDATA% settings (it did once).
    from app.core import config as config_mod
    config_mod.CONFIG_DIR = tmp
    config_mod.CONFIG_PATH = tmp / "config.json"
    paths = make_assets(tmp)

    apply_theme(app, theme_name)
    print(f"[{theme_name}]", flush=True)

    cfg = default_config()
    cfg["theme"] = theme_name
    w = MainWindow(StubDetector(), cfg)
    w.resize(1440, 900)
    w.thumbnail_list.add_paths(paths)
    w._load_path(paths[0], save_current=False)
    # put something on the canvas so overlays are visible
    from app.core.selection import Sticker
    w.canvas.add_sticker("watermark_sample", x=200, y=120, width=180, height=72)
    w.canvas._sel_sticker = 0

    tab_names = ["edit", "auto", "sticker", "watermark", "batch"]
    for i, tab in enumerate(tab_names):
        w.tool_panel.setCurrentIndex(i)
        for _ in range(3):
            app.processEvents()
        shoot(w, f"{theme_name}_main_{tab}", (1440, 900))
        check_overlaps(w.tool_panel, f"{theme_name} tab {tab}")

    # v1.22.0 自动打码页的三个额外状态（静帧验收缺了这几张就判不了状态机）：
    # 真掩码 + 结果条（清除按钮应变可用）、展开高级参数、无图空态。
    from types import SimpleNamespace
    w.tool_panel.setCurrentIndex(1)
    demo = [SimpleNamespace(class_name=k)
            for k in ("vagina", "vagina", "penis", "nipple")]
    m = np.zeros(w.canvas.image_np.shape[:2], dtype=bool)
    m[60:200, 100:260] = True
    w.canvas.set_auto_mask(m)                 # 真掩码 → 「清除自动掩码」才可用
    w.tool_panel.show_detect_result(demo)
    for _ in range(3):
        app.processEvents()
    shoot(w, f"{theme_name}_main_auto_detected", (1440, 900))
    print(f"     clear-btn enabled after detect: "
          f"{w.tool_panel.btn_clear_auto.isEnabled()}", flush=True)
    w.tool_panel.btn_advanced.setChecked(True)
    for _ in range(3):
        app.processEvents()
    shoot(w, f"{theme_name}_main_auto_advanced", (1440, 900))
    w.tool_panel.btn_advanced.setChecked(False)
    # 无图空态：清掉画布与选中路径，让结果条退回「先导入一张图片」
    w.tool_panel.setCurrentIndex(1)
    w.canvas.clear_image()
    w.current_path = None
    w.tool_panel.on_image_changed()
    for _ in range(3):
        app.processEvents()
    shoot(w, f"{theme_name}_main_auto_noimage", (1440, 900))
    print(f"     no-image: detect enabled={w.tool_panel.btn_detect.isEnabled()}"
          f" status='{w.tool_panel.auto_status.text()}'", flush=True)
    print(f"     no-image: clear enabled={w.tool_panel.btn_clear_auto.isEnabled()}",
          flush=True)
    # 后面还有「批量」子页截图：必须把面板切回批量页，否则截出来是自动页
    w.tool_panel.setCurrentIndex(4)
    w._load_path(paths[0], save_current=False)
    for _ in range(3):
        app.processEvents()

    # ---------------------------------------------------------- geometry probe
    # hard evidence for the defects the reviewer found (D1 collapsed combos,
    # D2 squeezed anchor grid, D5 preview box)
    tp = w.tool_panel
    print("  -- geometry --", flush=True)
    for nm, wdg in (("效果", tp.effect_combo),
                    ("检测分辨率", tp.input_size_combo),
                    ("自动效果", tp.auto_effect_combo),
                    ("素材", tp.wm_asset)):
        print(f"     {nm:<6} size={wdg.width()}x{wdg.height()}"
              f"  hint={wdg.sizeHint().width()}x{wdg.sizeHint().height()}"
              f"  visible={wdg.isVisible()}", flush=True)
    ap = tp.wm_anchor_picker
    print(f"     AnchorPicker size={ap.width()}x{ap.height()}"
          f"  minimum={ap.minimumSizeHint().width()}x{ap.minimumSizeHint().height()}",
          flush=True)
    buttons = ap.findChildren(type(tp.wm_color))
    ys = sorted({b.y() for b in buttons})
    print(f"     anchor button rows at y={ys}"
          f"  button h={buttons[0].height() if buttons else 0}", flush=True)
    pv = tp.wm_preview
    print(f"     Preview size={pv.width()}x{pv.height()}", flush=True)
    for i in range(tp.count()):
        page = tp.widget(i)
        need, have = page.sizeHint().height(), tp.height()
        print(f"     tab {tab_names[i]:<9} needs={need:>4} available={have:>4}"
              f"  {'OK' if need <= have else 'CLIPPED'}", flush=True)

    # batch sub-tabs
    from PySide6.QtWidgets import QTabWidget
    batch = w.tool_panel.widget(4)
    sub = batch.findChild(QTabWidget)
    if sub is not None:
        for j, nm in enumerate(["convert", "resize", "rename", "pipeline"]):
            sub.setCurrentIndex(j)
            for _ in range(3):
                app.processEvents()
            shoot(w, f"{theme_name}_batch_{nm}", (1440, 900))
    else:
        print("  batch sub-tabs not found", flush=True)

    # combo popup: the CURRENT item once rendered blank (a stylesheet bug where
    # the selection text colour was applied without its background). Capture the
    # OPEN popup so it can be checked visually.
    w.tool_panel.setCurrentIndex(1)          # 自动打码 tab
    for _ in range(3):
        app.processEvents()
    combo = w.tool_panel.input_size_combo
    combo.setCurrentIndex(combo.count() - 1)
    combo.showPopup()
    for _ in range(6):
        app.processEvents()
    popup = combo.view().window()
    pm = popup.grab()
    if pm.width():
        p = OUT / f"{theme_name}_combo_popup.png"
        pm.save(str(p))
        print(f"  {p.name}  {pm.width()}x{pm.height()}", flush=True)
    combo.hidePopup()
    for _ in range(3):
        app.processEvents()

    # settings dialog
    try:
        from app.ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(w.config, w)
        # shoot at the dialog's OWN size: forcing one compresses the layout
        # and invents overlaps the shipped dialog never has
        shoot(dlg, f"{theme_name}_settings")
        print(f"     settings size={dlg.width()}x{dlg.height()}"
              f" hint needed={dlg.sizeHint().width()}x{dlg.sizeHint().height()}",
              flush=True)
        check_overlaps(dlg, f"{theme_name} settings")
        dlg.close()
    except Exception as e:
        print(f"  settings failed: {e}", flush=True)

    # save/export dialog
    try:
        from app.ui.save_dialog import SaveDialog
        sd = SaveDialog(w)
        shoot(sd, f"{theme_name}_export")
        sd.close()
    except Exception as e:
        print(f"  save dialog failed: {e}", flush=True)

    # mini window
    try:
        from app.ui.mini_window import MiniWindow
        mini = MiniWindow(StubDetector(), w.config)
        shoot(mini, f"{theme_name}_mini", (420, 300))
        mini.close()
    except Exception as e:
        print(f"  mini failed: {e}", flush=True)

    w.close()


for t in ("xiaodu", "xiaodunight"):
    run_theme(t)

print(f"\nshots -> {OUT}", flush=True)
