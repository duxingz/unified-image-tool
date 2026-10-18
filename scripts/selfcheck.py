# -*- coding: utf-8 -*-
"""Full-flow self-check for Unified Image Tool (v1.13 sticker system + v1.14 UI).

Runs OUTSIDE pytest on purpose: nothing here is auto-answered or isolated by
conftest, so this is the independent pass that catches harness artefacts. It
asserts on delivered bytes (exported files, library files on disk) and on
RENDERED PIXELS, not on data structures.

Run:  C:\\Users\\qu\\miniconda3\\python.exe scripts\\selfcheck.py
Exit code 0 = all pass.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The Windows console defaults to GBK, which cannot encode every label this
# script prints; force UTF-8 so a stray character cannot crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np                                  # noqa: E402
from PIL import Image, ImageDraw                    # noqa: E402
from PySide6.QtCore import QEvent, QPointF             # noqa: E402
from PySide6.QtGui import QMouseEvent                  # noqa: E402
from PySide6.QtGui import QImage                    # noqa: E402
from PySide6.QtWidgets import (                     # noqa: E402
    QApplication, QDialog, QMessageBox, QSizePolicy,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}"
          f"{('  ' + str(detail)) if detail else ''}", flush=True)
    if not cond:
        FAIL.append(name)


app = QApplication.instance() or QApplication([])
# headless: modal dialogs must never block the run
QDialog.exec = lambda self: 1
for _n in ("information", "warning", "critical", "question"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: QMessageBox.Yes))

tmp = Path(tempfile.mkdtemp(prefix="uit_selfcheck_"))
print(f"workdir: {tmp}\n", flush=True)

# ---------------------------------------------------------------- isolation
# MainWindow.closeEvent() calls save_config(); make sure this script can never
# touch the real user config (it once did).
from app.core import config as config_mod                 # noqa: E402
REAL_CONFIG = config_mod.CONFIG_PATH


def _cfg_digest():
    try:
        return hashlib.sha256(Path(REAL_CONFIG).read_bytes()).hexdigest()
    except OSError:
        return "missing"


config_mod.CONFIG_DIR = tmp
config_mod.CONFIG_PATH = tmp / "config.json"
REAL_BEFORE = _cfg_digest()

# ---------------------------------------------------------------- assets
src = Image.new("RGBA", (80, 40), (0, 0, 0, 0))
d = ImageDraw.Draw(src)
d.rectangle((0, 0, 79, 39), fill=(255, 0, 0, 255))
d.rectangle((0, 0, 15, 9), fill=(0, 0, 255, 255))       # blue marker, top-LEFT
stickers_dir = tmp / "app" / "stickers"
stickers_dir.mkdir(parents=True)
src.save(stickers_dir / "logo.png")

from app.core.export import _scale_sticker, compose_from_state   # noqa: E402
from app.core.image_ops import compose_image                     # noqa: E402
from app.core.selection import RectSelection, Sticker            # noqa: E402
from app.ui.editor_canvas import EditorCanvas                    # noqa: E402
from app.ui.theme import apply_theme, current_palette, theme_qss  # noqa: E402
from app.ui.widgets import SectionCard, WatermarkPreview         # noqa: E402

W, H = 400, 300
base = Image.new("RGBA", (W, H), (255, 255, 255, 255))
canvas = EditorCanvas()
canvas.image_np = np.array(base.convert("RGB"))
canvas.original_pil = base
canvas.sticker_library = {"logo": src}
check("canvas ready", canvas.image_np is not None)

# ================================================================ v1.13
# --- rotation geometry must match the painted pixels ---------------------
st = Sticker("logo", 200, 150, 80, 40)
canvas.stickers = [st]
canvas._sel_sticker = 0
rp = canvas._rotate_handle_pos(st)
check("rotate grip exists", rp is not None)
canvas._begin_sticker_drag(rp)
canvas._drag_sticker_to(QPointF(300, 150))          # 90 deg to the right
canvas._sticker_drag = None
check("rotate 90 via grip", st.angle == 90.0, f"angle={st.angle}")

worst = 0
for ang in (0.0, 30.0, 90.0, 180.0, 225.0):
    st.angle = ang
    b = canvas._sticker_bbox_scene(st)
    painted = np.array(compose_image(base, [], [], [st], canvas.sticker_library))
    ys, xs = np.where((painted[:, :, :3] != 255).any(axis=2))
    worst = max(worst,
                abs(int(xs.min()) - round(b.left())),
                abs(int(ys.min()) - round(b.top())),
                abs(int(xs.max()) + 1 - round(b.right())),
                abs(int(ys.max()) + 1 - round(b.bottom())))
check("selection outline tracks rotated pixels", worst <= 2, f"maxerr={worst}px")

# --- delete / duplicate / restack ---------------------------------------
canvas.stickers = [Sticker("logo", 200, 150, 80, 40)]
canvas._sel_sticker = 0
canvas.copy_selected_sticker()
canvas.paste_sticker_from_clipboard()
check("duplicate -> 2 stickers", len(canvas.stickers) == 2)
sel = canvas.stickers[canvas._sel_sticker]
canvas.move_selected_sticker(False)
check("send to back reorders", canvas.stickers[0] is sel)
canvas.flip_selected_sticker(True)
check("flip toggles the selected sticker", sel.flip_h is True)
n = len(canvas.stickers)
canvas._sel_sticker = 0
canvas.delete_selected_sticker()
check("delete removes exactly one", len(canvas.stickers) == n - 1)
canvas.undo()
check("delete is undoable", len(canvas.stickers) == n)

# --- watermarks ----------------------------------------------------------
for i in (0, 1):
    canvas.stickers.append(Sticker(
        "logo", 0, 0, 40, 20, nx=0.3 + 0.2 * i, ny=0.3, size_frac=0.2,
        anchor="center", watermark=True))
plain = [s for s in canvas.stickers if not s.watermark]
check("remove all watermarks", canvas.remove_watermarks() == 2)
check("plain stickers survive", canvas.stickers == plain)
canvas.undo()
check("remove-watermarks undoable", len(canvas.stickers) == len(plain) + 2)

# --- exported bytes -----------------------------------------------------
orig_path = tmp / "orig.png"
Image.new("RGB", (W * 2, H * 2), (255, 255, 255)).save(orig_path)
ex = Sticker("logo", 200, 150, 80, 40, angle=90.0)
ex.flip_h = True
out = compose_from_state(
    str(orig_path), {str(orig_path): {"pil": base.convert("RGB"),
                                      "orig_size": (W * 2, H * 2),
                                      "stickers": [ex]}},
    {"logo": src}, {"auto_effect": "blur", "auto_strength": 10}, {})
check("export keeps original resolution", out.size == (W * 2, H * 2), out.size)
a = np.array(out.convert("RGB"))
ys, xs = np.where((a[:, :, :3] != 255).any(axis=2))
check("exported sticker is rotated (tall)",
      (int(ys.max()) - int(ys.min())) > (int(xs.max()) - int(xs.min())))
blue = np.where((a[:, :, 2] > 200) & (a[:, :, 0] < 100))
red = np.where((a[:, :, 0] > 200) & (a[:, :, 2] < 100))
check("exported sticker is flipped", blue[1].mean() > red[1].mean())
ex_path = tmp / "out.png"
out.save(ex_path)
reloaded = Image.open(ex_path)
reloaded.load()
check("exported file written and reloadable",
      ex_path.stat().st_size > 0 and reloaded.size == (W * 2, H * 2),
      f"{ex_path.stat().st_size} bytes")
check("scale carries flip/opacity",
      _scale_sticker(Sticker("logo", 1, 2, 3, 4, opacity=0.5, flip_h=True),
                     2.0, 2.0).flip_h is True)

# --- sticker library on disk -------------------------------------------
check("library asset on disk",
      sorted(p.name for p in stickers_dir.glob("*")) == ["logo.png"])

# ================================================================ v1.14
# --- both themes are real, complete stylesheets -------------------------
for name in ("xiaodu", "xiaodunight"):
    qss = theme_qss(name)
    check(f"{name} theme is a full stylesheet", len(qss) > 3000, f"{len(qss)} chars")
    check(f"{name} theme has no unresolved template braces",
          "{{" not in qss and "}}" not in qss)
apply_theme(app, "xiaodu")
check("palette follows apply_theme(xiaodu)",
      current_palette()["canvas"] == "#FAF6EF")
apply_theme(app, "xiaodunight")
check("palette follows apply_theme(xiaodunight)",
      current_palette()["canvas"] == "#211E1A")

# --- components render ---------------------------------------------------
card = SectionCard("标题", "说明")
card.resize(300, 200)
card.show()
app.processEvents()
pm = card.grab()
check("SectionCard renders", pm.width() > 0 and pm.height() > 0)


def _rgb(widget):
    widget.resize(widget.size().expandedTo(widget.minimumSizeHint()))
    img = widget.grab().toImage().convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8)
    return buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)


pv = WatermarkPreview()
pv.resize(240, 160)


def _red_count(frac):
    pv.set_state(asset=src, size_frac=frac, anchor="bottom-right")
    arr = _rgb(pv)
    return int(((arr[:, :, 0] > 180) & (arr[:, :, 1] < 90) & (arr[:, :, 2] < 90)).sum())


small, big = _red_count(0.15), _red_count(0.90)
check("watermark preview paints the watermark", small > 0, f"{small}px")
check("watermark preview scales with 图上大小", big > small * 3, f"{small} -> {big}px")

# --- the confusion that started the restyle ------------------------------
st2 = Sticker("logo", 200, 150, 80, 40)
canvas.stickers = [st2]
canvas._sel_sticker = 0
check("watermark never rotates", canvas.set_sticker_angle(45.0) is True
      and canvas.stickers[0].angle == 45.0)
wm = Sticker("logo", 0, 0, 40, 20, nx=0.5, ny=0.5, size_frac=0.2,
             anchor="center", watermark=True)
canvas.stickers = [wm]
canvas._sel_sticker = 0
check("watermark rotation refused", canvas.set_sticker_angle(45.0) is False
      and wm.angle == 0.0)

# --- main window: pages build, panels stay narrowable --------------------
import app.ui.main_window as mw_mod                          # noqa: E402
from app.core.config import default_config                   # noqa: E402
from app.ui.main_window import MainWindow                    # noqa: E402


class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, *a, **k):
        return []


mw_mod.resource_path = lambda rel: tmp / rel
win = MainWindow(StubDetector(), default_config())
win.resize(1440, 900)
win.show()
for _ in range(3):
    app.processEvents()

tab_names = ["编辑", "自动打码", "贴纸", "水印", "批量"]
check("tool panel keeps its 5 tabs", win.tool_panel.count() == 5,
      [win.tool_panel.tabText(i) for i in range(win.tool_panel.count())])
check("tab labels unchanged",
      [win.tool_panel.tabText(i) for i in range(5)] == tab_names)
check("right panel keeps Ignored width policy",
      win.tool_panel.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored)
win.splitter.setSizes([220, 800, 340])
app.processEvents()
check("left panel narrows to ~220", win.splitter.sizes()[0] <= 260,
      win.splitter.sizes())
win.splitter.setSizes([900, 400, 220])
app.processEvents()
check("right panel narrows to ~220", win.splitter.sizes()[2] <= 260,
      win.splitter.sizes())

for i, nm in enumerate(tab_names):
    win.tool_panel.setCurrentIndex(i)
    for _ in range(2):
        app.processEvents()
    shot = win.grab()
    check(f"tab [{nm}] renders", shot.width() > 0 and not shot.isNull())

# --- watermark page semantics -------------------------------------------
from PySide6.QtWidgets import QLabel                          # noqa: E402
wt = win.tool_panel.widget(3)
labels = " ".join(lbl.text() for lbl in wt.findChildren(QLabel))
check("字号 row is labelled as clarity", "文字精度（字号）" in labels)
check("size row is labelled as on-image size", "图上大小（占短边）" in labels)
check("字号 carries a 'does not change size' note",
      "不改变水印在照片上的大小" in labels)
check("素材 lives in the 图片水印 card",
      any(c.isAncestorOf(win.tool_panel.wm_asset)
          for c in wt.findChildren(SectionCard)
          if any(l.objectName() == "cardTitle" and l.text() == "图片水印"
                 for l in c.findChildren(QLabel))))

win.tool_panel.wm_size.setValue(70)
check("preview follows the panel spin box",
      abs(win.tool_panel.wm_preview._size_frac - 0.70) < 1e-6)

# ================================================================ v1.22.0
# --- 自动打码页：动作区在页首 / 结果条 / 两列类别 / 控件统一 ----------
print("\n--- v1.22.0: 自动打码页重排 ---")
from types import SimpleNamespace as _NS                       # noqa: E402

from app.core.config import CLASS_KEYS as _CLASS_KEYS          # noqa: E402

_auto_img = tmp / "auto_page.png"
Image.fromarray(np.full((120, 160, 3), 60, dtype=np.uint8)).save(_auto_img)
win.thumbnail_list.add_paths([str(_auto_img)])
win._load_path(str(_auto_img), save_current=False)
win.tool_panel.setCurrentIndex(1)
for _ in range(3):
    app.processEvents()

_tp = win.tool_panel
_cards = [c for c in _tp.widget(1).findChildren(SectionCard) if not c.isHidden()]
check("主按钮排在卡片之前（旧版压在 3 张卡之后、页面 76% 处）",
      _tp.btn_detect.y() < _cards[0].y(),
      f"btn y={_tp.btn_detect.y()}  first card y={_cards[0].y()}")
check("主按钮 36 高；清除按钮是 ghost 且更小（不再同宽同高相贴）",
      _tp.btn_detect.height() == 36
      and _tp.btn_clear_auto.property("variant") == "ghost"
      and _tp.btn_clear_auto.width() < _tp.btn_detect.width(),
      f"{_tp.btn_detect.width()}x{_tp.btn_detect.height()}"
      f" / {_tp.btn_clear_auto.width()}x{_tp.btn_clear_auto.height()}")
_sizes = {(c.width(), c.height()) for c in (
    _tp.conf_spin, _tp.iou_spin, _tp.input_size_combo,
    _tp.auto_effect_combo, _tp.auto_strength_spin, _tp.mask_dilate_spin)}
check("六个控件统一 132×30（旧版 93/90/82×31 与 217/230×28）",
      _sizes == {(132, 30)}, _sizes)
check("检测类别排成两列",
      len({round(_tp.class_checks[k].x()) for k in _CLASS_KEYS}) == 2)
check("高级参数默认收起、点开后展开",
      _tp.plate_card.isHidden() is True)
_tp.btn_advanced.setChecked(True)
app.processEvents()
check("高级参数点开后展开", _tp.plate_card.isHidden() is False)
_tp.btn_advanced.setChecked(False)
app.processEvents()


class _DetStub(StubDetector):
    """固定返回 3 个框（阴部×2 + 阴茎×1），用来验结果条与每类计数。"""

    def detect(self, image_bgr, *a, **k):
        h, w = image_bgr.shape[:2]
        out = []
        for i, key in enumerate(("vagina", "vagina", "penis")):
            m = np.zeros((h, w), dtype=bool)
            m[10:40, 10 + i * 20:30 + i * 20] = True
            out.append(_NS(mask=m, class_name=key, bbox=(10, 10, 40, 40),
                           class_id=0, confidence=0.9))
        return out


win.detector = _DetStub()
win.auto_detect_current()
check("结果条写出命中数与每类计数（旧版只在状态栏报一句）",
      "3 处" in _tp.auto_status.text() and "阴部×2" in _tp.auto_status.text(),
      _tp.auto_status.text())
check("每类命中数贴到勾选框上", "×2" in _tp.class_checks["vagina"].text(),
      _tp.class_checks["vagina"].text())
check("有掩码时「清除自动掩码」才可用", _tp.btn_clear_auto.isEnabled())
_tp.conf_spin.setValue(0.61)
check("改置信度后旧结果作废并提示重新检测",
      "需重新检测" in _tp.auto_status.text(), _tp.auto_status.text())
win.detector = StubDetector()

# ================================================================ v1.14.3
# --- an erased spot must be censorable again by hand ---------------------
print("\n--- v1.14.3: 橡皮擦之后能否再次手动打码 ---")


def _checker(w=200, h=150):
    a = (np.indices((h, w)).sum(axis=0) % 2)
    rgb = np.where(a[..., None] == 0, 20, 230).astype(np.uint8)
    return Image.fromarray(np.repeat(rgb, 3, axis=2), "RGB")


def _contrast(im):
    a = np.array(im.convert("RGB")).astype(int)[60:100, 60:140]
    return int(((a < 60).any(axis=2) | (a > 190).any(axis=2)).sum())


cv2 = EditorCanvas()
cv2.image_np = np.array(_checker())
cv2.original_pil = _checker()
cv2.sticker_library = {}
full = _contrast(_checker())
sel1 = RectSelection(60, 60, 140, 100, effect="blur", strength=25)
check("先手动打码 → 打上了",
      _contrast(compose_image(cv2.original_pil, [sel1], [], [], {})) < full * 0.2)
erase2 = np.zeros((150, 200), dtype=bool)
erase2[55:105, 55:145] = True
cv2.erase_mask = erase2
check("橡皮擦擦过后 → 还原了",
      _contrast(compose_image(cv2.original_pil, [sel1], [], [], {},
                              erase_mask=erase2)) > full * 0.8)
sel2 = RectSelection(60, 60, 140, 100, effect="blur", strength=25)
cv2._punch_through_erase(sel2)
check("擦过之后再手动画码 → 能打上（v1.14.3 修）",
      _contrast(compose_image(cv2.original_pil, [sel1, sel2], [], [], {},
                              erase_mask=cv2.erase_mask)) < full * 0.2)
check("穿透只作用在新选区范围内",
      not cv2.erase_mask[65:95, 65:135].any()      # 选区内被清掉
      and not cv2.erase_mask[0:20, 0:20].any()     # 本来没擦的地方仍没有
      and cv2.erase_mask[55:59, 60:140].all())     # 选区外那条擦除带保留

# --- numpad recognition without relying on one Qt flag -------------------
print("\n--- v1.14.3: 小键盘识别（含 Fn 叠加键盘）---")


class _FakeKey:
    def __init__(self, key, mods, vk=0):
        self._k, self._m, self._vk = key, mods, vk

    def key(self):
        return self._k

    def modifiers(self):
        return self._m

    def nativeVirtualKey(self):
        return self._vk


from PySide6.QtCore import Qt as _Qt                 # noqa: E402
KP = _Qt.KeyboardModifier.KeypadModifier
NOMOD = _Qt.KeyboardModifier.NoModifier
check("真方向键 ↑ = 上一张（v1.14.4 全局换图）",
      win._pad_delta(_FakeKey(_Qt.Key_Up, NOMOD)) == -1)
check("真方向键 → = 下一张",
      win._pad_delta(_FakeKey(_Qt.Key_Right, NOMOD)) == 1)
check("小键盘↑ (Key_Up+Keypad) = 上一张",
      win._pad_delta(_FakeKey(_Qt.Key_Up, KP)) == -1)
check("小键盘8 (Key_8+Keypad, NumLock 开) = 上一张",
      win._pad_delta(_FakeKey(_Qt.Key_8, KP)) == -1)
check("VK_NUMPAD2 虚拟键 = 下一张",
      win._pad_delta(_FakeKey(_Qt.Key_2, NOMOD, 0x62)) == 1)
check("顶排数字 8 不再换图（v1.14.4 去掉的兜底）",
      win._pad_delta(_FakeKey(_Qt.Key_8, NOMOD)) is None)
check("Ctrl+方向键不被抢",
      win._pad_delta(_FakeKey(_Qt.Key_Up,
                              _Qt.KeyboardModifier.ControlModifier)) is None)

# ================================================================ v1.14.4 笔刷
print("\n--- v1.14.4: 笔刷所见即所得 ---")
from app.core.selection import BrushSelection as _BS         # noqa: E402
from app.core.image_ops import apply_mask_effect as _apply_eff  # noqa: E402

_rng = np.random.default_rng(5)
_big = Image.fromarray(_rng.integers(0, 256, (300, 400, 3), dtype=np.uint8), "RGB")
cvv = EditorCanvas()
cvv.image_np = np.array(_big)
cvv.original_pil = _big
cvv.sticker_library = {}
cvv._rebuild()
_stroke = [(x, 150) for x in range(80, 321, 6)]
cvv.set_brush_size(30)
_prev = cvv._stroke_mask(_stroke, 30)
_comm = np.array(_BS(_stroke, 30).to_mask((400, 300)).convert("L")) > 127
check("预览与提交的笔迹几何完全一致（曾差 2000+ px）",
      int((_prev ^ _comm).sum()) == 0,
      f"{int((_prev ^ _comm).sum())} px differ")

cvv.tool = "brush"
cvv.current_effect = "pixel"
cvv.current_strength = 30
_vp0 = cvv.mapFromScene(QPointF(*_stroke[0]))
cvv.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(_vp0),
                                QPointF(_vp0), _Qt.MouseButton.LeftButton,
                                _Qt.MouseButton.LeftButton,
                                _Qt.KeyboardModifier.NoModifier))
for _p in _stroke[1:]:
    _vp = cvv.mapFromScene(QPointF(*_p))
    cvv.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(_vp),
                                   QPointF(_vp), _Qt.MouseButton.NoButton,
                                   _Qt.MouseButton.LeftButton,
                                   _Qt.KeyboardModifier.NoModifier))
    cvv._render_drag_stroke()
cvv._render_drag_stroke()
_live = cvv._live_img.copy()
_vpe = cvv.mapFromScene(QPointF(*_stroke[-1]))
cvv.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(_vpe),
                                  QPointF(_vpe), _Qt.MouseButton.LeftButton,
                                  _Qt.MouseButton.NoButton,
                                  _Qt.KeyboardModifier.NoModifier))
_band = (slice(120, 180), slice(80, 320))
_d = float(np.abs(_live[_band].astype(int)
                  - cvv._composed_base[_band].astype(int)).mean())
check("拖动中的预览 = 松手后的结果（马赛克，曾差 31/255）",
      _d < 1.0, f"{_d:.2f}/255")

cvv.set_brush_size(30)
cvv.resetTransform()
check("缩放 100% 时光标 = 真实笔刷 30px",
      cvv._make_brush_cursor()[0].width() == 30)
cvv.scale(2.0, 2.0)
check("缩放 200% 时光标跟随放大到 60px（曾恒定 30px）",
      cvv._make_brush_cursor()[0].width() == 60)
cvv.set_brush_size(100)
cvv.resetTransform()
check("笔刷 100 时光标 = 100px（曾被截断到 64px）",
      cvv._make_brush_cursor()[0].width() == 100)

# ============================================================ v1.14.5 记忆
# 面板上改过的自动打码设置必须落盘，重启后必须回到面板与画布
print("\n--- v1.14.5: 设置记忆 ---")
check("默认手动强度 = 15",
      default_config()["edit_defaults"]["strength"] == 15)
check("默认自动强度 = 15",
      default_config()["auto_strength"] == 15)

tp = win.tool_panel
tp.auto_strength_spin.setValue(77)
tp.mask_dilate_spin.setValue(3)
for _k, _cb in tp.class_checks.items():
    _cb.setChecked(_k == "anus")
check("面板值已推给画布（不再等图片加载）",
      win.canvas.auto_params["auto_strength"] == 77)
win._dirty_paths.clear()
win.close()

_saved = json.loads(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))
check("退出时自动强度已落盘（用户报的 bug）", _saved["auto_strength"] == 77)
check("退出时遮罩膨胀已落盘", _saved["auto_mask_dilate"] == 3)
check("退出时检测类别已落盘", _saved["selected_regions"] == ["anus"])

from app.ui.main_window import MainWindow as _MW            # noqa: E402
cfg2 = default_config()
cfg2.update({"auto_strength": 77, "auto_mask_dilate": 3,
             "selected_regions": ["anus"]})
win2 = _MW(StubDetector(), cfg2)
check("重启后面板恢复自动强度 77", win2.tool_panel.auto_strength_spin.value() == 77)
check("重启后检测类别勾选被恢复（此前完全没恢复）",
      [k for k, cb in win2.tool_panel.class_checks.items() if cb.isChecked()] == ["anus"])
check("重启后画布用的是存下来的强度",
      win2.canvas.auto_params["auto_strength"] == 77)
win2._dirty_paths.clear()
win2.close()

# ============================================================ v1.21.0 橡皮擦
# 用户报「擦完留一圈细碎毛刺」：实测残边厚 5–20px（中位 5–7px），
# 外扩 2px 只能清掉 0.3% → 外扩量必须跟随「覆盖扩展」。
print("\n--- v1.21.0: 橡皮擦外扩（跟随覆盖扩展）+ 圆角接头 ---")
from app.core.effects import expand_mask                     # noqa: E402

_cv = EditorCanvas()
_cv.image_np = np.array(Image.new("RGB", (300, 240), (120, 120, 120)))
_cv.original_pil = Image.new("RGB", (300, 240), (120, 120, 120))
_cv.sticker_library = {}
_cv.auto_params = {"auto_effect": "pixel", "auto_strength": 15,
                   "auto_mask_dilate": 10}
_cv.erase_mask = np.zeros((240, 300), dtype=bool)
_cv._rebuild()
check("外扩量跟随「覆盖扩展」= 10", _cv._eraser_grow() == 10)
_cv.auto_params["auto_mask_dilate"] = 0
check("覆盖扩展为 0 时不外扩（保持旧行为）", _cv._eraser_grow() == 0)
_cv.auto_params["auto_mask_dilate"] = 10

_pts = [(60, 120), (240, 120)]
_cv._apply_erase_points(_pts)
_stroke = _cv._stroke_mask(_pts, _cv.brush_size)
check("擦除层包含整条笔迹", int((_stroke & ~_cv.erase_mask).sum()) == 0)
check("擦除层恰好 = expand(笔迹, 覆盖扩展)",
      int((_cv.erase_mask ^ expand_mask(_stroke, 10)).sum()) == 0)

_cv.set_brush_size(30)
_cv.tool = "eraser"
_cv.resetTransform()
check("橡皮擦光标 = 笔迹 30 + 2×外扩 10 = 50px",
      _cv._make_brush_cursor()[0].width() == 50)
_cv.scale(2.0, 2.0)
check("放大 2 倍后橡皮擦光标跟随 = 100px",
      _cv._make_brush_cursor()[0].width() == 100)
_cv.resetTransform()
_cv.tool = "brush"
check("画笔光标不受外扩影响 = 30px",
      _cv._make_brush_cursor()[0].width() == 30)

# 圆角接头：90° 拐角外侧不能被平头接头切掉（否则擦除会留楔形缺口）
_corner = _cv._stroke_mask([(100, 100), (200, 100), (200, 200)], 30)
check("笔迹 90° 拐角外侧被圆角接头补齐",
      int((~_corner[95:105, 195:205]).sum()) == 0,
      f"拐角处缺 {int((~_corner[95:105, 195:205]).sum())} px")

# ---------------------------------------------------------------- isolation
check("real user config untouched by this script",
      _cfg_digest() == REAL_BEFORE, str(REAL_CONFIG))

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"), flush=True)
sys.exit(1 if FAIL else 0)
