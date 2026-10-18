# -*- coding: utf-8 -*-
"""v1.21.0 — 橡皮擦外扩（跟随「覆盖扩展」），根治擦不干净的锯齿残边。

用户症状（2026-10-13 报，拿他的导出图逐像素 A/B 量出来的）：
贴着打码区边缘擦完，会留下**一圈细碎毛刺**；实测残边厚 5–20px（中位 5–7px），
而原来的橡皮擦几何是"硬边圆笔刷减去硬边打码区"，两者不可能对齐。
外扩 2px 只能清掉 0.3% 的残边 → 所以外扩量必须跟随 `auto_mask_dilate`（默认 10）。

本文件锁五件事：
  1. 外扩量 = 当前 `auto_mask_dilate`（0 时不外扩，保持旧行为）；
  2. 提交进 erase_mask 的几何确实被外扩；
  3. 拖动预览与松手提交**逐像素一致**（v1.14.4 的教训：预览不能与最终不同）；
  4. 橡皮擦光标按"笔迹 + 外扩"画大（所见即所得），画笔光标不受影响；
  5. **贴着膨胀边界擦一圈 → 打码残留为 0**（这条就是用户报的症状本身；
     用 monkeypatch 把外扩关掉可复现出残边，证明这条锁真的咬得住）。
"""
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from app.core.effects import expand_mask
from app.core.image_ops import compose_image
from app.core.selection import MaskSelection
from app.ui.editor_canvas import EditorCanvas

W, H = 400, 300
CX, CY, R = 200, 150, 50          # 检测圆；真正被打码的是 r = R + dilate


def texture():
    rng = np.random.default_rng(21)
    return Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), "RGB")


def disc(radius, cx=CX, cy=CY):
    yy, xx = np.ogrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


@pytest.fixture()
def canvas(qapp):
    cv = EditorCanvas()
    cv.image_np = np.array(texture())
    cv.original_pil = texture()
    cv.sticker_library = {}
    cv.auto_mask = disc(R)
    cv.auto_params = {"auto_effect": "pixel", "auto_strength": 15,
                      "auto_mask_dilate": 10}
    cv.erase_mask = np.zeros((H, W), dtype=bool)
    cv._rebuild()
    return cv


def _stroke_points(max_r, step=10):
    """覆盖半径 max_r 的锯齿路径（笔刷宽 24 时足以铺满）。"""
    pts = []
    for y in range(CY - max_r, CY + max_r + 1, step):
        xs = range(CX - max_r, CX + max_r + 1, step)
        row = list(xs) if (y // step) % 2 == 0 else list(reversed(xs))
        for x in row:
            if (x - CX) ** 2 + (y - CY) ** 2 <= max_r ** 2:
                pts.append((x, y))
    return pts


def _drag_erase(cv, pts, render_each_frame=True):
    cv.tool = "eraser"          # 不设就会走"指针"分支，松开时根本不提交擦除
    vp = cv.mapFromScene(QPointF(*pts[0]))
    cv.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(vp),
                                   QPointF(vp), Qt.MouseButton.LeftButton,
                                   Qt.MouseButton.LeftButton,
                                   Qt.KeyboardModifier.NoModifier))
    for p in pts[1:]:
        q = cv.mapFromScene(QPointF(*p))
        cv.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(q), QPointF(q),
                                      Qt.MouseButton.NoButton,
                                      Qt.MouseButton.LeftButton,
                                      Qt.KeyboardModifier.NoModifier))
        if render_each_frame:
            cv._render_drag_erase()
    if render_each_frame:
        cv._render_drag_erase()
    e = cv.mapFromScene(QPointF(*pts[-1]))
    cv.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(e),
                                     QPointF(e), Qt.MouseButton.LeftButton,
                                     Qt.MouseButton.NoButton,
                                     Qt.KeyboardModifier.NoModifier))


def _residue_mask(cv):
    """导出图与原图不同的像素（= 仍被打码的地方）。

    compose_image(original, selections, auto_masks, stickers, library, ...)
    —— 第 5 个位置参数是贴纸库，不是 params。"""
    out = np.asarray(compose_image(
        cv.original_pil, cv.selections,
        [MaskSelection(cv.auto_mask, effect="pixel", strength=15)],
        cv.stickers, cv.sticker_library,
        params=cv.params, auto_params=cv.auto_params,
        erase_mask=cv.erase_mask).convert("RGB")).astype(np.int16)
    orig = np.asarray(cv.original_pil.convert("RGB")).astype(np.int16)
    return np.abs(out - orig).max(axis=2) > 8


def _residue(cv, radius):
    return int((_residue_mask(cv) & disc(radius)).sum())


# ---------------------------------------------------------------- 1/2 外扩量

def test_grow_follows_the_dilate_setting(canvas):
    assert canvas._eraser_grow() == 10
    canvas.auto_params["auto_mask_dilate"] = 0
    assert canvas._eraser_grow() == 0
    canvas.auto_params["auto_mask_dilate"] = 13
    assert canvas._eraser_grow() == 13
    canvas.auto_params.pop("auto_mask_dilate")
    assert canvas._eraser_grow() == 0, "缺少设置时必须退回 0（旧行为）"


def test_erase_layer_is_grown(canvas):
    pts = [(100, 150), (300, 150)]
    canvas._apply_erase_points(pts)
    stroke = canvas._stroke_mask(pts, canvas.brush_size)
    assert int((stroke & ~canvas.erase_mask).sum()) == 0, "笔迹本身必须被擦掉"
    want = expand_mask(stroke, 10)
    assert int((canvas.erase_mask ^ want).sum()) == 0, \
        "erase_mask 必须正好等于 笔迹 ∪ 外扩(覆盖扩展)"

    cv2 = EditorCanvas()
    cv2.image_np = np.array(texture())
    cv2.original_pil = texture()
    cv2.auto_params = {"auto_mask_dilate": 0}
    cv2._apply_erase_points(pts)
    assert int((cv2.erase_mask ^ cv2._stroke_mask(pts, cv2.brush_size)).sum()) == 0, \
        "覆盖扩展为 0 时必须与旧行为完全一致（不外扩）"


# ---------------------------------------------------------------- 3 预览=提交

def test_drag_preview_matches_the_commit(canvas):
    """拖动中看到的擦除范围必须与松手后的结果一致（v1.14.4 的教训）。"""
    pts = _stroke_points(R + 4, step=10)
    _drag_erase(canvas, pts, render_each_frame=True)
    live = canvas._live_img.copy()
    committed = canvas._composed_base.copy()
    band = (slice(CY - R - 12, CY + R + 12), slice(CX - R - 12, CX + R + 12))
    d = float(np.abs(live[band].astype(int) - committed[band].astype(int)).mean())
    assert d < 1.0, f"拖动预览与松手结果不一致（{d:.2f}/255）"


# ---------------------------------------------------------------- 4 光标

@pytest.mark.parametrize("zoom", [0.5, 1.0, 2.0])
def test_eraser_cursor_shows_the_grown_area(canvas, zoom):
    canvas.set_brush_size(30)
    canvas.tool = "eraser"
    canvas.resetTransform()
    canvas.scale(zoom, zoom)
    pm, _ = canvas._make_brush_cursor()
    assert pm.width() == max(8, min(200, int(round((30 + 2 * 10) * zoom))))

    canvas.tool = "brush"          # 画笔不受外扩影响
    pm2, _ = canvas._make_brush_cursor()
    assert pm2.width() == max(8, min(200, int(round(30 * zoom))))


def test_cursor_follows_a_dilate_change(canvas):
    canvas.set_brush_size(30)
    canvas.tool = "eraser"
    canvas.resetTransform()
    before = canvas._make_brush_cursor()[0].width()
    canvas.auto_params["auto_mask_dilate"] = 4
    canvas._rebuild()              # 面板改设置后会走到这里
    after = canvas._make_brush_cursor()[0].width()
    assert after < before, f"改小覆盖扩展后橡皮擦光标应变小（{before} → {after}）"


# ------------------------------------------------------- 5 用户症状的回归锁
# 与用户实测同构：真正被打码的是 r = R + dilate(10) = 60；用户"贴着看得见的边缘"擦，
# 但圆笔刷只有切点落在边缘上、两侧都会差几像素 —— 旧的硬边几何就在那里留下残边。
# 笔迹路径取 R-8 = 42，笔刷半宽 15 → 擦除包络约 r = 57，比打码半径差 3px。
PATH_R = R - 8
PATH_STEP = 6      # 真实鼠标是密集采样的（实测：步距 6px 时旧行为留 1842px 残边，
                   # 步距 10px 的粗台阶会额外产生合成伪影，不代表真实使用）


def test_erasing_along_the_edge_leaves_no_residue(canvas):
    """贴着打码区边缘擦一圈 → 必须一点打码都不剩（这就是用户报的症状）。"""
    _drag_erase(canvas, _stroke_points(PATH_R, step=PATH_STEP))
    residue = _residue(canvas, R + 10)
    assert residue == 0, f"贴着边缘擦完仍残留 {residue} 个打码像素"


def test_residue_reappears_without_the_grow(canvas, monkeypatch):
    """把外扩关掉就应该复现出残边 —— 证明上面那条锁真的咬得住、不是空断言。"""
    monkeypatch.setattr(canvas, "_eraser_grow", lambda: 0)
    _drag_erase(canvas, _stroke_points(PATH_R, step=PATH_STEP))
    residue = _residue(canvas, R + 10)
    assert residue > 300, f"关掉外扩后本应出现明显残边，实测只有 {residue} 像素"


def test_the_residue_is_a_thin_rim(canvas, monkeypatch):
    """残边的形状是"薄薄一圈"（贴在边界、不在内部）——对应用户看到的细碎毛刺。"""
    monkeypatch.setattr(canvas, "_eraser_grow", lambda: 0)
    _drag_erase(canvas, _stroke_points(PATH_R, step=PATH_STEP))
    m = _residue_mask(canvas)
    total = int(m.sum())
    inner = int((m & disc(PATH_R + 10)).sum())     # 笔刷包络盖到的内部（≈r57）
    censored = int(disc(R + 10).sum())
    assert total > 300
    assert inner == 0, f"笔迹覆盖到的内部不该有残边（{inner} 像素）"
    assert total < censored * 0.5, \
        f"残边应只是薄环（{total}px），不该接近整块打码面积（{censored}px）"
