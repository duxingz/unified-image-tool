# -*- coding: utf-8 -*-
"""Unified editor canvas: manual selections, stickers, auto mask overlay."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QLineF, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
)

from ..core.effects import expand_mask
from ..core.image_ops import apply_mask_effect, compose_image, paste_sticker
from ..core.selection import (
    BrushSelection,
    LassoSelection,
    MaskSelection,
    RectSelection,
    Sticker,
    draw_brush_stroke,
)


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    h, w = arr.shape[:2]
    if arr.shape[2] == 3:
        return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def _round_half_up(x: float) -> int:
    """int(x + 0.5) instead of Python's banker's rounding (round(0.5) == 0)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


class EditorCanvas(QGraphicsView):
    state_changed = Signal()
    zoomChanged = Signal(float)  # current zoom relative to fit (1.0 = fit)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self.image_np: np.ndarray | None = None
        self.original_pil: Image.Image | None = None
        self.auto_mask: np.ndarray | None = None
        self.selections: list = []
        self.stickers: list = []
        self.sticker_library: dict = {}
        self.params: dict = {}
        self.show_mask_overlay: bool = False
        self.double_click_fit: bool = False
        self.auto_params: dict = {
            "auto_effect": "blur",
            "auto_strength": 15,
        }
        self.current_effect = "blur"
        self.current_strength = 15
        self.brush_size = 30
        self.tool = "pointer"  # pointer | brush | eraser | rect | lasso
        self._panning = False
        self._last_view_pos: QPointF | None = None
        self._drag_start: QPointF | None = None
        self._drag_points: list = []
        self._rubber_rect: QGraphicsRectItem | None = None
        self._rubber_path_item: QGraphicsPathItem | None = None
        self._last_rebuild_time: float = 0.0
        # Sticker / watermark editing (pointer tool): currently selected sticker
        # index, which handle is being dragged ("move" | "br" | "tl" | ... ),
        # the grab offset in scene px, and geometry captured at drag start
        # (fixed corner / aspect / angle) used by the resize.
        self._sel_sticker: int = -1
        self._sticker_drag: str | None = None
        self._sticker_grab_dx: float = 0.0
        self._sticker_grab_dy: float = 0.0
        self._s_drag: dict | None = None
        # Ctrl+C / Ctrl+V clipboard for sticker duplication (canvas-local, so
        # copying a sticker on one image can be pasted onto another one)
        self._sticker_clipboard = None
        # Where the right button went down. Right-drag pans the canvas, and the
        # context menu arrives on RELEASE, so a drag that actually moved must
        # not pop a menu. Comparing release vs press position is order
        # independent (Qt sends the release/context events in a
        # platform-dependent order, so a "did we pan?" flag would race).
        self._rmb_press_pos: QPointF | None = None

        self._undo_stack: list = []
        self._redo_stack: list = []
        # Ctrl+Z stack: manual edits only (selections, stickers, erase_mask).
        # auto_mask is deliberately NOT included: undoing a manual stroke must
        # not undo an auto-censor result.
        self._manual_undo_stack: list = []
        # Erased regions (True = restored to the original image). Both manual
        # selections and the auto mask are cut by this layer at compose time;
        # the auto mask subtracts it BEFORE the coverage dilation so erased
        # areas are never re-covered.
        self.erase_mask: np.ndarray | None = None
        # Cache of the composed image for committed selections/auto mask/stickers.
        # Brush drag frames render only the in-progress stroke over this base
        # instead of re-composing every committed selection (the old code did
        # N full-image passes per frame, which made painting crawl).
        self._composed_base: np.ndarray | None = None
        # Live-drag state (incremental rendering): how many stroke points have
        # already been rendered into the live image, and the live image itself.
        self._rendered_upto = 0
        self._live_img: np.ndarray | None = None
        # Zoom limits relative to the fit view (set after every load/fit).
        self._fit_scale: float = 1.0
        self._padded: bool = False
        # The mask overlay is kept consistent with compose: expand(auto_mask,
        # dilate) & ~erase_mask, cached once per _rebuild.
        self._overlay_mask: np.ndarray | None = None
        # Hold-to-compare: while True the canvas shows the untouched original
        # (no censor, no stickers, no overlay) instead of the composed image.
        self._show_original: bool = False
        # the brush cursor is sized in screen pixels, so it must be rebuilt on
        # every zoom change (zoomChanged is emitted by fit and _apply_zoom)
        self.zoomChanged.connect(lambda _ratio: self._update_cursor())
        self._update_cursor()

    # ------------------------------------------------------------------
    def set_tool(self, tool: str):
        self.tool = tool
        self._update_cursor()

    def _update_cursor(self):
        if self.tool in ("brush", "eraser"):
            pm, hotspot = self._make_brush_cursor()
            self.setCursor(QCursor(pm, hotspot, hotspot))
        elif self.tool == "rect":
            self.setCursor(Qt.CrossCursor)
        elif self.tool == "lasso":
            self.setCursor(self._make_lasso_cursor())
        elif self.tool == "pointer":
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _eraser_grow(self) -> int:
        """Extra pixels the eraser removes beyond its cursor circle (v1.21.0).

        Why: the auto mask is dilated by ``auto_mask_dilate`` before censoring,
        so the censored area reaches that many pixels past the detection, and a
        round hard-edged stroke can never line up with that hard edge -- erasing
        "along the visible edge" always left ragged 5-20px slivers of mosaic
        (measured on the user's exports; a fixed 2px grow removed only 0.3% of
        them). Tracking the dilation makes one pass along the edge clean.

        Cost: none on already-clean pixels (the eraser only REMOVES censoring,
        it never paints). The only effect is that it also removes censoring up
        to this many pixels outside the circle -- which is why the cursor is
        drawn that much bigger (see ``_make_brush_cursor``)."""
        try:
            return max(0, int(self.auto_params.get("auto_mask_dilate", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _make_brush_cursor(self):
        """Circle cursor sized to what the brush ACTUALLY paints on screen.

        The brush operates in IMAGE pixels, so its on-screen size is
        ``brush_size * zoom``. The old cursor was a fixed 16..64 screen-pixel
        circle, which made it up to 2x too big when zoomed out (users thought
        they had erased far more than they had -- reported as "橡皮擦经常擦不
        干净") and up to 2x too small when zoomed in (v1.14.4 fix).

        For the ERASER the circle shows the real erased area, i.e. the stroke
        plus the ``_eraser_grow()`` ring (v1.21.0)."""
        zoom = self.transform().m11() or 1.0
        image_px = self.brush_size
        if self.tool == "eraser":
            image_px += 2 * self._eraser_grow()
        diameter = int(round(image_px * zoom))
        # keep the pixmap sane: tiny brushes still need a visible ring, and a
        # huge one would allocate a giant cursor (the ring is still drawn at the
        # real size, the clamp only bounds the pixmap)
        size = max(8, min(200, diameter))
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 220), 2))
        r = size // 2
        p.drawEllipse(1, 1, size - 2, size - 2)
        if self.tool == "eraser":
            p.drawLine(r - 5, r - 5, r + 5, r + 5)
            p.drawLine(r - 5, r + 5, r + 5, r - 5)
        p.end()
        return pm, size // 2

    def _make_lasso_cursor(self) -> QCursor:
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(255, 255, 255, 230), 2)
        p.setPen(pen)
        # simple pen icon
        p.drawLine(6, 26, 24, 8)
        p.drawLine(24, 8, 26, 6)
        p.drawLine(20, 12, 22, 14)
        p.drawLine(8, 24, 10, 26)
        p.end()
        return QCursor(pm, 2, 26)

    def set_brush_size(self, size: int):
        self.brush_size = max(2, int(size))
        if self.tool in ("brush", "eraser"):
            self._update_cursor()

    # ------------------------------------------------------------------
    def set_image(self, rgb: np.ndarray):
        self.image_np = rgb.copy()
        self.original_pil = Image.fromarray(rgb).convert("RGB")
        self.auto_mask = np.zeros(rgb.shape[:2], dtype=bool)
        self.erase_mask = np.zeros(rgb.shape[:2], dtype=bool)
        self.selections = []
        self.stickers = []
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._manual_undo_stack.clear()
        self._rebuild(fit=True)
        self.state_changed.emit()

    def clear_image(self):
        """Clear the canvas (used when the last image is removed)."""
        self.image_np = None
        self.original_pil = None
        self.auto_mask = None
        self.erase_mask = None
        self.selections = []
        self.stickers = []
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._manual_undo_stack.clear()
        self._composed_base = None
        self._padded = False
        self._rebuild()
        self.zoomChanged.emit(1.0)

    def set_auto_mask(self, mask: np.ndarray):
        if self.image_np is None:
            return
        self._push_undo()
        self.auto_mask = mask.copy()
        self._rebuild()
        self.state_changed.emit()

    def clear_auto_mask(self):
        if self.image_np is not None:
            self.set_auto_mask(np.zeros(self.image_np.shape[:2], dtype=bool))

    # ------------------------------------------------------------------
    def _push_undo(self):
        if self.image_np is None:
            return
        # masks are shared by reference (copy-on-write): every mutation path
        # REBINDS a fresh array, so the snapshot can never see later changes
        self._undo_stack.append((
            copy.deepcopy(self.selections),
            copy.deepcopy(self.stickers),
            self.auto_mask,
            self.erase_mask,
        ))
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append((
            copy.deepcopy(self.selections),
            copy.deepcopy(self.stickers),
            self.auto_mask,
            self.erase_mask,
        ))
        self.selections, self.stickers, self.auto_mask, self.erase_mask = \
            self._undo_stack.pop()
        self._rebuild()
        self.state_changed.emit()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append((
            copy.deepcopy(self.selections),
            copy.deepcopy(self.stickers),
            self.auto_mask,
            self.erase_mask,
        ))
        self.selections, self.stickers, self.auto_mask, self.erase_mask = \
            self._redo_stack.pop()
        self._rebuild()
        self.state_changed.emit()

    # ------------------------------------------------------------------
    # Ctrl+Z: manual edits only (auto mask untouched)
    # ------------------------------------------------------------------
    def _push_manual_undo(self):
        if self.image_np is None:
            return
        # masks shared by reference, see _push_undo
        self._manual_undo_stack.append((
            copy.deepcopy(self.selections),
            copy.deepcopy(self.stickers),
            self.erase_mask,
        ))
        if len(self._manual_undo_stack) > 30:
            self._manual_undo_stack.pop(0)

    def undo_manual(self):
        """Undo the last manual edit (selection/sticker/erase stroke) only."""
        if not self._manual_undo_stack:
            return
        self.selections, self.stickers, self.erase_mask = \
            self._manual_undo_stack.pop()
        self._rebuild()
        self.state_changed.emit()

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo_manual()
            event.accept()
            return
        # Delete/Backspace removes the SELECTED sticker only (see
        # delete_selected_sticker for why this is not delete_selected()).
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.delete_selected_sticker():
                event.accept()
                return
        # Ctrl+C / Ctrl+V duplicate the selected sticker
        if event.matches(QKeySequence.StandardKey.Copy):
            if self.copy_selected_sticker():
                self.setFocus()
                event.accept()
                return
        if event.matches(QKeySequence.StandardKey.Paste):
            if self.paste_sticker_from_clipboard():
                event.accept()
                return
        # hold SPACE to compare against the pristine original
        if (event.key() == Qt.Key_Space and not event.isAutoRepeat()
                and self.image_np is not None):
            self.set_show_original(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if (event.key() == Qt.Key_Space and not event.isAutoRepeat()
                and self._show_original):
            self.set_show_original(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _make_preview_image(self) -> np.ndarray:
        """Recompute the WYSIWYG image with manual selections, auto masks and stickers."""
        if self.original_pil is None:
            if self.image_np is None:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            return self.image_np

        auto_masks = []
        if self.auto_mask is not None and self.auto_mask.any():
            auto_masks = [MaskSelection(
                self.auto_mask,
                effect=self.auto_params.get("auto_effect", "blur"),
                strength=int(self.auto_params.get("auto_strength", 15)),
            )]

        # Merge config params into auto_params so preview matches export
        # (export spreads config params the same way, see BatchExportWorker).
        composed = compose_image(
            self.original_pil,
            self.selections,
            auto_masks,
            self.stickers,
            self.sticker_library,
            params=self.params,
            auto_params={**self.params, **self.auto_params},
            auto_sticker_rgba=None,
            erase_mask=self.erase_mask,
        )
        return np.array(composed.convert("RGB"))

    # ------------------------------------------------------------------
    def set_show_original(self, on: bool):
        """Hold-to-compare: show the pristine original while held."""
        on = bool(on)
        if self._show_original == on:
            return
        self._show_original = on
        if on and self.original_pil is not None:
            original = np.array(self.original_pil.convert("RGB"))
            self._render_scene(original)
        else:
            self._rebuild()

    def is_showing_original(self) -> bool:
        return self._show_original

    # ------------------------------------------------------------------
    def _apply_scene_rect(self):
        """Content rect when fitted; padded rect when zoomed in.

        The padding is purely a view border: the image can be dragged past
        its edges (exposing the canvas background) so censor strokes can
        start right at the image border. It never touches composition or
        export - the scene is not read by those paths at all.
        """
        if self.image_np is None:
            return
        h, w = self.image_np.shape[:2]
        if self._padded:
            pad = max(60, self.viewport().height() // 5)
            self._scene.setSceneRect(QRectF(-pad, -pad, w + 2 * pad, h + 2 * pad))
        else:
            self._scene.setSceneRect(QRectF(0, 0, w, h))

    def _render_scene(self, composed: np.ndarray, fit: bool = False):
        """Draw the given composed image plus overlays onto the scene."""
        self._scene.clear()
        self._rubber_rect = None
        self._rubber_path_item = None
        if composed is None or self.image_np is None:
            return

        base = QPixmap.fromImage(numpy_to_qimage(composed))
        self._scene.addPixmap(base)
        self._apply_scene_rect()
        if fit:
            # fit against the CONTENT rect only, so the fitted view shows the
            # whole image without any padding/border artifacts
            h, w = self.image_np.shape[:2]
            self.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)

        # Auto mask overlay (optional: the censor effect itself is already
        # painted into the image; the red layer shows the REAL covered area:
        # dilated for coverage plus minus the eraser layer, matching compose)
        # -- hidden while holding "compare original"
        if (self.show_mask_overlay and not self._show_original
                and self._overlay_mask is not None
                and self._overlay_mask.any()):
            h, w = self._overlay_mask.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[self._overlay_mask] = (255, 60, 60, 110)
            self._scene.addPixmap(QPixmap.fromImage(numpy_to_qimage(overlay)))

        # Manual selections: brush leaves no outline; rect/lasso use faint dashed lines
        pen = QPen(QColor(150, 150, 150, 140), 1, Qt.DashLine)
        for sel in ([] if self._show_original else self.selections):
            if isinstance(sel, RectSelection):
                item = self._scene.addRect(
                    sel.x1, sel.y1, sel.x2 - sel.x1, sel.y2 - sel.y1, pen
                )
                item.setZValue(2)
            elif isinstance(sel, LassoSelection):
                if len(sel.points) >= 3:
                    path = QPainterPath()
                    path.moveTo(*sel.points[0])
                    for p in sel.points[1:]:
                        path.lineTo(*p)
                    path.closeSubpath()
                    item = self._scene.addPath(path, pen)
                    item.setZValue(2)
            # BrushSelection intentionally not drawn as an outline

        # Selected sticker / watermark: dashed outline + white corner handles
        # -- hidden while holding "compare original". The outline is a POLYGON
        # through the rotated corners: a plain rect would drift off a rotated
        # sticker as soon as angle != 0 (identical to a rect at angle 0).
        if (self._sel_sticker >= 0 and not self._show_original
                and self._sel_sticker < len(self.stickers)):
            st = self.stickers[self._sel_sticker]
            pts = self._sticker_corners_scene(st)
            if pts:
                sel_pen = QPen(QColor("#FFFFFF"), 1.5, Qt.DashLine)
                path = QPainterPath(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                path.closeSubpath()
                box = self._scene.addPath(path, sel_pen)
                box.setZValue(8)
                hs = self._handle_size()
                hpen = QPen(QColor("#FFFFFF"), 1.5)
                for p in pts:
                    hnd = self._scene.addEllipse(
                        p.x() - hs / 2, p.y() - hs / 2, hs, hs,
                        hpen, QBrush(QColor("#1E88E5"))
                    )
                    hnd.setZValue(9)
                # rotation grip (amber): plain stickers only
                if not st.watermark:
                    rp = self._rotate_handle_pos(st)
                    if rp is not None:
                        top_mid = QPointF((pts[0].x() + pts[1].x()) / 2,
                                          (pts[0].y() + pts[1].y()) / 2)
                        stem = self._scene.addLine(
                            QLineF(top_mid, rp), QPen(QColor("#FFFFFF"), 1.2)
                        )
                        stem.setZValue(8)
                        grip = self._scene.addEllipse(
                            rp.x() - hs / 2, rp.y() - hs / 2, hs, hs,
                            hpen, QBrush(QColor("#FFC107"))
                        )
                        grip.setZValue(9)

    def _rebuild(self, fit: bool = False):
        if self.image_np is None:
            self._scene.clear()
            self._rubber_rect = None
            self._rubber_path_item = None
            self._overlay_mask = None
            return
        composed = self._make_preview_image()
        self._composed_base = composed
        if self.tool == "eraser":
            # the eraser ring is drawn at stroke + auto_mask_dilate, so the
            # cursor must follow that setting (v1.21.0). _rebuild runs right
            # after the panel pushes new auto_params.
            self._update_cursor()
        # overlay mask follows the compose path exactly: dilate first
        # (coverage), then subtract the eraser layer
        if self.auto_mask is not None and self.auto_mask.any():
            m = self.auto_mask.copy()
            dilate = int(self.auto_params.get("auto_mask_dilate", 0))
            if dilate > 0:
                m = expand_mask(m, dilate)
            if self.erase_mask is not None:
                m &= ~self.erase_mask
            self._overlay_mask = m
        else:
            self._overlay_mask = None
        if fit:
            self._padded = False
        self._render_scene(composed, fit)
        if fit:
            self._fit_scale = self.transform().m11()
            self.zoomChanged.emit(1.0)

    def _stroke_new_segment(self):
        """Return (new_points, prev_point) - the not-yet-rendered tail of the
        current stroke, plus the last rendered point for continuity."""
        upto = min(self._rendered_upto, len(self._drag_points))
        new_pts = list(self._drag_points[upto:])
        prev = self._drag_points[upto - 1] if upto > 0 else None
        return new_pts, prev

    def _render_drag_stroke(self):
        """Re-render the in-progress brush stroke from the pre-stroke base.

        Three things make this match the FINAL result (v1.14.4 WYSIWYG fix):

        * the effect is always computed from ``_composed_base`` -- the image as
          it was BEFORE this stroke -- never from ``_live_img``. Rendering onto
          the already-affected live image re-applied the effect to the same
          pixels on every overlapping frame, so the drag showed a much stronger
          result than the release (worst with light/white_mist);
        * the mask comes from ``_stroke_mask`` = the exact geometry the
          committed BrushSelection draws, so the shape cannot change at release;
        * the rendered crop is the WHOLE stroke's bbox (padded), not just the new
          segment. This is required, not an optimisation detail: the pixel-block
          mosaic aligns its grid to the mask's bounding box and the light effect
          derives its radial centre from it, so clipping the mask into a smaller
          crop visibly shifts both (measured 31/255 for mosaic). Re-rendering the
          whole stroke's box each frame also means no earlier frame can leave a
          stale result behind.

        The crop keeps this cheap: for a typical stroke the box is a small
        fraction of the image, and every effect only works inside the mask bbox
        anyway."""
        if self._composed_base is None or not self._drag_points:
            return
        new_pts, _prev = self._stroke_new_segment()
        if not new_pts:
            return
        H, W = self._composed_base.shape[:2]
        pad = self.brush_size + int(self.current_strength) + 8
        xs = [p[0] for p in self._drag_points]
        ys = [p[1] for p in self._drag_points]
        x1 = max(0, int(min(xs)) - pad); y1 = max(0, int(min(ys)) - pad)
        x2 = min(W, int(max(xs)) + pad); y2 = min(H, int(max(ys)) + pad)
        if x2 <= x1 or y2 <= y1:
            self._rendered_upto = len(self._drag_points)
            return
        # Align the crop origin down to a block boundary: the pixel mosaic
        # anchors its grid to absolute block cells, so a cropped mask only
        # renders the same blocks as the full-image commit when the crop
        # origin is a multiple of the block size (pixel effect: block =
        # strength). Other effects are crop-invariant, aligning is harmless.
        block = max(1, int(self.current_strength))
        x1 -= x1 % block
        y1 -= y1 % block
        seg_mask = self._stroke_mask(self._drag_points, self.brush_size,
                                     bbox=(x1, y1, x2, y2))
        # .copy() matters: Image.fromarray may SHARE memory with a contiguous
        # numpy slice, and an in-place effect would then write into the base
        region = self._composed_base[y1:y2, x1:x2].copy()
        out = apply_mask_effect(Image.fromarray(region), seg_mask,
                                self.current_effect, self.current_strength,
                                self.params)
        self._live_img[y1:y2, x1:x2] = np.array(out.convert("RGB"))
        self._rendered_upto = len(self._drag_points)
        self._render_scene(self._live_img)

    def _render_drag_erase(self):
        """Incrementally erase: paste original pixels back over the new
        segment only, and clear the red overlay in that region live (the
        overlay item is updated in place).

        The segment is grown by ``_eraser_grow()`` exactly like the commit in
        ``_apply_erase_points``. Dilation distributes over a union, so growing
        each frame equals growing the finished stroke -- the preview cannot
        drift from the released result (the v1.14.4 lesson)."""
        if self._live_img is None or not self._drag_points:
            return
        new_pts, prev = self._stroke_new_segment()
        if not new_pts:
            return
        if prev is not None:
            new_pts = [prev] + new_pts
        grow = self._eraser_grow()
        xs = [p[0] for p in new_pts]; ys = [p[1] for p in new_pts]
        r = self.brush_size // 2 + 2 + grow
        x1 = max(0, int(min(xs)) - r); y1 = max(0, int(min(ys)) - r)
        x2 = min(self._live_img.shape[1], int(max(xs)) + r)
        y2 = min(self._live_img.shape[0], int(max(ys)) + r)
        if x2 <= x1 or y2 <= y1:
            self._rendered_upto = len(self._drag_points)
            return
        # mask (and its eraser grow) in the crop only: the crop is padded by
        # `grow`, so dilating it is identical to cropping the dilated stroke
        region_mask = self._stroke_mask(new_pts, self.brush_size,
                                        bbox=(x1, y1, x2, y2))
        if grow:
            region_mask = expand_mask(region_mask, grow)
        # image_np IS the pristine original (never written in place), so use
        # it directly: converting original_pil per frame cost a full-image
        # PIL round trip ~30x/s (v1.16.2)
        orig = self.image_np
        # live image: paste original pixels
        self._live_img[y1:y2, x1:x2][region_mask] = orig[y1:y2, x1:x2][region_mask]
        # Stickers are applied AFTER the censor layer in compose and the
        # erase layer never cuts them, so the raw paste above would punch
        # holes into any sticker the stroke crosses (they popped back at
        # release). Redraw the erased part of every sticker reaching into
        # the region as a single blend from the original -- exactly what the
        # release rebuild produces there. No intersecting sticker = no cost.
        hits = []
        if self.stickers and self.sticker_library:
            region_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            for st in self.stickers:
                r = self._sticker_bbox_scene(st)
                if r is not None and r.intersects(region_rect):
                    hits.append(st)
        if hits:
            layer = Image.fromarray(orig[y1:y2, x1:x2].copy())
            for st in hits:
                rect = self._sticker_rect_scene(st)
                if rect is None:
                    continue
                # resolve to the pixel placement first: a watermark copy
                # would make place() key off the REGION size, and only a
                # plain pixel sticker pastes position-independent
                plain = Sticker(
                    st.name, rect.center().x() - x1, rect.center().y() - y1,
                    rect.width(), rect.height(),
                    angle=float(getattr(st, "angle", 0.0) or 0.0),
                    opacity=getattr(st, "opacity", 1.0),
                    flip_h=getattr(st, "flip_h", False),
                    flip_v=getattr(st, "flip_v", False),
                )
                paste_sticker(layer, plain, self.sticker_library)
            stickered = np.array(layer.convert("RGB"))
            reg = self._live_img[y1:y2, x1:x2]
            reg[region_mask] = stickered[region_mask]
        # red overlay: clear the erased region live (subtract the segment).
        # The overlay only exists when an AUTO mask is present -- with
        # manual-only censoring it is None, and unguarded access crashed the
        # whole live preview on every eraser frame (v1.15.0 fix).
        if self._overlay_mask is not None:
            self._overlay_mask[y1:y2, x1:x2] &= ~region_mask
        self._rendered_upto = len(self._drag_points)
        self._render_scene(self._live_img)

    # ------------------------------------------------------------------
    def mouseDoubleClickEvent(self, event):
        if self.double_click_fit and self.image_np is not None:
            if self.itemAt(event.position().toPoint()) is None:
                self._fit_to_content()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def _fit_to_content(self):
        """Restore the fitted view (same as after loading an image)."""
        if self.image_np is None:
            return
        self._padded = False
        h, w = self.image_np.shape[:2]
        self._apply_scene_rect()
        self.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)
        self._fit_scale = self.transform().m11()
        self.zoomChanged.emit(1.0)

    def wheelEvent(self, event: QWheelEvent):
        dy = event.angleDelta().y()
        dx = event.angleDelta().x()
        steps = dy if dy else dx
        if steps == 0:
            event.ignore()
            return
        # vertical wheel (up = zoom in); horizontal wheel: right = zoom in
        factor = 1.15 if steps > 0 else 1 / 1.15
        self._apply_zoom(factor)
        event.accept()

    def _apply_zoom(self, factor: float):
        """Zoom towards the anchor, clamped to [5%, 1600%] of the fit view."""
        if self.image_np is None:
            return
        cur = self.transform().m11()
        fit = self._fit_scale or 1.0
        new = min(max(cur * factor, fit * 0.05), fit * 16.0)
        if abs(new - cur) < 1e-9:
            return
        self.resetTransform()
        self.scale(new, new)
        # padding only while zoomed in; back at/below fit the content rect
        # is restored so the fitted view keeps exact fit (no scrollbars)
        self._padded = new > fit * 1.001
        self._apply_scene_rect()
        self.zoomChanged.emit(new / fit)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._panning = True
            self._last_view_pos = event.position()
            self._rmb_press_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.image_np is not None:
            pos = self.mapToScene(event.position().toPoint())
            if self.tool == "pointer":
                # Sticker editing takes priority over panning: clicking a
                # sticker selects it (or starts a resize when on a handle),
                # clicking empty space pans the canvas as before.
                if not self._begin_sticker_drag(pos):
                    self._panning = True
                    self._last_view_pos = event.position()
                    self.setCursor(Qt.ClosedHandCursor)
            elif self.tool == "brush":
                self._push_undo()
                self._push_manual_undo()
                self._drag_points = [(pos.x(), pos.y())]
                self._rendered_upto = 0
                self._live_img = self._composed_base.copy()
            elif self.tool == "eraser":
                if not self._has_censor_content():
                    return
                self._push_undo()
                self._push_manual_undo()
                self._drag_points = [(pos.x(), pos.y())]
                self._rendered_upto = 0
                self._live_img = self._composed_base.copy()
                self._render_drag_erase()
            elif self.tool in ("rect", "lasso"):
                self._drag_start = pos
                self._drag_points = [(pos.x(), pos.y())]
                if self.tool == "rect":
                    self._rubber_rect = self._scene.addRect(
                        QRectF(pos, pos), QPen(QColor("#00E676"), 2, Qt.DashLine)
                    )
                    self._rubber_rect.setZValue(5)
                elif self.tool == "lasso":
                    self._rubber_path_item = self._scene.addPath(
                        QPainterPath(), QPen(QColor("#00E676"), 2, Qt.DashLine)
                    )
                    self._rubber_path_item.setZValue(6)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            # Scrollbars are in viewport pixels (verified empirically), so
            # the drag must follow the physical mouse displacement - using
            # scene coordinates would shrink it by the zoom factor (the
            # "not responsive when zoomed / hyperactive at 110%" bug).
            pos = event.position()
            delta = pos - self._last_view_pos
            self._last_view_pos = pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - _round_half_up(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - _round_half_up(delta.y())
            )
            event.accept()
            return
        if self.image_np is not None and event.buttons() & Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            if self.tool == "pointer" and self._sticker_drag:
                # Throttle like the brush drags (v1.16.2): every geometry
                # update triggers a FULL recompose (all selections + stickers),
                # and raw mouse events can arrive at 100+ Hz -- on a 2048²
                # image that is 50-200ms per frame. The release applies the
                # release position itself, so the final state is exact.
                now = time.time()
                if now - self._last_rebuild_time > 0.03:
                    self._last_rebuild_time = now
                    self._drag_sticker_to(pos)
                event.accept()
                return
            if self.tool in ("brush", "eraser"):
                self._drag_points.append((pos.x(), pos.y()))
                # throttle measured at entry: a rebuild slower than the budget
                # must not re-arm the timer and trigger every event again
                now = time.time()
                if now - self._last_rebuild_time > 0.03:
                    self._last_rebuild_time = now
                    if self.tool == "brush":
                        self._render_drag_stroke()
                    else:
                        self._render_drag_erase()
            elif self.tool == "rect" and self._drag_start is not None and self._rubber_rect is not None:
                rect = QRectF(self._drag_start, pos).normalized()
                self._rubber_rect.setRect(rect)
            elif self.tool == "lasso":
                self._drag_points.append((pos.x(), pos.y()))
                if self._rubber_path_item is not None:
                    path = QPainterPath()
                    if self._drag_points:
                        path.moveTo(*self._drag_points[0])
                        for p in self._drag_points[1:]:
                            path.lineTo(*p)
                    self._rubber_path_item.setPath(path)
            event.accept()
            return
        # pointer hover feedback: resize cursor on a selected corner handle,
        # open-hand over a sticker (clicks select/drag it)
        if (self.tool == "pointer" and self.image_np is not None
                and not event.buttons()):
            pos = self.mapToScene(event.position().toPoint())
            if self._handle_at(pos):
                self.setCursor(Qt.SizeFDiagCursor)
            elif self._sticker_at(pos) >= 0:
                self.setCursor(Qt.OpenHandCursor)
            elif self.cursor().shape() in (Qt.SizeFDiagCursor, Qt.OpenHandCursor):
                self.setCursor(Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self._last_view_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.image_np is not None:
            pos = self.mapToScene(event.position().toPoint())
            if self.tool == "pointer" and self._sticker_drag:
                # apply the release position itself (the throttle may have
                # skipped the last moves): the final render is exactly under
                # the cursor; a plain click no-ops inside (sub-pixel guard)
                self._drag_sticker_to(pos)
                # committed already during the drag; just end the interaction
                self._sticker_drag = None
                self._s_drag = None
                self._last_view_pos = None
                self.setCursor(Qt.ArrowCursor)
                self._rebuild()
                self.state_changed.emit()
                event.accept()
                return
            if self.tool == "pointer" and self._panning:
                self._panning = False
                self._last_view_pos = None
                self.setCursor(Qt.ArrowCursor)
            elif self.tool == "brush" and self._drag_points:
                pts = self._drag_points
                if len(pts) >= 1:
                    sel = BrushSelection(
                        pts, self.brush_size, effect=self.current_effect,
                        strength=self.current_strength,
                    )
                    self.selections.append(sel)
                    self._punch_through_erase(sel)
                self._drag_points = []
                self._rebuild()
                self.state_changed.emit()
            elif self.tool == "eraser" and self._drag_points:
                self._apply_erase_points(self._drag_points)
                self._drag_points = []
                self._rebuild()
                self.state_changed.emit()
            elif self.tool == "rect" and self._drag_start is not None:
                x1, y1 = self._drag_start.x(), self._drag_start.y()
                x2, y2 = pos.x(), pos.y()
                if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                    sel = RectSelection(
                        int(x1), int(y1), int(x2), int(y2),
                        effect=self.current_effect, strength=self.current_strength,
                    )
                    self._push_undo()
                    self._push_manual_undo()
                    self.selections.append(sel)
                    self._punch_through_erase(sel)
                self._drag_start = None
                self._drag_points = []
                self._rebuild()
                self.state_changed.emit()
            elif self.tool == "lasso" and self._drag_points:
                pts = self._drag_points
                if len(pts) >= 3:
                    sel = LassoSelection(
                        pts, effect=self.current_effect, strength=self.current_strength,
                    )
                    self._push_undo()
                    self._push_manual_undo()
                    self.selections.append(sel)
                    self._punch_through_erase(sel)
                self._drag_points = []
                if self._rubber_path_item is not None:
                    self._scene.removeItem(self._rubber_path_item)
                    self._rubber_path_item = None
                self._rebuild()
                self.state_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Sticker / watermark editing helpers
    # ------------------------------------------------------------------
    def _sticker_rect_scene(self, st) -> QRectF | None:
        """Scene-space rect of a sticker, matching paste_sticker's centered
        placement (watermarks resolved from image size)."""
        if self.image_np is None:
            return None
        h, w = self.image_np.shape[:2]
        src = self.sticker_library.get(st.name) if self.sticker_library else None
        if src is None:
            return None
        if st.watermark:
            st._aspect = src.width / max(1, src.height)
        cx, cy, cw, ch = st.place(w, h)
        cw = max(1, round(cw)); ch = max(1, round(ch))
        return QRectF(cx - cw / 2, cy - ch / 2, cw, ch)

    def _sticker_corners_scene(self, st) -> list[QPointF]:
        """The sticker's four corners (tl, tr, br, bl) in scene space.

        Rotation-aware, and mirrors paste_sticker exactly: PIL's
        ``rotate(-angle)`` turns the layer CLOCKWISE on screen, which for
        image coordinates (y down) maps an offset (dx, dy) to
        ``(dx*cos - dy*sin, dx*sin + dy*cos)``. Verified against a rendered
        pixel; keep this in sync with paste_sticker or the selection outline
        drifts off a rotated sticker."""
        r = self._sticker_rect_scene(st)
        if r is None:
            return []
        cx, cy = r.center().x(), r.center().y()
        a = math.radians(float(getattr(st, "angle", 0.0) or 0.0))
        ca, sa = math.cos(a), math.sin(a)
        hw, hh = r.width() / 2, r.height() / 2
        return [
            QPointF(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
            for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        ]

    def _sticker_bbox_scene(self, st) -> QRectF | None:
        """Axis-aligned bounding box of the (possibly rotated) sticker."""
        pts = self._sticker_corners_scene(st)
        if not pts:
            return None
        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def _rotate_handle_pos(self, st) -> QPointF | None:
        """Scene pos of the rotation grip: off the mid-top edge, along its
        outward normal so it follows the sticker as it turns."""
        pts = self._sticker_corners_scene(st)
        if not pts:
            return None
        mid = QPointF((pts[0].x() + pts[1].x()) / 2, (pts[0].y() + pts[1].y()) / 2)
        cx = sum(p.x() for p in pts) / 4
        cy = sum(p.y() for p in pts) / 4
        dx, dy = mid.x() - cx, mid.y() - cy
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        gap = self._handle_size() * 2.2
        return QPointF(mid.x() + dx / n * gap, mid.y() + dy / n * gap)

    def _handle_size(self) -> float:
        """Scene-space handle size in scene units (~8px, zoom independent)."""
        m = self.transform().m11()
        return 8.0 / max(0.05, m) if m > 1e-9 else 8.0

    def _sticker_at(self, pos) -> int:
        """Top-most sticker whose (rotated) box contains pos, else -1."""
        for i in range(len(self.stickers) - 1, -1, -1):
            r = self._sticker_bbox_scene(self.stickers[i])
            if r is not None and r.adjusted(-4, -4, 4, 4).contains(pos):
                return i
        return -1

    def _handle_at(self, pos) -> str | None:
        """Which grip of the selected sticker is at pos ("tl".."bl" | "rot")."""
        if self._sel_sticker < 0 or self._sel_sticker >= len(self.stickers):
            return None
        st = self.stickers[self._sel_sticker]
        pts = self._sticker_corners_scene(st)
        r = self._sticker_rect_scene(st)
        if not pts or r is None:
            return None
        hs = self._handle_size()
        # never let the hit zone exceed ~20% of the sticker's short side:
        # a degenerate zoom (e.g. window not yet laid out -> tiny transform)
        # must not make every press "hit" a corner handle
        hs = min(hs, min(r.width(), r.height()) * 0.2 + 2.0)
        # watermarks do not rotate: place() resolves an anchored watermark from
        # the image edge, so spinning it would break the anchor contract
        if not st.watermark:
            rp = self._rotate_handle_pos(st)
            if (rp is not None and abs(pos.x() - rp.x()) <= hs * 1.6
                    and abs(pos.y() - rp.y()) <= hs * 1.6):
                return "rot"
        for name, p in zip(("tl", "tr", "br", "bl"), pts):
            if abs(pos.x() - p.x()) <= hs * 1.6 and abs(pos.y() - p.y()) <= hs * 1.6:
                return name
        return None

    def _redraw_overlay(self):
        """Re-render the current scene (overlay follows every rebuild)."""
        if self.image_np is None:
            return
        self._rebuild()

    def _begin_sticker_drag(self, pos) -> bool:
        """Handle press on the pointer tool. True = consumed by sticker edit.

        Undo is NOT pushed here: a plain click must not leave a no-op undo
        record. The snapshot taken at press is pushed on the first actual
        geometry change (see _drag_sticker_to), so click-select stays clean
        and drags are undoable in one step."""
        if not self.stickers:
            return False
        # grip of the currently selected sticker: corner resize or rotation
        if self._sel_sticker >= 0 and self._sel_sticker < len(self.stickers):
            hnd = self._handle_at(pos)
            if hnd:
                st = self.stickers[self._sel_sticker]
                if hnd == "rot":
                    r = self._sticker_rect_scene(st)
                    c = r.center() if r is not None else QPointF(st.x, st.y)
                    self._sticker_drag = "rot"
                    self._s_drag = {
                        "pushed": False,
                        "angle0": float(getattr(st, "angle", 0.0) or 0.0),
                        "grab0": math.degrees(math.atan2(pos.y() - c.y(),
                                                         pos.x() - c.x())),
                    }
                    return True
                self._sticker_drag = hnd
                self._capture_drag_geom(hnd, pos)
                return True
        idx = self._sticker_at(pos)
        if idx >= 0:
            if self._sel_sticker != idx:
                self._sel_sticker = idx
                self._redraw_overlay()
            self._sticker_drag = "move"
            r = self._sticker_rect_scene(self.stickers[idx])
            self._sticker_grab_dx = pos.x() - r.center().x() if r else 0.0
            self._sticker_grab_dy = pos.y() - r.center().y() if r else 0.0
            self._s_drag = {"pushed": False}
            return True
        # empty space: clear the selection
        if self._sel_sticker >= 0:
            self._sel_sticker = -1
            self._redraw_overlay()
        return False

    def _capture_drag_geom(self, handle: str, pos):
        """Freeze the geometry used to compute a resize while the opposite
        corner stays fixed (classic corner-drag behavior)."""
        if self._sel_sticker < 0 or self._sel_sticker >= len(self.stickers):
            return
        st = self.stickers[self._sel_sticker]
        r = self._sticker_rect_scene(st)
        pts = self._sticker_corners_scene(st)
        if r is None or not pts:
            return
        # pts = tl, tr, br, bl -> the corner that stays fixed is the opposite
        # one. At angle 0 this is byte-for-byte the old rect-based behaviour.
        opposite = {"tl": 2, "tr": 3, "bl": 1, "br": 0}
        fix = pts[opposite[handle]] if handle in opposite else r.center()
        self._s_drag = {
            "fix_x": fix.x(), "fix_y": fix.y(),
            "aspect": (st.width / st.height) if st.height else 1.0,
            "angle": float(getattr(st, "angle", 0.0) or 0.0),
            "pushed": False,
        }

    def _push_sticker_undo_once(self):
        if self._s_drag is None or self._s_drag.get("pushed"):
            return
        self._s_drag["pushed"] = True
        self._push_undo()
        self._push_manual_undo()

    def _drag_sticker_to(self, pos):
        """Move or proportionally resize the selected sticker to scene pos."""
        if self._sel_sticker < 0 or self._sel_sticker >= len(self.stickers):
            self._sticker_drag = None
            return
        st = self.stickers[self._sel_sticker]
        if self._sticker_drag == "move":
            r = self._sticker_rect_scene(st)
            if r is None:
                return
            # plain stickers: move the pixel center by the grab offset
            nx = pos.x() - self._sticker_grab_dx
            ny = pos.y() - self._sticker_grab_dy
            if abs(nx - st.x) < 0.5 and abs(ny - st.y) < 0.5:
                return
            self._push_sticker_undo_once()
            st.x = nx
            st.y = ny
            if st.watermark and st.nx is not None:
                # First manual move detaches the anchored placement: place()
                # ignores nx/ny while an anchor is set, so the sticker would
                # snap back to its corner every frame. From now on the
                # position lives in nx/ny fractions (image-size relative).
                st.anchor = "center"
                h, w = self.image_np.shape[:2]
                st.nx = st.x / w if w else st.nx
                st.ny = st.y / h if h else st.ny
            self._redraw_overlay()
            self.state_changed.emit()
            return
        if self._sticker_drag == "rot":
            snap = self._s_drag or {}
            r = self._sticker_rect_scene(st)
            if r is None or "grab0" not in snap:
                return
            c = r.center()
            now = math.degrees(math.atan2(pos.y() - c.y(), pos.x() - c.x()))
            ang = float(snap.get("angle0", 0.0)) + (now - snap["grab0"])
            # 15 deg steps by default, hold Shift for free rotation
            if not (QApplication.keyboardModifiers() & Qt.ShiftModifier):
                ang = round(ang / 15.0) * 15.0
            ang %= 360.0
            if abs(ang - (float(getattr(st, "angle", 0.0) or 0.0))) < 0.01:
                return
            self._push_sticker_undo_once()
            st.angle = ang
            self._redraw_overlay()
            self.state_changed.emit()
            return
        # resize about the captured opposite corner. v1.15.0: corner drag
        # stretches width/height INDEPENDENTLY (free stretch, the pending
        # §9 item); Shift keeps the old proportional behaviour. Watermarks
        # always keep the ratio: place() derives their height from the
        # source aspect, so a stretched height would snap back next frame
        # (same anchor contract that forbids watermark rotation).
        snap = self._s_drag
        if not snap:
            return
        a = math.radians(float(snap.get("angle", 0.0) or 0.0))
        ca, sa = math.cos(a), math.sin(a)
        dxw = pos.x() - snap["fix_x"]
        dyw = pos.y() - snap["fix_y"]
        if abs(dxw) < 1 and abs(dyw) < 1:
            return
        # project the pointer onto the sticker's LOCAL axes so a rotated
        # sticker resizes along its own edges, not along the screen axes
        dx = dxw * ca + dyw * sa
        dy = -dxw * sa + dyw * ca
        aspect = snap["aspect"] if snap["aspect"] else 1.0
        keep_ratio = (bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
                      or st.watermark)
        cw = max(2.0, abs(dx))
        ch = max(2.0, cw / aspect if keep_ratio else abs(dy))
        # clamp to the canvas so a sticker cannot be dragged out of reach
        h, w = self.image_np.shape[:2]
        if keep_ratio:
            scale = min(1.0, (w * 0.9) / cw, (h * 0.9) / ch)
            cw *= scale
            ch *= scale
        else:
            cw = min(cw, w * 0.9)
            ch = min(ch, h * 0.9)
        if abs(cw - st.width) < 0.5 and abs(ch - st.height) < 0.5:
            return
        self._push_sticker_undo_once()
        # keep the fixed corner fixed: step half the local delta, rotated back
        olx = math.copysign(cw, dx) / 2
        oly = math.copysign(ch, dy) / 2
        st.width = cw
        st.height = ch
        st.x = snap["fix_x"] + olx * ca - oly * sa
        st.y = snap["fix_y"] + olx * sa + oly * ca
        if st.watermark:
            h, w = self.image_np.shape[:2]
            st.size_frac = cw / min(w, h) if min(w, h) else st.size_frac
            st.nx = st.x / w if w else st.nx
            st.ny = st.y / h if h else st.ny
        self._redraw_overlay()
        self.state_changed.emit()

    def _has_censor_content(self) -> bool:
        return bool(self.selections) or (
            self.auto_mask is not None and self.auto_mask.any()
        )

    def _stroke_mask(self, points, size, bbox=None) -> np.ndarray:
        """Boolean mask of a brush stroke in image space.

        Uses ``draw_brush_stroke`` -- the SAME geometry the committed
        BrushSelection draws -- so the live preview cannot disagree with the
        finished censor (they used to: circles per sampled point here vs PIL
        line segments there).

        ``bbox`` = (x1, y1, x2, y2) renders only that crop (points are shifted
        by -x1/-y1); the caller gets a mask of the crop's size. Used by the
        incremental preview to stay cheap."""
        w = self.image_np.shape[1]
        h = self.image_np.shape[0]
        if bbox is None:
            x1, y1, x2, y2 = 0, 0, w, h
        else:
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((0, 0), dtype=bool)
        im = Image.new("L", (x2 - x1, y2 - y1), 0)
        draw_brush_stroke(ImageDraw.Draw(im), points, size, origin=(x1, y1))
        return np.array(im) > 127

    def _apply_erase_points(self, points):
        """Commit an erase stroke into the erase layer.

        Rebinds a NEW array instead of ``|=``-mutating: undo/redo snapshots
        SHARE this array (copy-on-write, v1.15.1 -- a 2048² mask costs ~4MB,
        30 snapshots used to copy ~250MB), and an in-place OR would corrupt
        every history entry still pointing at it."""
        if self.erase_mask is None:
            self.erase_mask = np.zeros(self.image_np.shape[:2], dtype=bool)
        stroke = self._stroke_mask(points, self.brush_size)
        grow = self._eraser_grow()
        if grow:
            # eat the ragged rim the round hard stroke leaves against the
            # dilated censoring mask (v1.21.0; mirrors _render_drag_erase)
            stroke = expand_mask(stroke, grow)
        self.erase_mask = self.erase_mask | stroke

    def _punch_through_erase(self, sel):
        """A newly drawn manual censor wins over an EARLIER erase stroke.

        compose_image subtracts the erase layer from manual selections too --
        that is what lets the eraser remove hand-drawn censoring -- but the layer
        carries NO notion of time, so without this an erased spot could never be
        censored again: the stroke got committed and then immediately subtracted
        back out. Reported as "用橡皮擦擦过的地方不能再次手动打码" (v1.14.3).
        """
        if self.erase_mask is None or not self.erase_mask.any():
            return
        if self.image_np is None:
            return
        h, w = self.image_np.shape[:2]
        try:
            area = np.array(sel.to_mask((w, h)).convert("L")) > 127
        except Exception:
            return
        if area.any():
            # rebind (see _apply_erase_points): snapshots share this array
            self.erase_mask = self.erase_mask & ~area

    def delete_selected(self):
        """Remove one censor item: manual selection -> sticker -> auto mask.

        A selected sticker is removed first (or the last one when nothing is
        selected); each click steps back one item so a click on a freshly
        censored image removes the auto mask as a whole; undo brings it back."""
        if self.selections:
            self._push_undo()
            self._push_manual_undo()
            self.selections.pop()
        elif self.stickers:
            self._push_undo()
            self._push_manual_undo()
            if 0 <= self._sel_sticker < len(self.stickers):
                self.stickers.pop(self._sel_sticker)
                self._sel_sticker = -1
            else:
                self.stickers.pop()
        elif self.auto_mask is not None and self.auto_mask.any():
            self._push_undo()
            self.auto_mask = np.zeros(self.image_np.shape[:2], dtype=bool)
        else:
            return
        self._rebuild()
        self.state_changed.emit()

    def delete_selected_sticker(self) -> bool:
        """Delete ONLY the selected sticker (canvas Delete key / context menu).

        Deliberately separate from delete_selected(): that one walks
        manual selection -> sticker -> auto mask, a layered order locked by
        test_v153_features. Pressing Delete on a selected sticker must do the
        obvious thing instead of silently eating a manual selection first."""
        if not (0 <= self._sel_sticker < len(self.stickers)):
            return False
        self._push_undo()
        self._push_manual_undo()
        self.stickers.pop(self._sel_sticker)
        self._sel_sticker = -1
        self._rebuild()
        self.state_changed.emit()
        return True

    def copy_selected_sticker(self) -> bool:
        """Stash the selected sticker on the canvas clipboard."""
        if not (0 <= self._sel_sticker < len(self.stickers)):
            return False
        self._sticker_clipboard = copy.deepcopy(self.stickers[self._sel_sticker])
        return True

    def paste_sticker_from_clipboard(self) -> bool:
        """Paste the clipboard sticker, offset so it does not hide the original."""
        if self._sticker_clipboard is None or self.image_np is None:
            return False
        st = copy.deepcopy(self._sticker_clipboard)
        h, w = self.image_np.shape[:2]
        off = max(8.0, min(w, h) * 0.03)
        st.x += off
        st.y += off
        if getattr(st, "watermark", False) and st.nx is not None:
            st.nx = st.x / w if w else st.nx
            st.ny = st.y / h if h else st.ny
        self._push_undo()
        self._push_manual_undo()
        self.stickers.append(st)
        self._sel_sticker = len(self.stickers) - 1
        self._rebuild()
        self.state_changed.emit()
        return True

    def duplicate_selected_sticker(self) -> bool:
        return self.copy_selected_sticker() and self.paste_sticker_from_clipboard()

    def set_sticker_angle(self, angle: float) -> bool:
        """Set the selected sticker's angle. Watermarks never rotate: place()
        resolves an anchored watermark from the image edge, so spinning it
        would break the anchor contract (documented decision, v1.13)."""
        if not (0 <= self._sel_sticker < len(self.stickers)):
            return False
        st = self.stickers[self._sel_sticker]
        if getattr(st, "watermark", False):
            return False
        ang = float(angle) % 360.0
        if abs(ang - float(getattr(st, "angle", 0.0) or 0.0)) < 0.01:
            return False
        self._push_undo()
        self._push_manual_undo()
        st.angle = ang
        self._rebuild()
        self.state_changed.emit()
        return True

    def nudge_sticker_angle(self, delta: float) -> bool:
        if not (0 <= self._sel_sticker < len(self.stickers)):
            return False
        st = self.stickers[self._sel_sticker]
        return self.set_sticker_angle(float(getattr(st, "angle", 0.0) or 0.0) + delta)

    def flip_selected_sticker(self, horizontal: bool) -> bool:
        """Mirror the selected sticker in place (works for watermarks too:
        mirroring keeps the centre, so anchor placement is unaffected)."""
        if not (0 <= self._sel_sticker < len(self.stickers)):
            return False
        st = self.stickers[self._sel_sticker]
        self._push_undo()
        self._push_manual_undo()
        if horizontal:
            st.flip_h = not getattr(st, "flip_h", False)
        else:
            st.flip_v = not getattr(st, "flip_v", False)
        self._rebuild()
        self.state_changed.emit()
        return True

    def move_selected_sticker(self, to_top: bool) -> bool:
        """Restack the selected sticker. Stickers are drawn in list order, so
        the last one ends up on top."""
        if not (0 <= self._sel_sticker < len(self.stickers)):
            return False
        i = self._sel_sticker
        j = len(self.stickers) - 1 if to_top else 0
        if i == j:
            return False
        self._push_undo()
        self._push_manual_undo()
        st = self.stickers.pop(i)
        self.stickers.insert(j, st)
        self._sel_sticker = j
        self._rebuild()
        self.state_changed.emit()
        return True

    def remove_watermarks(self) -> int:
        """Drop every watermark on the current image as ONE undo step."""
        keep = [st for st in self.stickers if not getattr(st, "watermark", False)]
        removed = len(self.stickers) - len(keep)
        if removed == 0:
            return 0
        self._push_undo()
        self._push_manual_undo()
        self.stickers[:] = keep
        self._sel_sticker = -1
        self._rebuild()
        self.state_changed.emit()
        return removed

    def sticker_context_menu(self, idx: int):
        """Build (but do NOT show) the right-click menu for sticker #idx.

        Split from contextMenuEvent so it is testable: PySide6 resolves
        QMenu.exec from the native method table, so monkeypatching it does not
        work and a test that let the real exec run would block forever."""
        if not (0 <= idx < len(self.stickers)):
            return None
        if self._sel_sticker != idx:
            self._sel_sticker = idx
            self._redraw_overlay()
        st = self.stickers[idx]
        plain = not getattr(st, "watermark", False)
        menu = QMenu(self)
        menu.addAction("复制一个").triggered.connect(self.duplicate_selected_sticker)
        menu.addSeparator()
        a90 = menu.addAction("旋转 90°")
        a90.triggered.connect(lambda: self.nudge_sticker_angle(90.0))
        a90.setEnabled(plain)
        am90 = menu.addAction("旋转 -90°")
        am90.triggered.connect(lambda: self.nudge_sticker_angle(-90.0))
        am90.setEnabled(plain)
        a0 = menu.addAction("角度归零")
        a0.triggered.connect(lambda: self.set_sticker_angle(0.0))
        a0.setEnabled(plain)
        menu.addSeparator()
        menu.addAction("水平翻转").triggered.connect(
            lambda: self.flip_selected_sticker(True))
        menu.addAction("垂直翻转").triggered.connect(
            lambda: self.flip_selected_sticker(False))
        menu.addSeparator()
        menu.addAction("移到最上层").triggered.connect(
            lambda: self.move_selected_sticker(True))
        menu.addAction("移到最下层").triggered.connect(
            lambda: self.move_selected_sticker(False))
        menu.addSeparator()
        menu.addAction("删除该贴纸").triggered.connect(self.delete_selected_sticker)
        return menu

    def contextMenuEvent(self, event):
        """Sticker menu (pointer tool only).

        Right-drag pans the canvas and the menu is delivered on RELEASE, so a
        drag that moved must not pop a menu -- compare release vs press
        position rather than keeping a "did we pan" flag, which would race
        against the platform-dependent release/context event order."""
        if self.image_np is None or self.tool != "pointer":
            super().contextMenuEvent(event)
            return
        if self._rmb_press_pos is not None:
            d = event.pos() - self._rmb_press_pos.toPoint()
            if abs(d.x()) > 4 or abs(d.y()) > 4:
                event.accept()
                return
        idx = self._sticker_at(self.mapToScene(event.pos()))
        if idx < 0:
            super().contextMenuEvent(event)
            return
        menu = self.sticker_context_menu(idx)
        if menu is not None:
            menu.exec(event.globalPos())
        event.accept()

    def add_sticker(self, name, x=None, y=None, width=100, height=100, angle=0):
        if self.image_np is None:
            return
        h, w = self.image_np.shape[:2]
        if x is None:
            x = w / 2
        if y is None:
            y = h / 2
        self._push_undo()
        self._push_manual_undo()
        self.stickers.append(Sticker(name, x, y, width, height, angle))
        self._rebuild()
        self.state_changed.emit()