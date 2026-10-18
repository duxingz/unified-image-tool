# -*- coding: utf-8 -*-
"""Unified selection/sticker/mask models."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# Effect ids used by manual selections and auto censor
EFFECTS = [
    "blur",
    "mosaic",
    "black_lines",
    "white_mist",
    "light",
    "pixel",
    "sticker",
]

# Chinese display labels (both families: manual panel + legacy MOSAIC_TYPES) -> effect ids
CHINESE_TO_EFFECT = {
    "模糊": "blur",
    "常规模糊": "blur",
    "马赛克": "mosaic",
    "黑色线条": "black_lines",
    "白色雾气": "white_mist",
    "光效": "light",
    "光效马赛克": "light",
    "像素块": "pixel",
    "像素块马赛克": "pixel",
    "贴纸": "sticker",
    "自定义图像": "sticker",
}


# Auto-censor effect choices offered in UI: (effect id, Chinese label).
# "sticker" is excluded: the auto pipeline has no sticker image source,
# selecting it would silently do nothing.
AUTO_EFFECT_CHOICES = [
    ("blur", "常规模糊"),
    ("black_lines", "黑色线条"),
    ("white_mist", "白色雾气"),
    ("light", "光效马赛克"),
    ("pixel", "像素块马赛克"),
]


def normalize_effect(name) -> str:
    """Map a Chinese effect label or legacy id to a canonical effect id.

    Unknown values are returned unchanged (apply_mask_effect falls back to
    blur for unknown ids, preserving the historical behavior)."""
    if name is None:
        return "blur"
    name = str(name).strip()
    return CHINESE_TO_EFFECT.get(name, name)


class Selection:
    """Manual censor selection base."""

    def __init__(self, effect="blur", strength=10, applied=True):
        self.effect = effect
        self.strength = strength
        self.applied = applied

    def to_mask(self, size) -> Image.Image:
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        self._draw_on(draw)
        return mask

    def _draw_on(self, draw):
        raise NotImplementedError


class RectSelection(Selection):
    def __init__(self, x1, y1, x2, y2, effect="blur", strength=10, applied=True):
        super().__init__(effect, strength, applied)
        self.type = "rect"
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    def _draw_on(self, draw):
        draw.rectangle((self.x1, self.y1, self.x2, self.y2), fill=255)


def draw_brush_stroke(draw, points, width, origin=(0, 0)):
    """Draw one brush stroke onto a PIL ImageDraw.

    THE single definition of a stroke's shape (v1.14.4). It used to exist twice
    -- the canvas preview ORed a circle per sampled point, while the committed
    BrushSelection drew PIL line segments with end caps -- so the live preview
    and the finished censor were literally different shapes (a black_lines or
    pixel stroke visibly changed at mouse-release).

    ``origin`` shifts the points, so a caller can draw into a crop.

    **Round joins (v1.21.0)**: ``ImageDraw.line`` has BUTT ends, so a turn left a
    notch on the outside of the corner. For the eraser that notch is a small
    crescent of censoring that survives the stroke -- reported as "擦完留一圈
    细碎毛刺". Stamping a disc at every sample makes the shape the true
    Minkowski sum of the path and the round brush.
    """
    if not points:
        return
    ox, oy = origin
    w = max(2, int(width))
    r = w // 2

    def disc(x, y):
        draw.ellipse((x - r - ox, y - r - oy, x + r - ox, y + r - oy), fill=255)

    if len(points) == 1:
        disc(*points[0])
        return
    for i in range(len(points) - 1):
        (ax, ay), (bx, by) = points[i], points[i + 1]
        draw.line([(ax - ox, ay - oy), (bx - ox, by - oy)], fill=255, width=w)
    for x, y in points:
        disc(x, y)


class BrushSelection(Selection):
    def __init__(self, points, width, effect="blur", strength=10, applied=True):
        super().__init__(effect, strength, applied)
        self.type = "brush"
        self.points = points
        self.width = width

    def _draw_on(self, draw):
        draw_brush_stroke(draw, self.points, self.width)


class LassoSelection(Selection):
    def __init__(self, points, effect="blur", strength=10, applied=True):
        super().__init__(effect, strength, applied)
        self.type = "lasso"
        self.points = points

    def _draw_on(self, draw):
        if len(self.points) >= 3:
            draw.polygon(self.points, fill=255)


class MaskSelection(Selection):
    """A selection that directly holds a boolean mask (for auto censor or custom mask)."""

    def __init__(self, mask, effect="blur", strength=10, applied=True):
        super().__init__(effect, strength, applied)
        self.type = "mask"
        self.mask = mask  # numpy bool array (H, W) or PIL L image

    def to_mask(self, size) -> Image.Image:
        if isinstance(self.mask, Image.Image):
            return self.mask.convert("L").resize(size)
        arr = (self.mask * 255).astype(np.uint8)
        return Image.fromarray(arr, "L").resize(size)


class Sticker:
    def __init__(self, name, x, y, width=100, height=100, angle=0.0,
                 nx=None, ny=None, size_frac=None, anchor="center",
                 opacity=1.0, watermark=False, flip_h=False, flip_v=False):
        """A pasted image element (library sticker or watermark).

        Coordinates are stored in pixels (x, y center + width/height), the
        format the compose path has always used. Watermarks additionally
        carry anchor + fractions so their placement RESIZES WITH THE IMAGE:
        ``nx``/``ny`` are the center as a fraction of image width/height,
        ``size_frac`` the width as a fraction of the shorter side. When these
        are set (watermark mode) the pixel values are DERIVED from the image
        size at render time and can move when the image size changes.
        """
        self.type = "sticker"
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.nx = nx
        self.ny = ny
        self.size_frac = size_frac
        self.anchor = anchor
        self.opacity = float(opacity)
        self.watermark = bool(watermark)
        # mirroring, applied before the rotation in paste_sticker
        self.flip_h = bool(flip_h)
        self.flip_v = bool(flip_v)

    @property
    def applied(self):
        return True

    def is_watermark(self) -> bool:
        return self.watermark and self.nx is not None and self.size_frac is not None

    def place(self, img_w, img_h):
        """Resolve this sticker to concrete center/size for an image.

        Watermark mode: nx/ny are fractions of (width, height); size_frac is
        a fraction of the shorter side; anchor picks the placement relative
        to the image edge (anchored stickers ignore nx/ny). Plain stickers
        keep their stored pixel x/y/width/height."""
        if not self.is_watermark():
            return self.x, self.y, self.width, self.height
        short = min(img_w, img_h)
        w = max(1.0, short * self.size_frac)
        lib = getattr(self, "_aspect", None)
        h = w / lib if lib else w
        if self.anchor == "center":
            # "center" with nx/ny set means a FREE position (the state after
            # a manual drag): follow the stored fractions, not the middle
            if self.nx is not None and self.ny is not None:
                cx, cy = self.nx * img_w, self.ny * img_h
            else:
                cx, cy = img_w / 2, img_h / 2
        elif self.anchor == "top-left":
            cx, cy = w / 2, h / 2
        elif self.anchor == "top-right":
            cx, cy = img_w - w / 2, h / 2
        elif self.anchor == "bottom-left":
            cx, cy = w / 2, img_h - h / 2
        elif self.anchor == "bottom-right":
            cx, cy = img_w - w / 2, img_h - h / 2
        elif self.anchor == "top-center":
            cx, cy = img_w / 2, h / 2
        elif self.anchor == "bottom-center":
            cx, cy = img_w / 2, img_h - h / 2
        elif self.anchor == "left-center":
            cx, cy = w / 2, img_h / 2
        elif self.anchor == "right-center":
            cx, cy = img_w - w / 2, img_h / 2
        else:  # fallback: center
            cx, cy = img_w / 2, img_h / 2
        return cx, cy, w, h