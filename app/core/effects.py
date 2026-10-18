# -*- coding: utf-8 -*-
"""Mask-based censoring effects."""
from __future__ import annotations

import cv2
import numpy as np

RGB = tuple  # (r, g, b)


def _mask_bounds(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def apply_blur_mask(
    img: np.ndarray,
    mask: np.ndarray,
    kernel_size=(31, 31),
    alpha: float = 1.0,
) -> np.ndarray:
    """Gaussian blur, applied only inside mask (processes only the mask bbox)."""
    if mask.ndim != 2 or not mask.any():
        return img.copy()

    bounds = _mask_bounds(mask)
    if bounds is None:
        return img.copy()
    x1, y1, x2, y2 = bounds

    out = img.copy()
    roi = img[y1:y2, x1:x2].copy()
    blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
    sub_mask = mask[y1:y2, x1:x2]

    if alpha >= 1.0:
        out[y1:y2, x1:x2][sub_mask] = blurred_roi[sub_mask]
    else:
        blended = (
            roi.astype(np.float32) * (1.0 - alpha)
            + blurred_roi.astype(np.float32) * alpha
        ).astype(np.uint8)
        out[y1:y2, x1:x2][sub_mask] = blended[sub_mask]
    return out


def apply_black_lines_mask(
    img: np.ndarray,
    mask: np.ndarray,
    line_thickness: int = 5,
    spacing: int = 10,
    direction: str = "horizontal",
    alpha: float = 1.0,
) -> np.ndarray:
    """Black lines, applied only inside mask."""
    if mask.ndim != 2 or not mask.any():
        return img.copy()

    bounds = _mask_bounds(mask)
    if bounds is None:
        return img.copy()
    x1, y1, x2, y2 = bounds

    lines_layer = img.copy()
    color = (0, 0, 0)
    if direction == "horizontal":
        for y in range(y1, y2, max(1, spacing)):
            cv2.line(lines_layer, (x1, y), (x2, y), color, line_thickness)
    elif direction == "vertical":
        for x in range(x1, x2, max(1, spacing)):
            cv2.line(lines_layer, (x, y1), (x, y2), color, line_thickness)
    elif direction == "diagonal":
        max_range = max(x2 - x1, y2 - y1)
        for i in range(-max_range, max_range, max(1, spacing)):
            x_start = max(x1, x1 + i)
            y_start = max(y1, y1 - i)
            x_end = min(x2, x2 + i)
            y_end = min(y2, y2 - i)
            if x_start < x_end and y_start < y_end:
                cv2.line(lines_layer, (x_start, y_start), (x_end, y_end), color, line_thickness)

    out = img.copy()
    if alpha >= 1.0:
        out[mask] = lines_layer[mask]
    else:
        out[mask] = (
            img[mask].astype(np.float32) * (1.0 - alpha)
            + lines_layer[mask].astype(np.float32) * alpha
        ).astype(np.uint8)
    return out


def apply_white_mist_mask(
    img: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.8,
    color: RGB = (255, 255, 255),
) -> np.ndarray:
    """White mist overlay, applied only inside mask."""
    if mask.ndim != 2 or not mask.any():
        return img.copy()
    out = img.copy()
    color_layer = np.full_like(img, color, dtype=np.uint8)
    out[mask] = (
        img[mask].astype(np.float32) * (1.0 - strength)
        + color_layer[mask].astype(np.float32) * strength
    ).astype(np.uint8)
    return out


def apply_light_mask(
    img: np.ndarray,
    mask: np.ndarray,
    intensity: float = 0.8,
    feather: float = 30.0,
    color: RGB = (255, 255, 255),
) -> np.ndarray:
    """Radial light effect, constrained by mask."""
    if mask.ndim != 2 or not mask.any():
        return img.copy()

    bounds = _mask_bounds(mask)
    if bounds is None:
        return img.copy()
    x1, y1, x2, y2 = bounds
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return img.copy()

    center_x = bw // 2
    center_y = bh // 2
    y_idx, x_idx = np.ogrid[0:bh, 0:bw]
    dist = np.sqrt((x_idx - center_x) ** 2 + (y_idx - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    if max_dist < 1e-6:
        max_dist = 1.0
    radial = np.clip(1.0 - dist / max_dist, 0.0, 1.0)
    radial = np.power(radial, 1.0 / (feather / 100.0 + 0.1))

    out = img.copy()
    roi = img[y1:y2, x1:x2].astype(np.float32)
    color_layer = np.full_like(roi, color, dtype=np.float32)
    weight = radial[..., None] * intensity
    blended = roi * (1.0 - weight) + color_layer * weight
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Write back only where original mask is true
    sub_mask = mask[y1:y2, x1:x2]
    out[y1:y2, x1:x2][sub_mask] = blended[sub_mask]
    return out


def apply_custom_image_mask(
    img: np.ndarray,
    mask: np.ndarray,
    custom_rgba: np.ndarray,
    alpha: float | None = None,
) -> np.ndarray:
    """Custom RGBA image, fills bounding box and clipped by mask."""
    if mask.ndim != 2 or not mask.any():
        return img.copy()

    bounds = _mask_bounds(mask)
    if bounds is None:
        return img.copy()
    x1, y1, x2, y2 = bounds
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return img.copy()

    resized = cv2.resize(custom_rgba, (bw, bh))
    custom_rgb = resized[:, :, :3].astype(np.float32)

    if resized.shape[2] >= 4 and alpha is None:
        custom_a = resized[:, :, 3:4].astype(np.float32) / 255.0
    else:
        a = 1.0 if alpha is None else alpha
        custom_a = np.full((bh, bw, 1), a, dtype=np.float32)

    roi = img[y1:y2, x1:x2].astype(np.float32)
    blended = custom_a * custom_rgb + (1.0 - custom_a) * roi
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out = img.copy()
    sub_mask = mask[y1:y2, x1:x2]
    out[y1:y2, x1:x2][sub_mask] = blended[sub_mask]
    return out


def apply_pixel_mosaic_mask(
    img: np.ndarray,
    mask: np.ndarray,
    block_size: int = 10,
) -> np.ndarray:
    """Pixel block mosaic, applied only inside the mask.

    The grid is anchored to ABSOLUTE block cells (multiples of ``block_size``
    from the image origin), not to the mask bbox: erasing part of a mosaic
    shrinks the bbox, and a bbox-anchored grid would re-phase every remaining
    block at the next rebuild (measured 167/255 on 79% of the remaining
    pixels -- reported as "马赛克残留"). Aligning the processing rect outward
    to whole cells keeps each block's phase and content fixed however the
    mask changes. Masks passed as CROPS must come from a block-aligned crop
    for this to hold (EditorCanvas._render_drag_stroke aligns it)."""
    if mask.ndim != 2 or not mask.any():
        return img.copy()

    bounds = _mask_bounds(mask)
    if bounds is None:
        return img.copy()
    x1, y1, x2, y2 = bounds
    block_size = max(1, block_size)
    H, W = img.shape[:2]
    ax1 = (x1 // block_size) * block_size
    ay1 = (y1 // block_size) * block_size
    ax2 = min(W, -(-x2 // block_size) * block_size)
    ay2 = min(H, -(-y2 // block_size) * block_size)
    bw = max(1, ax2 - ax1)
    bh = max(1, ay2 - ay1)

    roi = img[ay1:ay2, ax1:ax2].copy()
    small = cv2.resize(
        roi,
        (max(1, bw // block_size), max(1, bh // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    pixelated = cv2.resize(
        small, (bw, bh), interpolation=cv2.INTER_NEAREST
    )

    out = img.copy()
    sub_mask = mask[y1:y2, x1:x2]
    grid = pixelated[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1]
    out[y1:y2, x1:x2][sub_mask] = grid[sub_mask]
    return out


def expand_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    """Dilate a boolean mask."""
    if pixels <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def shrink_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    """Erode a boolean mask."""
    if pixels <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1)
    )
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
