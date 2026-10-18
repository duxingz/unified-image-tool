# -*- coding: utf-8 -*-
"""Image composition, effects and conversion helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from . import effects
from .selection import normalize_effect


# Common CJK fonts tried in order when no explicit font_path is given.
# ImageFont.truetype needs an ABSOLUTE path: bare names like "msyh.ttc"
# depend on Pillow's font search, which is unreliable in frozen builds and
# silently degrades to a bitmap font with no CJK glyphs (tofu blocks).
_CJK_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",   # 黑体
    r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    r"C:\Windows\Fonts\Deng.ttf",     # 等线
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def _pick_font(font_size: int, font_path=None):
    """Resolve a TTF/TTC with CJK support to a loadable ImageFont.

    Tries the explicit path first, then well-known system CJK fonts, and
    finally Pillow's built-in default (ASCII-only; used only as last resort
    so rendering never crashes)."""
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(_CJK_FONT_CANDIDATES)
    for p in candidates:
        try:
            if p and Path(p).exists():
                return ImageFont.truetype(p, font_size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(font_size)
    except Exception:
        return ImageFont.load_default()


def render_text_watermark(text: str, font_size: int = 64, color=(255, 255, 255, 255),
                          stroke=0, stroke_fill=(0, 0, 0, 255), opacity=1.0,
                          font_path=None):
    """Render ``text`` into a transparent RGBA layer sized exactly to the
    glyphs, usable as a watermark / sticker image.

    A CJK-capable system font is resolved automatically (absolute font paths,
    never bare names); an empty string returns a 1x1 transparent image."""
    text = (text or "").strip()
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    font = _pick_font(font_size, font_path)
    pad = stroke + 4
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    bbox = d.textbbox((0, 0), text, font=font, stroke_width=stroke)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    w = tw + pad * 2
    h = th + pad * 2
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(layer)
    d2.text((pad - bbox[0], pad - bbox[1]), text, font=font,
            fill=color, stroke_width=stroke, stroke_fill=stroke_fill)
    if opacity < 1.0:
        a = layer.split()[-1].point(lambda v: round(v * opacity))
        layer.putalpha(a)
    return layer


def _pil_to_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _mask_to_bool(mask) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
        return arr > 127
    return np.asarray(mask, dtype=bool)


def apply_mask_effect(
    img: Image.Image,
    mask,
    effect: str = "blur",
    strength: int = 10,
    params: dict | None = None,
    sticker_rgba: np.ndarray | None = None,
) -> Image.Image:
    """Apply a censor effect to a PIL image restricted by a mask."""
    params = params or {}
    effect = normalize_effect(effect)
    rgb = _pil_to_rgb(img)
    mask_bool = _mask_to_bool(mask)

    if effect == "blur":
        radius = max(0.5, float(strength))
        kernel = (int(radius) * 2 + 1, int(radius) * 2 + 1)
        out = effects.apply_blur_mask(
            rgb, mask_bool, kernel_size=kernel,
            alpha=float(params.get("blur_alpha", params.get("alpha", 0.95))),
        )
    elif effect == "mosaic":
        out = effects.apply_pixel_mosaic_mask(
            rgb, mask_bool, block_size=max(1, int(strength))
        )
    elif effect == "pixel":
        out = effects.apply_pixel_mosaic_mask(
            rgb, mask_bool, block_size=max(1, int(strength))
        )
    elif effect == "black_lines":
        out = effects.apply_black_lines_mask(
            rgb, mask_bool,
            line_thickness=int(params.get("line_thickness", strength)),
            spacing=int(params.get("line_spacing", max(4, int(strength)))),
            direction=params.get("line_direction", "horizontal"),
            alpha=float(params.get("line_alpha", params.get("alpha", 1.0))),
        )
    elif effect == "white_mist":
        color = tuple(params.get("mist_color", [255, 255, 255]))
        out = effects.apply_white_mist_mask(
            rgb, mask_bool,
            strength=float(params.get("mist_alpha", params.get("alpha", 0.8))),
            color=color,
        )
    elif effect == "light":
        color = tuple(params.get("light_color", [255, 255, 255]))
        out = effects.apply_light_mask(
            rgb, mask_bool,
            intensity=float(params.get("light_intensity", 0.8)),
            feather=float(params.get("light_feather", 30)),
            color=color,
        )
    elif effect == "sticker":
        if sticker_rgba is None:
            return img.copy()
        out = effects.apply_custom_image_mask(
            rgb, mask_bool, sticker_rgba,
            alpha=float(params.get("custom_alpha", params.get("alpha", 1.0))),
        )
    else:
        radius = max(0.5, float(strength))
        kernel = (int(radius) * 2 + 1, int(radius) * 2 + 1)
        out = effects.apply_blur_mask(
            rgb, mask_bool, kernel_size=kernel,
            alpha=float(params.get("blur_alpha", params.get("alpha", 0.95))),
        )

    return Image.fromarray(out)


def paste_sticker(img, sticker, library):
    """Paste a sticker onto a PIL image.

    Watermark stickers place themselves from the image size (anchor +
    fractions), so the same watermark lands at the same spot (bottom-right
    of a portrait AND a landscape photo)."""
    src = library.get(sticker.name) if library else None
    if src is None:
        return img
    if sticker.watermark and getattr(sticker, "_aspect", None) is None:
        # remember the source aspect ratio so place() can derive the height
        sticker._aspect = src.width / max(1, src.height)
    cx, cy, w, h = sticker.place(img.width, img.height)
    w = max(1, round(w))
    h = max(1, round(h))
    layer = src.resize((w, h), Image.LANCZOS)
    # mirroring first, so a flipped sticker still rotates about its centre
    if getattr(sticker, "flip_h", False):
        layer = layer.transpose(Image.FLIP_LEFT_RIGHT)
    if getattr(sticker, "flip_v", False):
        layer = layer.transpose(Image.FLIP_TOP_BOTTOM)
    if sticker.angle:
        layer = layer.rotate(-sticker.angle, expand=True, resample=Image.BICUBIC)
    # rotate(expand=True) grows the layer; recenter by the actual size
    left = round(cx - layer.width / 2)
    top = round(cy - layer.height / 2)
    if sticker.opacity < 1.0:
        layer = layer.copy()
        if layer.mode in ("RGBA", "LA"):
            a = layer.split()[-1].point(lambda v: round(v * sticker.opacity))
            layer.putalpha(a)
        else:
            layer = layer.convert("RGBA")
            layer.putalpha(layer.split()[-1].point(lambda v: round(v * sticker.opacity)))
    # mask = the layer's (possibly dimmed) alpha: paste fades exactly by opacity
    mask = layer.split()[-1] if layer.mode in ("RGBA", "LA") else None
    img.paste(layer, (left, top), mask)
    layer.close()
    return img


def compose_image(
    original: Image.Image,
    selections,
    auto_masks,
    stickers,
    library,
    params: dict | None = None,
    auto_params: dict | None = None,
    auto_sticker_rgba: np.ndarray | None = None,
    erase_mask: np.ndarray | None = None,
) -> Image.Image:
    """Compose final image: manual censor + auto censor first, then stickers.

    ``erase_mask`` (bool HxW, True = erased) is applied to BOTH the manual
    selections and the auto masks: erased pixels keep the original image.
    The auto mask subtracts it AFTER the coverage dilation, so an erased
    area is never re-covered by the dilate step (dilating after the
    subtraction would leak ~dilate px back across the erase edge)."""
    params = params or {}
    auto_params = auto_params or {}
    erase = erase_mask
    if erase is not None and (erase.shape[1] != original.width
                              or erase.shape[0] != original.height):
        erase = np.array(
            Image.fromarray((erase * 255).astype(np.uint8), "L")
            .resize((original.width, original.height))
        ) > 127
    img = original.copy()

    # Manual selections (censor)
    for sel in selections:
        if not getattr(sel, "applied", True):
            continue
        mask = sel.to_mask(img.size)
        if erase is not None and erase.any():
            mask = np.array(mask.convert("L")) > 127
            mask &= ~erase
        img = apply_mask_effect(
            img, mask, sel.effect, sel.strength, params,
            sticker_rgba=auto_sticker_rgba if sel.effect == "sticker" else None,
        )

    # Auto censor masks
    for am in auto_masks:
        if not getattr(am, "applied", True):
            continue
        mask = am.to_mask(img.size)
        mask_bool = np.array(mask.convert("L")) > 127
        dilate = int(auto_params.get("auto_mask_dilate", 0))
        if dilate > 0:
            # Dilate first so small detection regions still cover their
            # target; then subtract the erase layer so erased areas stay
            # clean no matter how much dilation was applied.
            mask_bool = effects.expand_mask(mask_bool, dilate)
        if erase is not None and erase.any():
            mask_bool &= ~erase
        mask = Image.fromarray((mask_bool.astype(np.uint8) * 255), "L")
        effect = auto_params.get("auto_effect", "blur")
        strength = int(auto_params.get("auto_strength", 15))
        img = apply_mask_effect(
            img, mask, effect, strength, auto_params,
            sticker_rgba=auto_sticker_rgba,
        )

    # Stickers
    for st in stickers:
        img = paste_sticker(img, st, library)

    return img


# ===== Convert / save / resize =====

def save_processed(img, fmt, quality=95, clean_metadata=False, original_info=None):
    if clean_metadata:
        original_info = None
    if fmt == "JPG" or fmt == "JPEG":
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            result = background
        else:
            result = img.convert("RGB")
        return result, "jpg", "JPEG", {"quality": quality}
    if fmt == "WEBP":
        result = img.convert("RGB")
        return result, "webp", "WEBP", {"quality": quality}
    if fmt == "BMP":
        return img, "bmp", "BMP", {}
    if fmt == "GIF":
        return img.convert("P"), "gif", "GIF", {}
    return img, "png", "PNG", {}


def resize_image(img, width=None, height=None, percent=None, keep_ratio=True):
    if percent:
        w = max(1, int(img.width * percent / 100))
        h = max(1, int(img.height * percent / 100))
        return img.resize((w, h), Image.LANCZOS)
    if width and height and keep_ratio:
        ratio = min(width / img.width, height / img.height)
        new_w = max(1, int(img.width * ratio))
        new_h = max(1, int(img.height * ratio))
        return img.resize((new_w, new_h), Image.LANCZOS)
    new_w = width if width else img.width
    new_h = height if height else img.height
    return img.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)


def _exif_upright(img: Image.Image) -> Image.Image:
    """Rotate/flip pixel data according to the EXIF orientation tag.

    Phone photos store e.g. orientation=6 (rotate 90°) with landscape
    pixels; without this the editor canvas shows the image lying on its
    side and every censor/watermark coordinate is based on the wrong
    orientation."""
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def original_size(src_path) -> tuple[int, int]:
    """Pixel size of the source AFTER EXIF orientation is applied.

    Must match what load_original()/load_resized() return, otherwise the
    export-side coordinate rescale (orig_size / preview size) uses a
    swapped width/height for rotated photos."""
    with Image.open(src_path) as im:
        return _exif_upright(im).size


def load_original(src_path, max_edit=0):
    """Load an image at full, un-downsampled resolution (export path).

    Unlike load_resized this never shrinks the image: exporting must preserve
    the real pixel size of the source so the result is not accidentally
    downscaled by the editing viewport's display cap."""
    img = Image.open(src_path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    return _exif_upright(img)


def load_resized(src_path, max_display=900, max_edit=2048):
    img = Image.open(src_path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    img = _exif_upright(img)
    if img.width > max_edit or img.height > max_edit:
        ratio = min(max_edit / img.width, max_edit / img.height)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.LANCZOS)
    display = img
    scale = 1.0
    if img.width > max_display or img.height > max_display:
        scale = min(max_display / img.width, max_display / img.height)
        display = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    return img, display, scale