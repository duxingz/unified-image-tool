# -*- coding: utf-8 -*-
"""Pure export pipeline: compose, name, scale and save a batch of images.

No Qt imports here on purpose - the same code path is exercised by
BatchExportWorker (UI thread wrapper) and by the unit tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .image_ops import (
    compose_image,
    load_original,
    original_size,
    resize_image,
    save_processed,
)
from .numbering import expand_rename_template
from .selection import (
    BrushSelection,
    LassoSelection,
    MaskSelection,
    RectSelection,
    Sticker,
)

EXT_MAP = {"PNG": "png", "JPG": "jpg", "WEBP": "webp", "BMP": "bmp", "GIF": "gif"}


def find_name_conflicts(paths, fmt: str) -> dict:
    """Return {output_name: [paths...]} for images that would collide under
    'preserve original name' mode (same file name from different folders)."""
    ext = EXT_MAP.get(fmt, "png")
    by_name: dict = {}
    for p in paths:
        name = Path(p).stem + "." + ext
        by_name.setdefault(name, []).append(p)
    return {name: ps for name, ps in by_name.items() if len(ps) > 1}


def _scale_selection(sel, sx: float, sy: float):
    """Map a selection from edit-viewport coordinates to original pixels.

    The editor downsizes images above 2048px long edge, so selections are
    recorded at that smaller scale. Exporting at full resolution needs the
    coordinates scaled back up. Mask selections are returned unchanged -
    MaskSelection.to_mask() already resizes to the target image size."""
    if isinstance(sel, RectSelection):
        return RectSelection(
            sel.x1 * sx, sel.y1 * sy, sel.x2 * sx, sel.y2 * sy,
            effect=sel.effect, strength=sel.strength, applied=sel.applied,
        )
    if isinstance(sel, BrushSelection):
        return BrushSelection(
            [(x * sx, y * sy) for x, y in sel.points],
            sel.width * (sx + sy) / 2,
            effect=sel.effect, strength=sel.strength, applied=sel.applied,
        )
    if isinstance(sel, LassoSelection):
        return LassoSelection(
            [(x * sx, y * sy) for x, y in sel.points],
            effect=sel.effect, strength=sel.strength, applied=sel.applied,
        )
    return sel


def _scale_sticker(st: Sticker, sx: float, sy: float) -> Sticker:
    return Sticker(
        st.name, st.x * sx, st.y * sy,
        width=st.width * sx, height=st.height * sy, angle=st.angle,
        opacity=getattr(st, "opacity", 1.0),
        flip_h=getattr(st, "flip_h", False), flip_v=getattr(st, "flip_v", False),
    )


def fmt_from_ext(ext: str) -> str:
    return {"jpg": "JPG", "jpeg": "JPG", "png": "PNG", "webp": "WEBP",
            "bmp": "BMP", "gif": "GIF"}.get(ext.lower(), "PNG")


def state_has_edits(state) -> bool:
    """True when the state carries real censor content (selections, stickers,
    auto mask or erase layer with pixels)."""
    st = state or {}
    return bool(
        st.get("selections") or st.get("stickers")
        or (st.get("auto_mask") is not None and st["auto_mask"].any())
        or (st.get("erase_mask") is not None and st["erase_mask"].any())
    )


def compose_from_state(path, states, sticker_library, auto_settings, params):
    """Compose the edited image for a path from its saved state.

    Pure (no Qt, no widget reads) so it can run in worker threads; callers
    capture ``auto_settings``/``params`` on the main thread first.
    Exports at the ORIGINAL resolution of the source file: the editor may
    have downsampled the preview (2048px cap), but the exported image must
    keep the real pixel size."""
    state = states.get(path)
    if state is None:
        return load_original(path)

    selections = list(state.get("selections", []))
    stickers = list(state.get("stickers", []))

    orig_size = state.get("orig_size")
    base = state.get("pil")
    # `preview_size` is the edit-viewport size recorded at save time; it
    # survives the v1.16.0 LRU eviction (which drops the bitmaps), so the
    # preview->original coordinate rescale works even when `pil` is gone.
    preview_size = state.get("preview_size") or (
        (base.width, base.height) if base is not None else None)
    if orig_size and preview_size:
        orig = load_original(path)
        sx = orig_size[0] / preview_size[0]
        sy = orig_size[1] / preview_size[1]
        if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
            selections = [_scale_selection(s, sx, sy) for s in selections]
            # Plain stickers are stored in preview pixels and scale with the
            # image. Watermark stickers are anchor/fraction based: they are
            # NOT rescaled here but re-placed against the FULL export size by
            # paste_sticker -> sticker.place(), so the same watermark keeps
            # its anchored corner and proportional size at any resolution.
            watermarks = [st for st in stickers if getattr(st, "watermark", False)]
            stickers = ([_scale_sticker(st, sx, sy) for st in stickers
                         if not getattr(st, "watermark", False)]
                        + watermarks)
    elif base is not None:
        orig = base.convert("RGB")
    else:
        orig = load_original(path)

    auto_masks = []
    am = state.get("auto_mask")
    if am is not None and getattr(am, "any", lambda: False)():
        auto_masks = [MaskSelection(
            am,
            effect=auto_settings["auto_effect"],
            strength=auto_settings["auto_strength"],
        )]

    return compose_image(
        orig,
        selections,
        auto_masks,
        stickers,
        sticker_library,
        params=params,
        auto_params={
            "auto_effect": auto_settings["auto_effect"],
            "auto_strength": auto_settings["auto_strength"],
            "auto_mask_dilate": auto_settings.get("auto_mask_dilate", 0),
            **params,
        },
        erase_mask=state.get("erase_mask"),
    )


def export_batch(
    paths,
    states,
    config,
    auto_settings,
    save_cfg,
    sticker_library,
    progress_cb=None,
    status_cb=None,
    is_cancelled=None,
    display_names=None,
) -> dict:
    """Export every image in ``paths`` using its state in ``states``.

    Callbacks (all optional) are invoked from the calling thread:
    - progress_cb(current, total)
    - status_cb(message)
    - is_cancelled() -> bool, checked between images

    Returns the result dict consumed by the UI result dialog:
    {total, processed, skipped, failed, details, cancelled}
    """
    total = len(paths)
    ok = skipped = failed = 0
    details = []
    cancelled = False
    out_dir = save_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    # 'preserve' mode: same-named images (from different folders) would
    # silently overwrite each other unless dedupe is requested
    dedupe = (save_cfg.get("on_name_conflict") == "dedupe"
              and save_cfg["name_mode"] == "preserve")
    used_names: set = set()

    for i, path in enumerate(paths):
        if is_cancelled and is_cancelled():
            cancelled = True
            if status_cb:
                status_cb("已取消")
            break
        if progress_cb:
            progress_cb(i + 1, total)
        if status_cb:
            status_cb(f"正在处理: {Path(path).name}")
        try:
            state = states.get(path)
            edited = bool(
                (state or {}).get("selections") or (state or {}).get("stickers")
                or ((state or {}).get("auto_mask") is not None
                    and state["auto_mask"].any())
                or ((state or {}).get("erase_mask") is not None
                    and state["erase_mask"].any())
            )
            # Fidelity fast path: untouched images are copied byte-for-byte
            # (same format as the source) so size/quality/metadata are 100%
            # preserved - no re-encode, no resample. Only valid when the user
            # did not request a resize or metadata stripping; those options
            # must still take effect on unedited images.
            wants_resize = save_cfg.get("scale_mode", "none") in ("percent", "size")
            wants_strip = bool(save_cfg.get("strip_metadata")) and (
                save_cfg.get("strip_metadata") != "preserve"
            )
            src_ext = Path(path).suffix.lower().lstrip(".")
            if src_ext == "jpeg":
                src_ext = "jpg"
            same_format = src_ext == EXT_MAP.get(save_cfg["fmt"], "png")
            if (not edited and same_format
                    and not wants_resize and not wants_strip):
                img = None
            else:
                img = compose_from_state(
                    path, states, sticker_library, auto_settings,
                    config.get("params", {}),
                )

            ext = EXT_MAP.get(save_cfg["fmt"], "png")
            if save_cfg["name_mode"] == "rename":
                # prefix doubles as a template: {原名} {日期} {时间} {序号}
                name = expand_rename_template(
                    save_cfg["prefix"], path, i,
                    save_cfg["start"], save_cfg["digits"],
                ) + "." + ext
            elif save_cfg["name_mode"] == "display":
                stem = (display_names or {}).get(path) or Path(path).stem
                name = f"{stem}.{ext}"
                if dedupe:
                    n = 2
                    while name in used_names:
                        name = f"{stem}_{n}.{ext}"
                        n += 1
            else:
                stem = Path(path).stem
                name = stem + "." + ext
                if dedupe:
                    n = 2
                    while name in used_names:
                        name = f"{stem}_{n}.{ext}"
                        n += 1
            used_names.add(name)
            dest = os.path.join(out_dir, name)

            if img is None:
                # untouched PNG -> byte-exact copy (no re-encode)
                with open(path, "rb") as fsrc, open(dest, "wb") as fdst:
                    fdst.write(fsrc.read())
            else:
                converted, _, pil_fmt, save_kwargs = save_processed(
                    img, save_cfg["fmt"], save_cfg["quality"],
                    save_cfg["strip_metadata"],
                )
                if save_cfg["scale_mode"] == "percent":
                    converted = resize_image(converted, percent=save_cfg["percent"])
                elif save_cfg["scale_mode"] == "size":
                    converted = resize_image(
                        converted, width=save_cfg["width"], height=save_cfg["height"]
                    )
                converted.save(dest, pil_fmt, **save_kwargs)
                converted.close()
            ok += 1
            details.append({"file": Path(path).name, "status": "成功", "info": dest})
        except Exception as e:
            failed += 1
            details.append({"file": Path(path).name, "status": "失败", "info": str(e)})

    return {
        "total": total, "processed": ok, "skipped": skipped, "failed": failed,
        "details": details, "cancelled": cancelled,
    }

# ----------------------------------------------------------------------
# Batch pipeline (SuperBatch-style action list)
# ----------------------------------------------------------------------
PIPELINE_ACTION_LABELS = {
    "auto_censor": "自动打码",
    "watermark": "加水印",
    "resize": "缩放",
    "convert": "转格式",
    "rename": "重命名",
}


def _ensure_state_loaded(states, path):
    """Load the preview-state for path if absent (thread-safe: no Qt).

    A state that carries `preview_size` + `orig_size` counts as loaded even
    when the v1.16.0 LRU eviction removed its bitmaps: masks/selections are
    still valid and compose_from_state reloads the pixels from disk."""
    state = states.get(path)
    if (state is not None and "orig_size" in state
            and ("pil" in state or "preview_size" in state)):
        return state
    from .image_ops import load_resized
    orig_pil, _, _ = load_resized(path)
    rgb = np.array(orig_pil.convert("RGB"))
    new_state = {
        "pil": orig_pil,
        "selections": [],
        "stickers": [],
        "auto_mask": np.zeros(rgb.shape[:2], dtype=bool),
        "erase_mask": np.zeros(rgb.shape[:2], dtype=bool),
        "orig_size": original_size(path),
        "preview_size": (rgb.shape[1], rgb.shape[0]),
    }
    if state:
        # preserve content appended by earlier batch operations (bitmaps
        # may have been evicted, but everything else is still authoritative)
        new_state["stickers"] = list(state.get("stickers") or [])
        new_state["selections"] = list(state.get("selections") or [])
        prev_auto = state.get("auto_mask")
        if isinstance(prev_auto, np.ndarray):
            new_state["auto_mask"] = prev_auto
        prev_erase = state.get("erase_mask")
        if isinstance(prev_erase, np.ndarray):
            new_state["erase_mask"] = prev_erase
    states[path] = new_state
    return new_state


def describe_pipeline_action(action) -> str:
    """Human-readable one-liner for the action list widget.

    A locked action (光影 SuperBatch parity: its parameters are protected
    from accidental edits) gets a 🔒 prefix; the flag is purely an
    editing guard -- execution reads the same params either way."""
    t = action.get("type")
    p = action.get("params", {})
    lock = "🔒 " if action.get("locked") else ""
    label = PIPELINE_ACTION_LABELS.get(t, t)
    if t == "watermark":
        return f"{lock}加水印: {p.get('asset', '?')} @ {p.get('anchor', 'bottom-right')}"
    if t == "resize":
        if p.get("mode") == "size":
            return f"{lock}缩放: {p.get('width')}x{p.get('height')}"
        return f"{lock}缩放: {p.get('percent', 100)}%"
    if t == "convert":
        q = p.get("quality", 100)
        return f"{lock}转格式: {p.get('fmt', 'PNG')} 质量{q}" + (" 去元数据" if p.get("strip") else "")
    if t == "rename":
        return f"{lock}重命名: {p.get('template', 'IMG')}"
    return f"{lock}{label}"


def run_pipeline_on_image(path, states, actions, sticker_library,
                          auto_settings, params, detector=None, index=0):
    """Execute a pipeline action list for ONE image.

    Content actions (自动打码/加水印) mutate the state; the composed image is
    built once, then output actions (缩放/转格式/重命名) are applied in list
    order. Returns (converted_PIL, pil_fmt, save_kwargs, ext, name_stem)."""
    state = _ensure_state_loaded(states, path)
    name_spec = None
    fmt = quality = strip = None
    resizes = []
    for act in actions:
        t = act.get("type")
        p = act.get("params", {})
        if t == "auto_censor":
            rgb = state.get("rgb")
            if rgb is None:
                rgb = np.array(state["pil"].convert("RGB"))
            if detector is None:
                raise ValueError("自动打码动作需要检测器")
            dets = detector.detect(
                rgb[:, :, ::-1],
                conf_threshold=p.get("conf", 0.3),
                iou_threshold=p.get("iou", 0.7),
                allowed_classes=p.get("classes"),
                input_size=p.get("input_size", 1024),
            )
            union = np.zeros(rgb.shape[:2], dtype=bool)
            for d in dets or []:
                union |= d.mask
            state["auto_mask"] = union
            state["rgb"] = rgb
        elif t == "watermark":
            from .selection import Sticker
            asset = p.get("asset")
            if not asset or asset not in (sticker_library or {}):
                raise ValueError(f"水印素材不存在: {asset}")
            h, w = state["pil"].size[1], state["pil"].size[0]
            short = min(w, h)
            anchor = p.get("anchor", "bottom-right")
            start_nx = 0.5 if anchor == "center" else 1.0
            start_ny = 0.5 if anchor == "center" else 1.0
            state.setdefault("stickers", []).append(Sticker(
                asset, 0, 0, short * p.get("size_frac", 0.15),
                short * p.get("size_frac", 0.15),
                nx=start_nx, ny=start_ny,
                size_frac=p.get("size_frac", 0.15),
                anchor=anchor, opacity=p.get("opacity", 1.0),
                watermark=True,
            ))
        elif t == "resize":
            resizes.append(p)
        elif t == "convert":
            fmt = p.get("fmt", "PNG")
            quality = int(p.get("quality", 100))
            strip = bool(p.get("strip", False))
        elif t == "rename":
            name_spec = (p.get("template", "IMG"), int(p.get("start", 1)),
                         int(p.get("digits", 3)))
        else:
            raise ValueError(f"未知动作类型: {t}")

    img = compose_from_state(path, states, sticker_library, auto_settings,
                             params)
    for rp in resizes:
        if rp.get("mode") == "size":
            img = resize_image(img, width=rp.get("width", 1920),
                               height=rp.get("height", 1080))
        else:
            img = resize_image(img, percent=rp.get("percent", 100))

    if fmt is None:
        fmt, quality, strip = "PNG", 100, False
    converted, _, pil_fmt, save_kwargs = save_processed(img, fmt, quality, strip)
    ext = EXT_MAP.get(fmt, "png")
    if name_spec:
        template, start, digits = name_spec
        name_stem = expand_rename_template(template, path, index, start, digits)
    else:
        name_stem = Path(path).stem
    return converted, pil_fmt, save_kwargs, ext, name_stem
