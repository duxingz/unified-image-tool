# -*- coding: utf-8 -*-
"""Export must preserve the ORIGINAL resolution of source images.

The editor downsamples images above 2048px long edge for display/perf;
exporting used to save that downsampled copy (e.g. 1664x2432 -> 1401x2048).
This locks the full-resolution export + coordinate rescaling."""
import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.export import compose_from_state, export_batch
from app.core.image_ops import load_resized
from app.core.selection import RectSelection

ORIG = (1664, 2432)  # user-facing case: 1664x2432 source


def _make_source(tmp_path, size=ORIG):
    w, h = size
    rng = np.random.default_rng(9)
    grad = np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))
    noise = rng.integers(0, 55, size=(h, w, 3), dtype=np.uint8)
    arr = np.clip(
        np.stack([grad, grad // 2, grad // 4], axis=-1) + noise, 0, 255
    )
    p = tmp_path / "src.png"
    Image.fromarray(arr).save(p)
    return str(p)


def _edited_state(path):
    """What the app keeps in memory after importing the source."""
    pil, _, _ = load_resized(path)  # downsampled edit copy (1401x2048)
    state = {
        "pil": pil,
        "rgb": np.array(pil),
        "selections": [RectSelection(10, 10, 30, 30, effect="blur", strength=8)],
        "stickers": [],
        "auto_mask": np.zeros(pil.size[::-1], dtype=bool),
        "orig_size": ORIG,
    }
    return {path: state}


def _save_cfg(tmp_path, tag):
    return {
        "out_dir": str(tmp_path / tag), "name_mode": "preserve",
        "prefix": "OUT", "start": 1, "digits": 3, "fmt": "PNG",
        "quality": 100, "scale_mode": "none", "percent": 100,
        "width": 1000, "height": 1000, "strip_metadata": False,
    }


def _auto():
    return {
        "conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
        "auto_strength": 8, "auto_mask_dilate": 0, "input_size": 1024,
    }


def test_export_keeps_original_resolution_with_edits(tmp_path):
    path = _make_source(tmp_path)
    states = _edited_state(path)
    # sanity: the edit copy really is downsampled
    assert states[path]["pil"].size == (1401, 2048)
    assert states[path]["pil"].size != ORIG

    cfg = _save_cfg(tmp_path, "out")
    result = export_batch([path], states, default_config(), _auto(), cfg, {})
    assert result["processed"] == 1
    out = Image.open(tmp_path / "out" / "src.png")
    assert out.size == ORIG  # 1664x2432 preserved

    # the manual selection (10..30 at edit scale) lands at ~12..36 px in the
    # full-resolution image: censored there, untouched in the corner
    arr = np.array(out.convert("RGB"))
    base = np.array(Image.open(path))
    sx = ORIG[0] / 1401
    sy = ORIG[1] / 2048
    x1, y1 = int(10 * sx), int(10 * sy)
    x2, y2 = int(30 * sx), int(30 * sy)
    assert not np.array_equal(arr[y1 + 4:y2, x1 + 4:x2], base[y1 + 4:y2, x1 + 4:x2])
    assert np.array_equal(arr[0:8, 0:8], base[0:8, 0:8])
    out.close()


def test_export_without_state_keeps_original_resolution(tmp_path):
    path = _make_source(tmp_path)
    cfg = _save_cfg(tmp_path, "out")
    result = export_batch([path], {}, default_config(), _auto(), cfg, {})
    assert result["processed"] == 1
    with Image.open(tmp_path / "out" / "src.png") as out:
        assert out.size == ORIG


def test_compose_from_state_without_scale_is_identity(tmp_path):
    """A source below the 2048 cap must not be touched by the rescaler."""
    path = _make_source(tmp_path, size=(800, 600))
    pil, _, _ = load_resized(path)
    states = {path: {
        "pil": pil, "selections": [], "stickers": [],
        "auto_mask": np.zeros((600, 800), dtype=bool),
        "orig_size": (800, 600),
    }}
    img = compose_from_state(path, states, {}, _auto(), {})
    assert img.size == (800, 600)