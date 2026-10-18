# -*- coding: utf-8 -*-
"""v1.16.0 — 「形象版」third theme, LRU bitmap eviction, import dedupe."""
import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.export import compose_from_state
from app.core.selection import RectSelection

W, H = 400, 300


def _win(qapp, tmp_path, monkeypatch):
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow

    class StubDetector:
        def __init__(self, *a, **k):
            pass

        def detect(self, *a, **k):
            return []

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    return MainWindow(StubDetector(), default_config())


def _png(tmp_path, name, w=W, h=H, seed=1):
    p = tmp_path / name
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
                    "RGB").save(p)
    return str(p)


# ==================================================== LRU bitmap eviction

def test_lru_evicts_old_bitmaps_but_keeps_content(qapp, tmp_path, monkeypatch):
    w = _win(qapp, tmp_path, monkeypatch)
    paths = [_png(tmp_path, f"img_{i}.png", seed=i) for i in range(8)]
    w._add_paths(paths)
    for p in paths:                      # visit all 8
        w._load_path(p)
    # one more save: eviction lags one save behind (the save of the current
    # image runs at the START of the next load), so this flips i0/i1 out
    w._load_path(paths[7])
    # the two earliest must have their bitmaps evicted, content intact
    for p in paths[:2]:
        st = w.image_states[p]
        assert "pil" not in st and "rgb" not in st, "old bitmap not evicted"
        assert st["preview_size"] == (W, H)
        assert "orig_size" in st
    for p in paths[2:]:
        assert "pil" in w.image_states[p], "recent bitmap must stay"
    # current image is always kept even if it is the oldest
    w._load_path(paths[0])
    assert "pil" in w.image_states[paths[0]]
    w.close()


def test_evicted_state_reload_restores_masks(qapp, tmp_path, monkeypatch):
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", seed=1)
    b = _png(tmp_path, "b.png", seed=2)
    w._add_paths([a, b])
    w._load_path(a)
    m = np.zeros((H, W), dtype=bool)
    m[40:120, 60:200] = True
    w.canvas.auto_mask = m.copy()
    w._save_current_state()              # records mask + preview_size
    w._load_path(b)                      # switch away
    st = w.image_states[a]
    st.pop("pil", None)
    st.pop("rgb", None)                  # simulate eviction
    w._load_path(a)                      # reopen the evicted image
    assert np.array_equal(w.canvas.auto_mask, m), "mask lost across eviction"
    st = w.image_states[a]
    assert "pil" in st, "reopening must reload the bitmap"
    w.close()


def test_export_of_evicted_state_is_resolution_exact(qapp, tmp_path, monkeypatch):
    """An evicted >2048px image must export at the ORIGINAL resolution with
    the selection rescaled via preview_size (byte-level correctness of the
    LRU feature on the export path)."""
    from pathlib import Path
    w = _win(qapp, tmp_path, monkeypatch)
    src = _png(tmp_path, "big.png", w=1400, h=2600, seed=5)
    w._add_paths([src])
    w._load_path(src)
    # preview is 1102x2048; draw a censor rect in preview coordinates
    sel = RectSelection(100, 100, 300, 200, effect="pixel", strength=10)
    w.canvas.selections = [sel]
    w._save_current_state()
    st = w.image_states[src]
    sx = 1400 / st["preview_size"][0]
    st.pop("pil", None)
    st.pop("rgb", None)                  # evict
    auto = {"auto_effect": "blur", "auto_strength": 15, "auto_mask_dilate": 0}
    img = compose_from_state(src, w.image_states, {}, auto, {})
    assert img.size == (1400, 2600), "export must keep original resolution"
    out = np.array(img.convert("RGB"))
    orig = np.array(Image.open(src).convert("RGB"))
    x1, y1 = int(100 * sx), int(100 * sx * 2600 / 2048 / (1400 / 1102))
    # sample inside the rescaled rect (y scales with the same factor family)
    yx = int(150 * (2600 / 2048))
    xx = int(200 * (1400 / 1102))
    assert np.abs(out[yx, xx].astype(int) - orig[yx, xx].astype(int)).max() > 20, \
        "censor missing at the rescaled location"
    assert np.abs(out[10, 10].astype(int) - orig[10, 10].astype(int)).max() == 0
    assert Path(src).exists()


# ==================================================== import dedupe

def test_duplicate_import_is_skipped(qapp, tmp_path, monkeypatch):
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", seed=1)
    b = _png(tmp_path, "b.png", seed=2)
    c = _png(tmp_path, "c.png", seed=3)
    w._add_paths([a, b])
    w._add_paths([a, c])                 # a is a duplicate
    got = w.thumbnail_list.item_paths()
    assert len(got) == 3, f"expected 3 unique images, got {len(got)}"
    assert str(c) in got and str(b) in got
    w.close()


def test_duplicate_import_is_case_insensitive(qapp, tmp_path, monkeypatch):
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", seed=1)
    w._add_paths([a])
    w._add_paths([a.upper()])            # same file, different case
    assert len(w.thumbnail_list.item_paths()) == 1
    w.close()
