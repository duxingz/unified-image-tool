# -*- coding: utf-8 -*-
"""v1.6.6: same-name export collision detection and dedupe.

Regression for the user report: 50+ images (same file names from different
folders) exported with 'preserve original name' silently overwrote each
other, leaving only a few looping images."""
import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.export import export_batch, find_name_conflicts


def _make_pair(src, names, colors):
    paths = []
    for name, color in zip(names, colors):
        p = src / name
        Image.new("RGB", (48, 48), color).save(p)
        paths.append(str(p))
    return paths


def _cfg(out_dir, **kw):
    cfg = {
        "out_dir": str(out_dir), "name_mode": "preserve", "prefix": "OUT",
        "start": 1, "digits": 3, "fmt": "PNG", "quality": 100,
        "scale_mode": "none", "percent": 100, "width": 1000, "height": 1000,
        "strip_metadata": False,
    }
    cfg.update(kw)
    return cfg


def _auto():
    return {"conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
            "auto_strength": 7, "auto_mask_dilate": 0, "input_size": 1024}


def test_find_name_conflicts(tmp_path):
    a = tmp_path / "sub_a"
    b = tmp_path / "sub_b"
    a.mkdir()
    b.mkdir()
    paths = _make_pair(a, ("037.png", "038.png"), [(1, 1, 1), (2, 2, 2)])
    paths += _make_pair(b, ("037.png", "099.png"), [(3, 3, 3), (4, 4, 4)])
    conflicts = find_name_conflicts(paths, "PNG")
    assert set(conflicts) == {"037.png"}
    assert len(conflicts["037.png"]) == 2


def test_dedupe_export_keeps_all_images(tmp_path):
    """THE regression: same-named images from two folders must all be
    exported (with _2/_3 suffixes) instead of overwriting each other."""
    a = tmp_path / "sub_a"
    b = tmp_path / "sub_b"
    a.mkdir()
    b.mkdir()
    paths = _make_pair(a, ("037.png", "038.png"), [(250, 0, 0), (0, 250, 0)])
    paths += _make_pair(b, ("037.png", "099.png"), [(0, 0, 250), (250, 250, 0)])

    cfg = _cfg(tmp_path / "out", on_name_conflict="dedupe")
    result = export_batch(paths, {}, default_config(), _auto(), cfg, {})
    assert result["processed"] == 4

    files = sorted((tmp_path / "out").glob("*.png"))
    assert len(files) == 4  # nothing lost
    names = {f.name for f in files}
    assert {"037.png", "037_2.png", "038.png", "099.png"} == names

    # each output keeps its own content (no overwrite loops)
    import numpy as np
    contents = {}
    for f in (tmp_path / "out").glob("*.png"):
        contents[f.name] = tuple(np.array(Image.open(f))[0, 0])
    assert contents["037.png"] == (250, 0, 0)
    assert contents["037_2.png"] == (0, 0, 250)


def test_overwrite_mode_unchanged(tmp_path):
    """Explicit overwrite (default) keeps the historical behaviour."""
    a = tmp_path / "sub_a"
    b = tmp_path / "sub_b"
    a.mkdir()
    b.mkdir()
    paths = _make_pair(a, ("037.png",), [(250, 0, 0)])
    paths += _make_pair(b, ("037.png",), [(0, 0, 250)])

    cfg = _cfg(tmp_path / "out")  # no on_name_conflict -> overwrite semantics
    result = export_batch(paths, {}, default_config(), _auto(), cfg, {})
    assert result["processed"] == 2
    files = list((tmp_path / "out").glob("*.png"))
    assert len(files) == 1  # second overwrote the first, as requested


def test_rename_mode_needs_no_conflict_check(tmp_path):
    from app.core.export import find_name_conflicts

    a = tmp_path / "sub_a"
    b = tmp_path / "sub_b"
    a.mkdir()
    b.mkdir()
    paths = _make_pair(a, ("037.png", "037-1.png"), [(9, 9, 9), (8, 8, 8)])
    paths += _make_pair(b, ("037.png", "037-1.png"), [(7, 7, 7), (6, 6, 6)])
    # rename mode numbers by list index, so preserve-mode conflicts are moot
    assert find_name_conflicts(paths, "PNG")  # conflict exists in preserve...
    cfg = _cfg(tmp_path / "out", name_mode="rename", prefix="IMG")
    result = export_batch(paths, {}, default_config(), _auto(), cfg, {})
    assert result["processed"] == 4
    assert len(list((tmp_path / "out").glob("*.png"))) == 4