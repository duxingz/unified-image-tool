# -*- coding: utf-8 -*-
"""Export pipeline tests against the pure export_batch() function."""
import numpy as np
import pytest
from PIL import Image

from app.core.config import default_config
from app.core.export import export_batch
from app.core.selection import RectSelection


def make_state(size=(64, 64)):
    pil = Image.new("RGB", size, (100, 120, 140))
    return {
        "pil": pil,
        "rgb": np.array(pil),
        "selections": [RectSelection(10, 10, 30, 30, effect="blur", strength=5)],
        "stickers": [],
        "auto_mask": np.zeros(size, dtype=bool),
    }


def save_cfg(tmp_path, **overrides):
    cfg = {
        "out_dir": str(tmp_path),
        "name_mode": "preserve",
        "prefix": "OUT",
        "start": 1,
        "digits": 3,
        "fmt": "PNG",
        "quality": 95,
        "scale_mode": "none",
        "percent": 100,
        "width": 1000,
        "height": 1000,
        "strip_metadata": False,
    }
    cfg.update(overrides)
    return cfg


def auto_settings(effect="blur", strength=10):
    return {
        "conf": 0.35, "iou": 0.50, "classes": ["vagina", "penis"],
        "auto_effect": effect, "auto_strength": strength,
    }


@pytest.fixture()
def two_images(tmp_path):
    paths = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        Image.new("RGB", (64, 64), (200, 60, 60)).save(p)
        paths.append(str(p))
    return paths


def test_preserve_naming(two_images, tmp_path):
    states = {p: make_state() for p in two_images}
    result = export_batch(two_images, states, default_config(),
                          auto_settings(), save_cfg(tmp_path / "out"), {})
    assert result["processed"] == 2 and result["failed"] == 0
    assert (tmp_path / "out" / "a.png").exists()
    assert (tmp_path / "out" / "b.png").exists()


def test_rename_naming(two_images, tmp_path):
    states = {p: make_state() for p in two_images}
    cfg = save_cfg(tmp_path / "out", name_mode="rename")
    result = export_batch(two_images, states, default_config(),
                          auto_settings(), cfg, {})
    assert result["processed"] == 2
    for name in ("OUT001.png", "OUT002.png"):
        assert (tmp_path / "out" / name).exists()


def test_jpg_format_converts_to_rgb(two_images, tmp_path):
    states = {p: make_state() for p in two_images}
    cfg = save_cfg(tmp_path / "out", fmt="JPG", quality=90)
    result = export_batch(two_images, states, default_config(),
                          auto_settings(), cfg, {})
    assert result["processed"] == 2
    img = Image.open(tmp_path / "out" / "a.jpg")
    assert img.format == "JPEG"
    assert img.mode == "RGB"
    img.close()


def test_percent_scale(two_images, tmp_path):
    states = {p: make_state() for p in two_images}
    cfg = save_cfg(tmp_path / "out", scale_mode="percent", percent=50)
    export_batch(two_images, states, default_config(), auto_settings(), cfg, {})
    img = Image.open(tmp_path / "out" / "a.png")
    assert img.size == (32, 32)
    img.close()


def test_missing_file_reported_as_failed(tmp_path):
    missing = str(tmp_path / "nope.png")
    result = export_batch([missing], {}, default_config(),
                          auto_settings(), save_cfg(tmp_path / "out"), {})
    assert result["failed"] == 1
    assert result["details"][0]["status"] == "失败"


def test_auto_effect_chinese_labels_actually_apply(tmp_path):
    path = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (150, 160, 170)).save(path)
    state = make_state()
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 20:44] = True
    state["auto_mask"] = mask
    states = {str(path): state}

    chinese_cfg = save_cfg(tmp_path / "zh")
    chinese = export_batch([str(path)], states, default_config(),
                           auto_settings(effect="黑色线条", strength=6), chinese_cfg, {})
    english_cfg = save_cfg(tmp_path / "en")
    english = export_batch([str(path)], states, default_config(),
                           auto_settings(effect="black_lines", strength=6), english_cfg, {})
    blur_cfg = save_cfg(tmp_path / "bl")
    blur = export_batch([str(path)], states, default_config(),
                        auto_settings(effect="常规模糊", strength=6), blur_cfg, {})

    assert chinese["processed"] == 1 == english["processed"] == blur["processed"]
    img_zh = np.array(Image.open(tmp_path / "zh" / "src.png"))
    img_en = np.array(Image.open(tmp_path / "en" / "src.png"))
    img_blur = np.array(Image.open(tmp_path / "bl" / "src.png"))
    # Chinese label resolves to the real black-lines effect (matches English id)
    assert np.array_equal(img_zh, img_en)
    # ...and is genuinely different from blur, which a no-op mapping would produce
    assert not np.array_equal(img_zh, img_blur)


def test_cancel_stops_between_images(two_images, tmp_path):
    states = {p: make_state() for p in two_images}
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] >= 2

    result = export_batch(two_images, states, default_config(),
                          auto_settings(), save_cfg(tmp_path / "out"),
                          {}, is_cancelled=cancel)
    assert result["cancelled"] is True
    assert result["processed"] <= 1


def test_compose_from_state_applies_edits(tmp_path):
    from app.core.export import compose_from_state

    p = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (100, 200, 50)).save(p)
    # uniform color would survive a blur unchanged; use a gradient so the
    # edit is actually visible in the pixels
    grad = np.tile(np.linspace(0, 200, 64, dtype=np.uint8), (64, 1))
    pil = Image.fromarray(np.stack([grad, grad // 2, grad // 4], axis=-1))
    state = make_state()
    state["pil"] = pil
    state["rgb"] = np.array(pil)
    state["selections"] = [RectSelection(10, 10, 40, 40, effect="blur", strength=8)]
    states = {str(p): state}
    composed = compose_from_state(
        str(p), states, {}, auto_settings(), default_config().get("params", {})
    )
    assert composed.size == (64, 64)
    base = np.array(pil)
    comp = np.array(composed)
    # edited region differs, untouched corner stays identical to the state image
    assert not np.array_equal(base[20:30, 20:30], comp[20:30, 20:30])
    assert np.array_equal(base[0:8, 0:8], comp[0:8, 0:8])


def test_compose_from_state_without_state_loads_file(tmp_path):
    from app.core.export import compose_from_state

    p = tmp_path / "src.png"
    Image.new("RGB", (40, 20), (5, 5, 5)).save(p)
    composed = compose_from_state(str(p), {}, {}, auto_settings(), {})
    assert composed.size == (40, 20)