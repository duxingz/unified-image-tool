# -*- coding: utf-8 -*-
"""Effect pipeline tests: each effect id produces distinct output, and
Chinese labels from the UI/legacy config normalize to the right effect."""
import numpy as np
import pytest
from PIL import Image

from app.core.image_ops import apply_mask_effect
from app.core.selection import CHINESE_TO_EFFECT, normalize_effect


@pytest.fixture()
def noisy_img():
    rng = np.random.default_rng(42)
    grad = np.tile(np.linspace(0, 200, 80, dtype=np.uint8), (80, 1))
    noise = rng.integers(0, 60, size=(80, 80, 3), dtype=np.uint8)
    arr = np.clip(np.stack([grad, grad // 2, grad // 4], axis=-1) + noise, 0, 255)
    return Image.fromarray(arr)


@pytest.fixture()
def half_mask():
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:60, 20:60] = True
    return mask


def test_normalize_effect_maps_chinese_labels():
    assert normalize_effect("常规模糊") == "blur"
    assert normalize_effect("模糊") == "blur"
    assert normalize_effect("光效马赛克") == "light"
    assert normalize_effect("光效") == "light"
    assert normalize_effect("像素块马赛克") == "pixel"
    assert normalize_effect("黑色线条") == "black_lines"
    assert normalize_effect("白色雾气") == "white_mist"
    assert normalize_effect("自定义图像") == "sticker"


def test_normalize_effect_passthrough():
    assert normalize_effect("blur") == "blur"
    assert normalize_effect(None) == "blur"
    assert normalize_effect("nonsense") == "nonsense"


def test_chinese_to_effect_covers_all_labels():
    # every label present in either UI family must resolve
    for label in ["模糊", "马赛克", "黑色线条", "白色雾气", "光效", "像素块", "贴纸",
                  "常规模糊", "光效马赛克", "自定义图像", "像素块马赛克"]:
        assert label in CHINESE_TO_EFFECT


def test_effects_produce_distinct_outputs(noisy_img, half_mask):
    outputs = {}
    for eff in ["blur", "mosaic", "black_lines", "white_mist", "light"]:
        out = apply_mask_effect(noisy_img, half_mask, eff, strength=10)
        outputs[eff] = np.array(out)
    for a, arr_a in outputs.items():
        for b, arr_b in outputs.items():
            if a != b:
                assert not np.array_equal(arr_a, arr_b), f"{a} == {b}"


def test_mosaic_and_pixel_are_the_same_effect(noisy_img, half_mask):
    # both ids intentionally delegate to the pixel mosaic implementation
    mosaic = apply_mask_effect(noisy_img, half_mask, "mosaic", strength=10)
    pixel = apply_mask_effect(noisy_img, half_mask, "pixel", strength=10)
    assert np.array_equal(np.array(mosaic), np.array(pixel))


def test_chinese_label_equals_english_id(noisy_img, half_mask):
    zh = apply_mask_effect(noisy_img, half_mask, "黑色线条", strength=10)
    en = apply_mask_effect(noisy_img, half_mask, "black_lines", strength=10)
    assert np.array_equal(np.array(zh), np.array(en))


def test_unknown_effect_falls_back_to_blur(noisy_img, half_mask):
    unknown = apply_mask_effect(noisy_img, half_mask, "gibberish", strength=10)
    blur = apply_mask_effect(noisy_img, half_mask, "blur", strength=10)
    assert np.array_equal(np.array(unknown), np.array(blur))


def test_sticker_effect_without_image_returns_copy(noisy_img):
    mask = np.zeros((80, 80), dtype=bool)
    out = apply_mask_effect(noisy_img, mask, "sticker", strength=10)
    assert np.array_equal(np.array(out), np.array(noisy_img))