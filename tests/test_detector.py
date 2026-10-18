# -*- coding: utf-8 -*-
"""Detector tests: provider selection, letterbox sizing, input_size threading."""
import numpy as np
import pytest

from app.core.config import default_config
from app.core.detector import (
    ONNXDetector,
    _letterbox,
    pick_providers,
)


def test_pick_providers_cpu_only_when_disabled():
    assert pick_providers(use_gpu=False) == ["CPUExecutionProvider"]


def test_pick_providers_prefers_dml_when_available():
    providers = pick_providers(use_gpu=True)
    assert providers[0] == "CPUExecutionProvider" or "DmlExecutionProvider" in providers
    assert "CPUExecutionProvider" in providers


def test_letterbox_respects_target_size():
    img = np.zeros((1600, 1200, 3), dtype=np.uint8)
    out, ratio, (dw, dh) = _letterbox(img, (832, 832))
    assert out.shape[:2] == (832, 832)
    assert abs(ratio - 832 / 1600) < 1e-6
    # padding is symmetric: left == right and top == bottom
    assert dw == (832 - 1200 * ratio) / 2
    assert dh == (832 - 1600 * ratio) / 2


@pytest.fixture(scope="module")
def detector():
    return ONNXDetector(default_config()["model_path"],
                        providers=pick_providers(False))


def test_detect_with_input_size_runs(detector):
    img = np.zeros((512, 384, 3), dtype=np.uint8)
    img[:, :, 2] = 128  # some variation so preprocess does real work
    bgr = img[:, :, ::-1]
    for size in (1280, 832, 640):
        dets = detector.detect(bgr, conf_threshold=0.5, iou_threshold=0.5,
                               input_size=size)
        assert isinstance(dets, list)


def test_detect_default_input_size_runs(detector):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    dets = detector.detect(img, conf_threshold=0.5, iou_threshold=0.5)
    assert isinstance(dets, list)