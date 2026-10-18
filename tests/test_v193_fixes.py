# -*- coding: utf-8 -*-
"""v1.9.3 regression tests: the five critical bugs found by the full
source-code review.

1. _load_path failure must not leave current_path pointing at an unloaded
   image (state pollution via the next _save_current_state).
2. export_batch fidelity fast path must honor scale_mode / strip_metadata.
3. _set_auto_running_ui must toggle the real button key (🤖, not ⚡).
4. Settings "恢复默认" must not write config.json until 确定 (取消 = no-op).
5. closeEvent must cancel AND wait for workers; _save_current_state must
   not write image_states while the batch auto-censor worker is running.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.core.config import default_config
from app.core.export import export_batch


@pytest.fixture()
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(
            QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.Yes))


class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[10:34, 10:34] = True
        return [SimpleNamespace(mask=mask, bbox=(10, 10, 34, 34),
                                class_id=3, class_name="vagina",
                                confidence=0.9)]


def make_image(tmp_path, name="img1.png", size=(320, 240), with_meta=False):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[..., 0] = 120
    p = tmp_path / name
    if with_meta:
        meta = PngInfo()
        meta.add_text("Comment", "secret-marker")
        Image.fromarray(arr).save(p, pnginfo=meta)
    else:
        Image.fromarray(arr).save(p)
    return str(p)


def _save_cfg(tmp_path, tag, **over):
    cfg = {
        "out_dir": str(tmp_path / tag), "name_mode": "preserve",
        "prefix": "OUT", "start": 1, "digits": 3, "fmt": "PNG",
        "quality": 100, "scale_mode": "none", "percent": 100,
        "width": 1000, "height": 1000, "strip_metadata": False,
    }
    cfg.update(over)
    return cfg


def _auto():
    return {
        "conf": 0.3, "iou": 0.7, "classes": [], "auto_effect": "blur",
        "auto_strength": 8, "auto_mask_dilate": 0, "input_size": 1024,
    }


class FakeWorker:
    """Duck-typed stand-in with controllable isRunning()."""

    def __init__(self, running=True):
        self._running = running
        self.cancelled = False
        self.waited = None

    def isRunning(self):
        return self._running

    def cancel(self):
        self.cancelled = True

    def wait(self, ms):
        self.waited = ms
        self._running = False
        return True


# ---------------------------------------------------------------- bug 1
def test_load_failure_does_not_pollute_current_path(qapp, tmp_path, no_dialogs):
    from app.ui.main_window import MainWindow
    good = make_image(tmp_path, "good.png")
    bad = str(tmp_path / "missing.png")
    w = MainWindow(StubDetector(), default_config())
    w._load_path(good)
    assert w.current_path == good

    w._load_path(bad)  # 弹窗被 conftest 吞掉
    # 关键断言：失败后 current_path 不能指向未加载的新路径
    assert w.current_path == good
    assert bad not in w.image_states

    # 之前的 bug：失败后画布像素会被 _save_current_state 写到新路径名下
    w._save_current_state()
    assert bad not in w.image_states
    assert "rgb" in w.image_states[good]


# ---------------------------------------------------------------- bug 2
def test_export_fast_path_respects_scale_mode(qapp, tmp_path):
    src = make_image(tmp_path, "src.png", size=(320, 240))
    cfg = _save_cfg(tmp_path, "scaled", scale_mode="percent", percent=50)
    result = export_batch([src], {}, default_config(), _auto(), cfg, {})
    assert result["processed"] == 1
    out = Image.open(tmp_path / "scaled" / "src.png")
    # 快速路径不得绕过缩放：50% 必须真的缩小
    assert out.size == (160, 120)


def test_export_fast_path_respects_strip_metadata(qapp, tmp_path):
    src = make_image(tmp_path, "src.png", with_meta=True)
    assert "Comment" in Image.open(src).info  # 源图确实带元数据
    cfg = _save_cfg(tmp_path, "stripped", strip_metadata=True)
    result = export_batch([src], {}, default_config(), _auto(), cfg, {})
    assert result["processed"] == 1
    out = Image.open(tmp_path / "stripped" / "src.png")
    assert "Comment" not in out.info


def test_export_fast_path_still_byte_exact_without_options(qapp, tmp_path):
    src = make_image(tmp_path, "src.png")
    cfg = _save_cfg(tmp_path, "plain")
    result = export_batch([src], {}, default_config(), _auto(), cfg, {})
    assert result["processed"] == 1
    with open(src, "rb") as f1, open(tmp_path / "plain" / "src.png", "rb") as f2:
        assert f1.read() == f2.read()  # 无缩放/无剥元数据时保持字节级一致


# ---------------------------------------------------------------- bug 3
def test_set_auto_running_ui_toggles_real_button(qapp, tmp_path, no_dialogs):
    from app.ui.main_window import MainWindow
    w = MainWindow(StubDetector(), default_config())
    btn = w.top_buttons.get("🤖 批量自动打码")
    assert btn is not None
    w._set_auto_running_ui(True)
    assert not btn.isEnabled()
    w._set_auto_running_ui(False)
    assert btn.isEnabled()


# ---------------------------------------------------------------- bug 4
def test_restore_defaults_no_disk_write_until_apply(qapp, tmp_path, monkeypatch):
    import app.ui.settings_dialog as sd
    from app.ui.settings_dialog import SettingsDialog
    calls = []
    monkeypatch.setattr(sd, "save_config", lambda cfg: calls.append(dict(cfg)))

    cfg = default_config()
    cfg["auto_strength"] = 5  # 用户改过的非默认值
    dlg = SettingsDialog(cfg, None)
    dlg.auto_strength.setValue(3)

    dlg._restore_defaults()  # QMessageBox.question 被 conftest 置为 Yes

    # 未点确定：不落盘、不改动外部 config 字典，仅控件被重置
    assert calls == []
    assert cfg["auto_strength"] == 5
    assert dlg.auto_strength.value() == 15

    dlg._apply()  # 点确定才写盘
    assert len(calls) == 1
    assert calls[0]["auto_strength"] == 15


# ---------------------------------------------------------------- bug 5
def test_close_event_cancels_and_waits_workers(qapp, tmp_path, no_dialogs,
                                               monkeypatch):
    import app.ui.main_window as mw
    from app.ui.main_window import MainWindow
    monkeypatch.setattr(mw, "save_config", lambda cfg: None)
    good = make_image(tmp_path, "good.png")
    w = MainWindow(StubDetector(), default_config())
    w._load_path(good)
    w.auto_worker = FakeWorker(running=True)
    w.batch_worker = FakeWorker(running=False)
    w.job_worker = FakeWorker(running=True)

    w.close()
    assert w.auto_worker.cancelled and w.auto_worker.waited == 15000
    # 未运行的线程不该被无谓打扰
    assert not w.batch_worker.cancelled and w.batch_worker.waited is None
    assert w.job_worker.cancelled and w.job_worker.waited == 15000


def test_save_current_state_guarded_while_auto_worker_runs(qapp, tmp_path,
                                                           no_dialogs):
    from app.ui.main_window import MainWindow
    good = make_image(tmp_path, "good.png")
    w = MainWindow(StubDetector(), default_config())
    w._load_path(good)
    state = w.image_states[good]

    state["selections"] = ["KEEP"]
    w.canvas.selections = ["NEW"]

    w.auto_worker = FakeWorker(running=True)
    w._save_current_state()
    assert state["selections"] == ["KEEP"]  # worker 写入的 mask 不被画布旧值覆盖

    w.auto_worker = FakeWorker(running=False)
    w._save_current_state()
    assert state["selections"] == ["NEW"]  # worker 结束后恢复正常保存
