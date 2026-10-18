# -*- coding: utf-8 -*-
"""v1.17.0 — single-instance lock, per-path unsaved-changes tracking."""
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QMessageBox

from app.core.config import default_config
from app.core.export import compose_from_state

W, H = 320, 240


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


def _png(tmp_path, name, seed):
    p = tmp_path / name
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8),
                    "RGB").save(p)
    return str(p)


# ==================================================== single instance lock

def test_second_instance_lock_fails_then_release(tmp_path):
    from app.main import acquire_single_instance_lock
    lock1 = acquire_single_instance_lock(tmp_path)
    assert lock1 is not None, "first acquire must succeed"
    lock2 = acquire_single_instance_lock(tmp_path)
    assert lock2 is None, "second instance must be rejected"
    del lock1                                  # released on GC
    lock3 = acquire_single_instance_lock(tmp_path)
    assert lock3 is not None, "release must allow a fresh acquire"


def test_stale_lock_path_uses_config_dir():
    """The lock lives next to config.json (CONFIG_DIR), not in CWD/temp."""
    import inspect
    from app import main as main_mod
    src = inspect.getsource(main_mod.acquire_single_instance_lock)
    assert "app.lock" in src


# ============================================ per-path unsaved tracking

def _edit_current(w, x1=10, y1=10, x2=60, y2=60):
    from app.core.selection import RectSelection
    w.canvas.selections = w.canvas.selections + [
        RectSelection(x1, y1, x2, y2, effect="blur", strength=5)]
    w.canvas.state_changed.emit()          # marks current path dirty + saves


def test_export_clears_only_exported_paths(qapp, tmp_path, monkeypatch):
    """THE reported bug: exporting/cancelling part of the batch used to
    disarm the close warning for every other edited image."""
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", 1)
    b = _png(tmp_path, "b.png", 2)
    w._add_paths([a, b])
    w._load_path(a)
    _edit_current(w)                        # a dirty
    w._load_path(b)
    _edit_current(w)                        # b dirty
    assert w._dirty_paths == {a, b}

    # export succeeded for a only (e.g. user cancelled at 1/2)
    w._on_batch_done({"processed": 1, "cancelled": True,
                      "exported_paths": [a]})
    assert a not in w._dirty_paths
    assert b in w._dirty_paths, "cancel must NOT disarm the remaining image"

    # the close warning now counts exactly the still-unsaved image
    seen = {}
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: seen.update(t=k.get("text") or a[2])
                                     or QMessageBox.No))
    assert w.close() is False               # No -> stays open
    assert "1 张" in seen.get("t", ""), f"prompt should count 1: {seen}"
    w.close()


def test_full_export_clears_warning_entirely(qapp, tmp_path, monkeypatch):
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", 1)
    w._add_paths([a])
    w._load_path(a)
    _edit_current(w)
    assert w._dirty_paths == {a}
    w._on_batch_done({"processed": 1, "exported_paths": [a]})
    assert w._dirty_paths == set()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    assert w.close() is True


def test_edits_after_export_rearm_warning(qapp, tmp_path, monkeypatch):
    """Export -> tweak the same image again -> warning must come back."""
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", 1)
    w._add_paths([a])
    w._load_path(a)
    _edit_current(w)
    w._on_batch_done({"processed": 1, "exported_paths": [a]})
    _edit_current(w, 100, 100, 200, 200)    # new edit after the export
    assert w._dirty_paths == {a}
    seen = {}
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: seen.update(t=a[2])
                                     or QMessageBox.No))
    w.close()
    assert "1 张" in seen.get("t", "")
    w.close()


def test_cleared_content_does_not_warn(qapp, tmp_path, monkeypatch):
    """A dirty path whose edits were removed (undo / delete) must not warn:
    _unsaved_paths requires the state to still carry censor content."""
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", 1)
    w._add_paths([a])
    w._load_path(a)
    _edit_current(w)
    assert w._dirty_paths == {a}
    st = w.image_states[a]
    st["selections"] = []                    # content cleared (e.g. undo)
    assert w._unsaved_paths() == set()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    assert w.close() is True


def test_job_and_pipeline_results_clear_done_paths(qapp, tmp_path, monkeypatch):
    w = _win(qapp, tmp_path, monkeypatch)
    a = _png(tmp_path, "a.png", 1)
    b = _png(tmp_path, "b.png", 2)
    w._add_paths([a, b])
    w._load_path(a)
    _edit_current(w)
    w._load_path(b)
    _edit_current(w)
    # batch job (convert/resize/rename) saved a out only
    w._on_job_done({"ok": 1, "total": 2, "done_paths": [a]})
    assert a not in w._dirty_paths and b in w._dirty_paths
    # pipeline exported b out
    w._on_pipeline_done({"ok": 1, "total": 1, "done_paths": [b]})
    assert w._dirty_paths == set()


def test_export_worker_reports_exported_paths(qapp, tmp_path, monkeypatch):
    """BatchExportWorker maps details[i] <-> paths[i] so the main window can
    clear exactly the images that really exported."""
    from app.ui.main_window import BatchExportWorker
    a = _png(tmp_path, "a.png", 1)
    b = _png(tmp_path, "b.png", 2)
    w = _win(qapp, tmp_path, monkeypatch)
    w._add_paths([a, b])
    w._load_path(a)
    _edit_current(w)                          # ensure compose path used
    results = []
    worker = BatchExportWorker(
        [a, b], w.image_states, w.config,
        w.tool_panel.auto_settings(),
        {"out_dir": str(tmp_path / "out"), "fmt": "PNG", "quality": 100,
         "name_mode": "preserve", "scale_mode": "none", "percent": 100,
         "strip_metadata": None, "on_name_conflict": "dedupe", "prefix": "IMG",
         "start": 1, "digits": 3},
        {}, parent=None,
    )
    worker.done.connect(results.append)
    worker.run()                              # synchronous in the test
    assert results, "done signal must fire"
    res = results[0]
    assert res["processed"] == 2 and not res["cancelled"]
    assert set(res["exported_paths"]) == {a, b}


def test_exported_files_match_compose(qapp, tmp_path, monkeypatch):
    """Sanity on the worker path: exported bytes equal compose_from_state."""
    a = _png(tmp_path, "a.png", 1)
    from pathlib import Path
    from app.ui.main_window import BatchExportWorker
    w = _win(qapp, tmp_path, monkeypatch)
    w._add_paths([a])
    w._load_path(a)
    _edit_current(w)
    worker = BatchExportWorker(
        [a], w.image_states, w.config, w.tool_panel.auto_settings(),
        {"out_dir": str(tmp_path / "out"), "fmt": "PNG", "quality": 100,
         "name_mode": "preserve", "scale_mode": "none", "percent": 100,
         "strip_metadata": None, "on_name_conflict": "dedupe", "prefix": "IMG",
         "start": 1, "digits": 3},
        {}, parent=None,
    )
    worker.run()
    out = tmp_path / "out" / (Path(a).stem + ".png")
    assert out.exists()
    ref = compose_from_state(a, w.image_states, {},
                             {"auto_effect": "blur", "auto_strength": 15,
                              "auto_mask_dilate": 0}, {})
    assert ref.size == Image.open(out).size
    orig = np.array(Image.open(a).convert("RGB")).astype(int)
    got = np.array(Image.open(out).convert("RGB")).astype(int)
    # the blur censor is present in the export (region differs from source)
    assert np.abs(got[15:55, 15:55] - orig[15:55, 15:55]).max() > 10
