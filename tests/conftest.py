# -*- coding: utf-8 -*-
"""Shared pytest fixtures. Adds project root to sys.path so tests can
import the app package, and provides a session-scoped QApplication."""
import os
import sys
from pathlib import Path

# Must be set before PySide6 is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _headless_dialogs(qapp, monkeypatch):
    """Test-env guard: every dialog auto-closes and queued worker signals are
    drained after each test.

    A background worker that finishes AFTER its test leaves a queued done()
    signal; the next test's event loop (QTest.qWait etc.) would dispatch it
    and pop a modal result dialog that hangs the whole run forever."""
    from PySide6.QtWidgets import QDialog, QMessageBox

    monkeypatch.setattr(QDialog, "exec", lambda self: 1, raising=False)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes),
                            raising=False)
    yield
    # drain queued cross-thread signals (worker done/status) after each test
    qapp.processEvents()
    qapp.processEvents()


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Never let a test read or write the REAL %APPDATA% config.

    MainWindow.closeEvent() calls save_config(), so every test that builds and
    closes a window (the whole _mk_window family) would otherwise overwrite the
    user's real settings with defaults. Redirecting the two module globals is
    enough: both are only ever read inside app/core/config.py.

    Tests that pin CONFIG_PATH themselves still win -- their monkeypatch runs
    after this fixture and replaces the temp path with their own.
    """
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")


class AutoStub:
    """Stand-in for MainWindow: attribute access returns a no-op callable."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None