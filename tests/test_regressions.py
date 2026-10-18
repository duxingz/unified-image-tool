# -*- coding: utf-8 -*-
"""Regression guards for bugs fixed in earlier versions.

The worst one: v1.3.3 export was completely broken because the code used
``dlg.Accepted``, which does not exist on PySide6 dialog instances.
These tests statically assert the fix stays in place.
"""
import inspect

from app.ui import main_window

CHECKED_METHODS = ("open_save_dialog", "open_settings")


def test_dialogs_use_dialogcode_accepted():
    for name in CHECKED_METHODS:
        src = inspect.getsource(getattr(main_window.MainWindow, name))
        assert "QDialog.DialogCode.Accepted" in src, name
        assert "dlg.Accepted" not in src, name


def test_export_worker_is_thin_wrapper_over_export_batch():
    src = inspect.getsource(main_window.BatchExportWorker.run)
    assert "export_batch(" in src
    # the old inline loop must not come back
    assert "compose_image(" not in src