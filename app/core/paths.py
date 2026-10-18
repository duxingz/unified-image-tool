# -*- coding: utf-8 -*-
"""Resource path helper for source and frozen (PyInstaller) modes."""
from __future__ import annotations

import sys
from pathlib import Path


def resource_path(relative: str) -> Path:
    """Return an absolute path to a bundled resource.

    In PyInstaller frozen mode resources are unpacked under sys._MEIPASS.
    In source mode they live under the project root.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / relative
    # app/core/paths.py -> project root
    return Path(__file__).resolve().parent.parent.parent / relative
