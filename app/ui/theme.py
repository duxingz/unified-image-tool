# -*- coding: utf-8 -*-
"""Application themes — 小独爱心 / 小独夜 only (v1.19.0).

v1.19.0 removed the two ORIGINAL built-in skins (light/dark) at the user's
request: the app ships exactly two themes, both from the 小独爱心 suite.
This module now only:

1. keeps the METRIC tokens (SPACE/RADIUS/TYPE/CONTROL_PAD) that the widget
   components share with the QSS;
2. re-exports the vendored suite drop-in (``app/ui/theme_xiaoduheart.py``,
   a byte-identical copy of xiaodu-heart-ui/qt/theme_xiaoduheart.py) as the
   ONLY theme engine — its palettes and its QSS template render both skins;
3. maps LEGACY theme names so old configs keep working:
   ``light → xiaodu``、``dark → xiaodunight``、``persona → xiaodu``; any
   unknown name falls back to xiaodu.

Recoloring happens in the SUITE (tokens/tokens.json → web css + qt py →
scripts/check_tokens.py all green → re-copy the drop-in file here); this
module intentionally holds no colors. 只换皮不动布局。
"""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from . import theme_xiaoduheart as _xh

# ---------------------------------------------------------------- metrics
# metric tokens (px) -- shared by the QSS below and by the widget components
SPACE = {"xs": 3, "sm": 6, "md": 10, "lg": 14, "xl": 20}
RADIUS = {"sm": 4, "md": 6, "lg": 8}
TYPE = {"caption": 11, "body": 12, "body_lg": 13, "subtitle": 14, "title": 16}
CONTROL_PAD = "3px 8px"     # keeps the narrow right panel narrowable

# ---------------------------------------------------------------- skins
XIAODU_QSS = _xh.LIGHT_QSS          # 「小独爱心」暖白亮色（默认）
XIAODUNIGHT_QSS = _xh.DARK_QSS      # 「小独夜」暖黑夜色

_THEMES = {
    "xiaodu": _xh.LIGHT,        # 小独爱心（暖白）
    "xiaodunight": _xh.DARK,    # 小独夜（暖黑）
}

# configs written before v1.19.0 carry these names; map them so a dark user
# lands on 小独夜 instead of being forced onto the light skin
_LEGACY_ALIASES = {
    "light": "xiaodu",
    "dark": "xiaodunight",
    "persona": "xiaodu",
}


def _resolve(name: str) -> str:
    """Legacy names → current skins; anything unknown → xiaodu (default)."""
    name = _LEGACY_ALIASES.get(name, name)
    return name if name in _THEMES else "xiaodu"


# Which palette is live, for widgets that paint themselves (QSS cannot reach
# inside a custom paintEvent).
_ACTIVE = {"name": "xiaodu"}


def current_palette() -> dict:
    """The active token dict (xiaodu or xiaodunight)."""
    return _THEMES[_ACTIVE["name"]]


def current_theme() -> str:
    return _ACTIVE["name"]


def theme_qss(name: str) -> str:
    if _resolve(name) == "xiaodunight":
        return XIAODUNIGHT_QSS
    return XIAODU_QSS


def apply_theme(app: QApplication, name: str) -> None:
    """Apply a theme app-wide: 'xiaodu' (小独爱心, default) or
    'xiaodunight' (小独夜). Legacy names ('light'/'dark'/'persona') and
    unknown values resolve through _resolve()."""
    _ACTIVE["name"] = _resolve(name)
    app.setStyleSheet(theme_qss(_ACTIVE["name"]))
