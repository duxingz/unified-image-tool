# -*- coding: utf-8 -*-
"""Configuration and presets for Automatic Coding 2."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .paths import resource_path

APP_NAME = "AutomaticCoding2"
CONFIG_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME
CONFIG_PATH = CONFIG_DIR / "config.json"

# Marks a config file as produced by this UI version. Config files written by
# older versions are migrated once so bundled default-value changes actually
# take effect for existing users.
UI_VERSION = "1.14.5"

# Model class internal keys -> display names (vagina is shown as pussy)
CLASS_KEYS = ["nipple", "vagina", "penis", "anus", "pubic_hair"]
CLASS_DISPLAY = {
    "nipple": "nipple（乳头）",
    "vagina": "pussy（阴部）",
    "penis": "penis（阴茎）",
    "anus": "anus（肛门）",
    "pubic_hair": "pubic hair（阴毛）",
}
IGNORED_CLASS_KEYS = ["female_face", "male_face"]
# v1.22.0：右侧面板的短标签（两列网格放得下；完整双语名仍在 CLASS_DISPLAY，
# 用作 tooltip，避免 "pubic hair（阴毛）" 这种长标签把两列挤成一列）。
CLASS_LABELS = {
    "nipple": "乳头",
    "vagina": "阴部",
    "penis": "阴茎",
    "anus": "肛门",
    "pubic_hair": "阴毛",
}

MOSAIC_TYPES = [
    "常规模糊",
    "黑色线条",
    "白色雾气",
    "光效马赛克",
    "自定义图像",
    "像素块马赛克",
]


def default_config() -> dict:
    return {
        "ui_version": UI_VERSION,
        "model_path": str(resource_path("models/nsfw-anime-medium-x1280.onnx")),
        "conf_threshold": 0.3,
        "iou_threshold": 0.70,
        "selected_regions": ["vagina", "penis"],
        "auto_effect": "blur",
        "auto_strength": 15,
        "auto_mask_dilate": 5,
        "show_mask_overlay": True,
        "last_rename": {
            "name_mode": "preserve",
            "prefix": "IMG",
            "start": 1,
            "digits": 3,
        },
        "display_names": {},
        "last_export": {
            "out_dir": "",
            "fmt": "",
            "quality": 100,
            "scale_mode": "none",
            "percent": 100,
            "width": 1920,
            "height": 1080,
            "strip_metadata": None,
        },
        "window_geometry": "",
        "double_click_fit": True,
        "auto_open_export_folder": False,
        "detect_size": 1024,
        "use_gpu": True,
        "theme": "xiaodu",   # v1.19.0: the original light/dark skins are gone
        "default_save_fmt": "PNG",
        "default_strip_meta": "strip",
        "edit_defaults": {
            "effect": "blur",
            "strength": 15,
            "brush_size": 30,
            "tool": "pointer",
        },
        "params": {
            "blur_alpha": 0.95,
            "line_alpha": 1.0,
            "mist_alpha": 0.8,
            "custom_alpha": 1.0,
            "alpha": 0.95,
            "line_thickness": 5,
            "line_spacing": 10,
            "line_direction": "horizontal",
            "mist_color": [255, 255, 255],
            "light_intensity": 0.8,
            "light_feather": 30,
            "light_color": [255, 255, 255],
        },
    }


def _apply_new_defaults(cfg: dict) -> None:
    """Overwrite the tuned default keys on a legacy config.

    Old configs keep every key they already stored, so tuning done in 1.5.1
    (confidence/IoU/strength/mask dilation defaults) never reached them. The
    loader only calls this once, for files without the ui_version marker —
    afterwards the user's own values are preserved.
    """
    cfg["conf_threshold"] = 0.3
    cfg["iou_threshold"] = 0.7
    cfg["auto_strength"] = 15
    cfg.setdefault("edit_defaults", {})["strength"] = 15
    cfg["auto_mask_dilate"] = 5
    cfg["show_mask_overlay"] = True
    cfg["ui_version"] = UI_VERSION


def _apply_new_defaults_v3(cfg: dict) -> None:
    """1.14.5: strength defaults moved 10 -> 15 (user request).

    Only a stored 10 is bumped, and only because 10 *was* the built-in default
    -- so it means "never touched" (the panel's own strength controls were not
    even persisted before this version, see MainWindow._collect_auto_defaults).
    Any other number is a deliberate choice and is left alone."""
    try:
        if int(cfg.get("auto_strength", 0) or 0) == 10:
            cfg["auto_strength"] = 15
    except (TypeError, ValueError):
        pass
    ed = cfg.setdefault("edit_defaults", {})
    try:
        if int(ed.get("strength", 0) or 0) == 10:
            ed["strength"] = 15
    except (TypeError, ValueError):
        pass
    cfg["ui_version"] = UI_VERSION


def _apply_new_defaults_v2(cfg: dict) -> None:
    """1.6.2: overlay default flipped to ON for configs migrated by 1.5.1.

    Only the overlay flag is forced (the user asked for the default), and the
    marker is bumped; thresholds/strength etc. that the user may have tuned
    are left untouched."""
    cfg["show_mask_overlay"] = True
    cfg["ui_version"] = UI_VERSION


def load_config() -> dict:
    cfg = default_config()
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update(data)
            marker = data.get("ui_version")
            if marker == "1.5.1":
                _apply_new_defaults_v2(cfg)
                save_config(cfg)
            elif marker == "1.6.2":
                # 1.7.2: strength defaults moved 7 -> 10 for migrated configs
                cfg["auto_strength"] = 10
                cfg.setdefault("edit_defaults", {})["strength"] = 10
                # ...and 1.14.5 moved them on to 15 (chain, so the end state of
                # every migration path is the same)
                _apply_new_defaults_v3(cfg)
                save_config(cfg)
            elif marker == "1.7.2":
                # 1.14.5: 10 -> 15 (only a stored 10 counts as "untouched")
                _apply_new_defaults_v3(cfg)
                save_config(cfg)
            elif not marker:
                _apply_new_defaults(cfg)
                # persist the migration so it only ever runs once
                save_config(cfg)
    except Exception:
        pass
    return cfg


def save_config(cfg: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
