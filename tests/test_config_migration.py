# -*- coding: utf-8 -*-
"""Config migration tests: stale defaults from old config files must be
replaced once, and user values must survive afterwards."""
import json

from app.core import config as config_mod
from app.core.config import UI_VERSION, default_config


def _write_legacy(tmp_path):
    """A config exactly like the one a pre-1.5.1 install would have left."""
    legacy = {
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "auto_strength": 4,
        "edit_defaults": {"strength": 5},
        "model_path": "old-model.onnx",
    }
    config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_mod.CONFIG_PATH.write_text(
        json.dumps(legacy), encoding="utf-8"
    )


def test_legacy_config_is_migrated_once(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")
    _write_legacy(tmp_path)

    cfg = config_mod.load_config()
    assert cfg["conf_threshold"] == 0.3
    assert cfg["iou_threshold"] == 0.7
    assert cfg["auto_strength"] == 15          # 1.14.5 default
    assert cfg["edit_defaults"]["strength"] == 15
    assert cfg["auto_mask_dilate"] == 5
    assert cfg["show_mask_overlay"] is True
    assert cfg["ui_version"] == UI_VERSION
    # unrelated user keys must survive
    assert cfg["model_path"] == "old-model.onnx"

    # migration was persisted to disk (so it never runs again)
    on_disk = json.loads(config_mod.CONFIG_PATH.read_text(encoding="utf-8"))
    assert on_disk["ui_version"] == UI_VERSION
    assert on_disk["auto_strength"] == 15


def test_migrated_config_keeps_user_values(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")
    _write_legacy(tmp_path)

    cfg = config_mod.load_config()
    # user tunes a value afterwards
    cfg["conf_threshold"] = 0.55
    config_mod.save_config(cfg)

    cfg2 = config_mod.load_config()
    assert cfg2["conf_threshold"] == 0.55
    assert cfg2["iou_threshold"] == 0.7  # migrated default preserved


def test_migrated_162_config_ends_at_the_current_default(tmp_path, monkeypatch):
    import json

    from app.core import config as config_mod
    from app.core.config import UI_VERSION

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.json")
    legacy = {
        "ui_version": "1.6.2",
        "conf_threshold": 0.5,          # user-tuned, must survive
        "auto_strength": 7,
        "edit_defaults": {"strength": 7},
        "show_mask_overlay": True,
    }
    config_mod.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_mod.CONFIG_PATH.write_text(json.dumps(legacy), encoding="utf-8")

    cfg = config_mod.load_config()
    assert cfg["auto_strength"] == 15          # 1.14.5 default
    assert cfg["edit_defaults"]["strength"] == 15
    assert cfg["ui_version"] == UI_VERSION
    assert cfg["conf_threshold"] == 0.5  # untouched
    assert cfg["show_mask_overlay"] is True  # untouched


def test_default_config_is_current():
    cfg = default_config()
    assert cfg["ui_version"] == UI_VERSION
    assert cfg["conf_threshold"] == 0.3
    assert cfg["iou_threshold"] == 0.7