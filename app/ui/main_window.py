# -*- coding: utf-8 -*-
"""Unified Image Tool main window."""
from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from PySide6.QtCore import QEvent, QUrl, Qt, QThread, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.config import load_config, save_config
from ..core.detector import ONNXDetector, pick_providers
from ..version import APP_NAME, __version__
import shutil

from ..core.export import (
    compose_from_state,
    export_batch,
    fmt_from_ext,
    run_pipeline_on_image,
    state_has_edits,
)
from ..core.image_ops import (
    load_resized,
    original_size,
    resize_image,
    save_processed,
)
from ..core.numbering import build_rename_plan, expand_rename_template, build_filename, natural_key
from ..core.paths import resource_path
from .editor_canvas import EditorCanvas
from .save_dialog import SaveDialog
from .settings_dialog import SettingsDialog
from .theme import apply_theme
from .thumbnail_list import ThumbnailList
from .tool_panel import ToolPanel
from .widgets import HeartBadge, HeartProgressBar, RenamePreviewDialog


class BatchExportWorker(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    done = Signal(dict)

    def __init__(self, paths, states, config, auto_settings, save_cfg,
                 sticker_library, parent=None, display_names=None):
        super().__init__(parent)
        self._paths = paths
        self._states = states
        self._config = config
        self._auto = auto_settings
        self._save = save_cfg
        self._stickers = sticker_library
        self._display_names = display_names or {}
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        result = export_batch(
            self._paths,
            self._states,
            self._config,
            self._auto,
            self._save,
            self._stickers,
            progress_cb=self.progress.emit,
            status_cb=self.status.emit,
            is_cancelled=lambda: self._cancel,
            display_names=self._display_names,
        )
        # details[i] corresponds to paths[i]: one entry per processed path
        # (failures included; paths skipped by a cancel have no entry)
        result["exported_paths"] = [
            p for p, d in zip(self._paths, result.get("details") or [])
            if d.get("status") == "成功"
        ]
        self.done.emit(result)


class BatchAutoCensorWorker(QThread):
    """Run auto censor detection over all images in a background thread."""

    progress = Signal(int, int)
    status = Signal(str)
    done = Signal(dict)

    def __init__(self, paths, states, detector, auto_settings, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._states = states
        self._detector = detector
        self._auto = auto_settings
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self._paths)
        ok = 0
        for i, path in enumerate(self._paths):
            if self._cancel:
                self.status.emit("已取消")
                break
            self.progress.emit(i + 1, total)
            self.status.emit(f"自动打码: {Path(path).name}")
            state = self._states.get(path)
            if state is None:
                try:
                    orig_pil, _, _ = load_resized(path)
                    rgb = np.array(orig_pil.convert("RGB"))
                    orig_size = original_size(path)
                    state = {
                        "pil": orig_pil,
                        "selections": [],
                        "stickers": [],
                        "auto_mask": np.zeros(rgb.shape[:2], dtype=bool),
                        "erase_mask": np.zeros(rgb.shape[:2], dtype=bool),
                        "orig_size": orig_size,
                        "preview_size": (rgb.shape[1], rgb.shape[0]),
                    }
                    self._states[path] = state
                except Exception as e:
                    print(f"跳过 {path}: {e}")
                    continue
            rgb = state.get("rgb")
            if rgb is None and state.get("pil") is not None:
                rgb = np.array(state["pil"].convert("RGB"))
            if rgb is None:
                # LRU-evicted state: bitmaps are gone but selections/masks/
                # preview_size remain -- reload the edit-size bitmap so the
                # detection runs at the same resolution as the main window.
                try:
                    orig_pil, _, _ = load_resized(path)
                except Exception as e:
                    print(f"跳过 {path}: {e}")
                    continue
                rgb = np.array(orig_pil.convert("RGB"))
                state["pil"] = orig_pil
                state.setdefault(
                    "orig_size", original_size(path))
                state["preview_size"] = (rgb.shape[1], rgb.shape[0])
                state.setdefault(
                    "erase_mask", np.zeros(rgb.shape[:2], dtype=bool))
            try:
                dets = self._detector.detect(
                    rgb[:, :, ::-1],
                    conf_threshold=self._auto["conf"],
                    iou_threshold=self._auto["iou"],
                    allowed_classes=self._auto["classes"],
                    input_size=self._auto.get("input_size", 1024),
                )
                if dets:
                    union = np.zeros(rgb.shape[:2], dtype=bool)
                    for d in dets:
                        union |= d.mask
                    state["auto_mask"] = union
                    state["rgb"] = rgb
                    ok += 1
                else:
                    state["auto_mask"] = np.zeros(rgb.shape[:2], dtype=bool)
            except Exception as e:
                print(f"自动打码失败 {path}: {e}")

        self.done.emit({
            "ok": ok, "total": total, "cancelled": self._cancel,
        })


class BatchPipelineWorker(QThread):
    """SuperBatch-style pipeline: content actions mutate the state, the
    composed image gets output actions (resize/convert/rename) and is saved
    to the pipeline output dir with conflict handling."""

    progress = Signal(int, int)
    status = Signal(str)
    done = Signal(dict)

    def __init__(self, paths, states, actions, sticker_library, auto_settings,
                 params, detector, out_dir, on_conflict, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._states = states
        self._actions = actions
        self._lib = sticker_library
        self._auto = auto_settings
        self._params = params
        self._detector = detector
        self._out = out_dir
        self._conflict = on_conflict
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self._paths)
        ok = failed = 0
        errors = []
        used = set()
        done_paths: list[str] = []
        os.makedirs(self._out, exist_ok=True)
        for i, path in enumerate(self._paths):
            if self._cancel:
                self.status.emit("已取消")
                break
            self.progress.emit(i + 1, total)
            self.status.emit(f"流水线: {Path(path).name}")
            try:
                converted, pil_fmt, kwargs, ext, stem = run_pipeline_on_image(
                    path, self._states, self._actions, self._lib,
                    self._auto, self._params, self._detector, index=i,
                )
                name = f"{stem}.{ext}"
                if self._conflict == "dedupe":
                    n = 2
                    while name in used:
                        name = f"{stem}_{n}.{ext}"
                        n += 1
                used.add(name)
                dest = os.path.join(self._out, name)
                converted.save(dest, pil_fmt, **kwargs)
                converted.close()
                ok += 1
                done_paths.append(path)
            except Exception as e:
                failed += 1
                errors.append(f"{Path(path).name}: {e}")
                print(f"流水线失败 {path}: {e}")
        self.done.emit({
            "ok": ok, "failed": failed, "total": total,
            "cancelled": self._cancel, "errors": errors,
            "done_paths": done_paths,
        })


class BatchJobWorker(QThread):
    """Generic background worker for the batch tab (convert/resize/rename).

    job_fn(path, index) is captured on the main thread and must not touch
    Qt widgets; it returns nothing and raises on failure (counted as failed).
    """

    progress = Signal(int, int)
    status = Signal(str)
    done = Signal(dict)

    def __init__(self, paths, job_fn, title, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._job = job_fn
        self._title = title
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self._paths)
        ok = failed = 0
        done_paths: list[str] = []
        for i, path in enumerate(self._paths):
            if self._cancel:
                self.status.emit("已取消")
                break
            self.progress.emit(i + 1, total)
            self.status.emit(f"{self._title}: {Path(path).name}")
            try:
                self._job(path, i)
                ok += 1
                done_paths.append(path)
            except Exception as e:
                failed += 1
                print(f"{self._title} 失败 {path}: {e}")
        self.done.emit({
            "ok": ok, "failed": failed, "total": total, "cancelled": self._cancel,
            "done_paths": done_paths,
        })


class MainWindow(QMainWindow):
    def __init__(self, detector: ONNXDetector, config: dict):
        super().__init__()
        self.detector = detector
        self.config = config
        self.setWindowTitle(f"{APP_NAME} v{__version__}")

        geometry = self._stored_geometry(config)
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1360, 840)

        self.current_path: str | None = None
        self.image_states: dict[str, dict] = {}
        self._state_lru: list[str] = []
        self.sticker_library: dict[str, Image.Image] = {}
        self.batch_worker: BatchExportWorker | None = None
        self.auto_worker: BatchAutoCensorWorker | None = None
        self.job_worker: BatchJobWorker | None = None
        self._pipeline_worker = None
        self._last_export_dir: str | None = None
        # unsaved-EDIT tracking, per image path (v1.17.0): a global bool
        # used to be cleared by ANY export, so cancelling an export of 300
        # at image 50 silently disarmed the close warning for the other 250
        self._dirty_paths: set[str] = set()
        self._mini_window = None

        self._build_ui()
        self._load_stickers()
        # Numpad image navigation needs an APPLICATION-level filter: the key
        # goes to whichever widget has focus (canvas or thumbnail list), and a
        # filter installed on this window would never see it.
        _app = QApplication.instance()
        if _app is not None:
            _app.installEventFilter(self)
        self.thumbnail_list.set_display_names(self.config.get("display_names", {}))
        self.thumbnail_list.displayNamesChanged.connect(self._save_display_names)
        self.statusBar().showMessage("就绪")

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        self.top_buttons: dict[str, QPushButton] = {}
        for text, handler in [
            ("📂 导入图片", self.import_images),
            ("📁 导入文件夹", self.import_folder),
            ("🤖 批量自动打码", self.batch_auto_censor),
            ("↩ 撤销", lambda: self.canvas.undo()),
            ("↪ 重做", lambda: self.canvas.redo()),
            ("🗑 删除选区/打码", lambda: self.canvas.delete_selected()),
            ("💾 保存/导出", self.open_save_dialog),
            ("⚙ 设置", self.open_settings),
            ("🧩 迷你模式", self.open_mini_window),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            self.top_buttons[text] = btn
            top.addWidget(btn)
        # hold-to-compare: show the pristine original while the button is
        # held (also available via holding SPACE on the canvas)
        btn_orig = QPushButton("👁 原图(按住)")
        btn_orig.setToolTip("按住查看未打码原图，松开恢复（也可在画布上按住空格）")
        btn_orig.pressed.connect(lambda: self.canvas.set_show_original(True))
        btn_orig.released.connect(lambda: self.canvas.set_show_original(False))
        top.addWidget(btn_orig)
        root.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter

        self.thumbnail_list = ThumbnailList(thumb_size=96)
        self.thumbnail_list.currentPathChanged.connect(self._on_current_path_changed)
        self.thumbnail_list.customContextMenuRequested.connect(self._on_thumb_context_menu)
        self.thumbnail_list.deleteRequested.connect(self._remove_selected_images)
        left_panel = QWidget()
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(2)
        left_lay.addWidget(self.thumbnail_list, 1)
        stats_bar = QFrame()
        stats_bar.setObjectName("statsBar")
        stats_row = QHBoxLayout(stats_bar)
        stats_row.setContentsMargins(2, 4, 2, 2)
        stats_row.setSpacing(6)
        self.thumb_stats_label = QLabel("共 0 张 ｜ 已打码 0 ｜ 未打码 0")
        self.thumb_stats_label.setObjectName("thumbStats")
        self.thumb_stats_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        stats_row.addWidget(self.thumb_stats_label)
        stats_row.addStretch()
        self.zoom_label = QLabel("缩放 100%")
        self.zoom_label.setObjectName("thumbStats")
        self.zoom_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        stats_row.addWidget(self.zoom_label)
        left_lay.addWidget(stats_bar)
        splitter.addWidget(left_panel)

        self.canvas = EditorCanvas()
        self.canvas.state_changed.connect(self._save_current_state)
        self.canvas.state_changed.connect(self._on_canvas_state_changed)
        self.canvas.zoomChanged.connect(self._on_canvas_zoom)
        self.canvas.show_mask_overlay = self.config.get("show_mask_overlay", True)
        self.canvas.double_click_fit = self.config.get("double_click_fit", True)
        splitter.addWidget(self.canvas)

        self.tool_panel = ToolPanel(self)
        self.canvas.params = self.config.get("params", {})
        self._apply_edit_defaults()
        self.tool_panel._apply_auto_defaults()
        splitter.addWidget(self.tool_panel)

        # 让主窗口统一处理外部文件拖拽；画布/右面板/缩略图列表都不拦截
        # （缩略图排序是 v1.6.5 的自管鼠标逻辑，不使用 Qt 拖放——历史上的
        # InternalMove 需要 acceptDrops 并曾因这里关掉而失效，已整体移除）。
        self.canvas.setAcceptDrops(False)
        self.canvas.viewport().setAcceptDrops(False)
        self.tool_panel.setAcceptDrops(False)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 760, 340])
        splitter.splitterMoved.connect(self._snap_left_panel_width)
        root.addWidget(splitter, 1)

        self.progress_bar = HeartProgressBar()   # 小独爱心：尽头骑爱心瞳
        self.progress_bar.setVisible(False)
        self.progress_cancel_btn = QPushButton("取消")
        self.progress_cancel_btn.setVisible(False)
        self.progress_cancel_btn.setFixedWidth(56)
        self.progress_cancel_btn.clicked.connect(self.cancel_batch)
        status = QStatusBar()
        status.addPermanentWidget(HeartBadge())   # v1.22.1 心粉常驻位（四色配额）
        status.addPermanentWidget(self.progress_bar)
        status.addPermanentWidget(self.progress_cancel_btn)
        self.setStatusBar(status)

        self.setCentralWidget(central)
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    def _load_stickers(self):
        stickers_dir = resource_path("app/stickers")
        self.sticker_library = {}
        # stem -> file on disk, so library management (rename / delete) can act
        # on the real file instead of guessing an extension
        self._sticker_paths: dict[str, Path] = {}
        if stickers_dir.is_dir():
            for f in sorted(stickers_dir.glob("*")):
                # .bmp is in this list on purpose: import_sticker_files accepts
                # it, so before v1.13 a dropped .bmp was copied into the
                # library and then silently never showed up
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp",
                                        ".gif", ".bmp"):
                    try:
                        img = Image.open(f)
                        img.load()
                        if img.mode not in ("RGBA", "LA"):
                            img = img.convert("RGBA")
                        self.sticker_library[f.stem] = img
                        self._sticker_paths[f.stem] = f
                    except Exception:
                        pass
        self.canvas.sticker_library = self.sticker_library
        self.tool_panel.refresh_sticker_list()
        # the watermark tab consumes the same library: keep its combo in sync
        if hasattr(self.tool_panel, "refresh_watermark_assets"):
            self.tool_panel.refresh_watermark_assets()

    def refresh_stickers(self, _listwidget=None):
        self._load_stickers()
        if _listwidget is not None:
            self.tool_panel.refresh_sticker_list()

    def import_sticker_files(self, files) -> list[str]:
        """Copy dropped/picked image files into the sticker library folder.

        Returns the list of imported file names. The sticker library is a
        folder scan (resource_path("app/stickers")), so importing == copying
        into that folder then refreshing; duplicate names get a numeric
        suffix instead of overwriting anything."""
        stickers_dir = resource_path("app/stickers")
        stickers_dir.mkdir(parents=True, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        images = []
        for raw in files:
            p = Path(raw)
            if p.is_dir():
                images.extend(
                    x for x in sorted(p.rglob("*"))
                    if x.suffix.lower() in exts and x.is_file()
                )
            elif p.suffix.lower() in exts and p.is_file():
                images.append(p)
        added = []
        for img in images:
            dest = stickers_dir / img.name
            stem = img.stem
            n = 2
            while dest.exists():
                dest = stickers_dir / f"{stem}_{n}{img.suffix.lower()}"
                n += 1
            shutil.copyfile(str(img), str(dest))
            added.append(dest.name)
        if added:
            self._load_stickers()
        return added

    # ------------------------------------------------------------------
    # Sticker library management (v1.13): the library used to be import-only,
    # so duplicate "name_2 / name_3" copies piled up in app/stickers with no
    # way to remove or rename them.
    # ------------------------------------------------------------------
    def _sticker_asset_path(self, name: str) -> Path | None:
        return (getattr(self, "_sticker_paths", None) or {}).get(name)

    # --- where an asset NAME can still be referenced (v1.14.2) -------------
    # The first cut of library management only fixed image_states +
    # canvas.stickers. That misses two reachable places:
    #   * the undo/redo snapshots, which are DEEP COPIES taken at edit time --
    #     after a rename, undoing a sticker edit restored the dead name, so the
    #     sticker rendered as nothing in the preview AND in the exported file;
    #   * the saved watermark presets / pipelines, which store the asset name
    #     as a string, so applying them after a rename failed with a misleading
    #     "asset no longer exists" message.
    def _live_sticker_lists(self):
        """Sticker lists that are part of the current document state."""
        for state in self.image_states.values():
            lst = (state or {}).get("stickers")
            if lst:
                yield lst
        yield self.canvas.stickers

    def _history_sticker_lists(self):
        """Sticker lists inside undo/redo snapshots.

        Every snapshot keeps its stickers at index 1 -- both the 4-tuple used by
        the full stack (selections, stickers, auto_mask, erase_mask) and the
        3-tuple used by the Ctrl+Z manual stack (selections, stickers,
        erase_mask)."""
        for stack in (self.canvas._undo_stack, self.canvas._redo_stack,
                      self.canvas._manual_undo_stack):
            for snap in stack:
                if len(snap) > 1 and isinstance(snap[1], list):
                    yield snap[1]

    def _asset_name_refs(self, name: str) -> dict:
        """Every place an asset name is still referenced, as counts."""
        def count(lists):
            n, seen = 0, set()
            for lst in lists:
                for st in lst:
                    if getattr(st, "name", None) == name and id(st) not in seen:
                        seen.add(id(st))
                        n += 1
            return n

        return {
            "live": count(self._live_sticker_lists()),
            "history": count(self._history_sticker_lists()),
            "presets": sum(1 for p in (self.config.get("watermark_presets") or [])
                           for it in (p.get("items") or [])
                           if it.get("asset") == name),
            "pipelines": sum(1 for p in (self.config.get("batch_pipelines") or [])
                             for a in (p.get("actions") or [])
                             if (a.get("params") or {}).get("asset") == name),
            "pending": sum(
                1 for a in (getattr(self.tool_panel, "_pipe_pending", None) or [])
                if (a.get("params") or {}).get("asset") == name),
        }

    def _repoint_sticker_assets(self, old: str, new: str) -> tuple[int, int]:
        """Point every reference to ``old`` at ``new``.

        Returns (placements re-pointed, config references re-pointed)."""
        placements = 0
        for lst in list(self._live_sticker_lists()) + \
                list(self._history_sticker_lists()):
            for st in lst:
                if getattr(st, "name", None) == old:
                    st.name = new
                    placements += 1
        cfg_hits = 0
        for preset in (self.config.get("watermark_presets") or []):
            for item in (preset.get("items") or []):
                if item.get("asset") == old:
                    item["asset"] = new
                    cfg_hits += 1
        for pipe in (self.config.get("batch_pipelines") or []):
            for act in (pipe.get("actions") or []):
                params = act.get("params") or {}
                if params.get("asset") == old:
                    params["asset"] = new
                    cfg_hits += 1
        for act in (getattr(self.tool_panel, "_pipe_pending", None) or []):
            params = act.get("params") or {}
            if params.get("asset") == old:
                params["asset"] = new
                cfg_hits += 1
        favs = self.config.get("sticker_favorites") or []
        if old in favs:
            self.config["sticker_favorites"] = [
                new if x == old else x for x in favs]
            cfg_hits += 1
        return placements, cfg_hits

    def _purge_sticker_asset(self, name: str) -> int:
        """Drop every placement of a now-deleted asset (live state + history).

        Keeping them would leave invisible ghosts: the canvas shows nothing
        while the thumbnail badge still claims a sticker, and a later undo
        resurrects a placement that can never render. Lists are mutated in
        place because the canvas and the saved state share Sticker objects."""
        seen = set()
        for lst in list(self._live_sticker_lists()) + \
                list(self._history_sticker_lists()):
            for st in lst:
                if getattr(st, "name", None) == name and id(st) not in seen:
                    seen.add(id(st))
            lst[:] = [st for st in lst if getattr(st, "name", None) != name]
        cur = self.canvas._sel_sticker
        if not (0 <= cur < len(self.canvas.stickers)):
            self.canvas._sel_sticker = -1
        return len(seen)

    def delete_sticker_asset(self, name: str):
        """Delete one asset from the library.

        The ONLY sticker action that touches the user's disk, so it confirms
        first and spells out every place the asset is still referenced --
        including the undo history and the saved presets/pipelines."""
        path = self._sticker_asset_path(name)
        if path is None:
            QMessageBox.warning(self, "删除素材", f"找不到素材「{name}」的文件。")
            return
        refs = self._asset_name_refs(name)
        msg = f"从素材库删除「{name}」？\n\n文件：{path.name}"
        bits = []
        if refs["live"]:
            bits.append(f"{refs['live']} 处正在使用")
        if refs["history"]:
            bits.append(f"{refs['history']} 处编辑历史（撤销/重做）")
        if refs["presets"]:
            bits.append(f"{refs['presets']} 个水印方案项")
        if refs["pipelines"] + refs["pending"]:
            bits.append(f"{refs['pipelines'] + refs['pending']} 个流水线动作")
        if bits:
            msg += ("\n\n注意：还被 " + "、".join(bits) + " 引用。\n"
                    "删除后这些贴纸会被一并清理（不留看不见的幽灵贴纸），"
                    "方案/流水线里的该项会失效。")
        if QMessageBox.question(self, "删除素材", msg) != QMessageBox.Yes:
            return
        try:
            path.unlink()
        except Exception as e:
            QMessageBox.critical(self, "删除素材失败", str(e))
            return
        purged = self._purge_sticker_asset(name)
        self.config["sticker_favorites"] = [
            n for n in (self.config.get("sticker_favorites") or []) if n != name]
        save_config(self.config)
        self._load_stickers()
        if self.current_path is not None:
            self.canvas._rebuild()
        self.statusBar().showMessage(
            f"已删除素材「{name}」"
            + (f"，并清理了 {purged} 处引用" if purged else ""))

    def rename_sticker_asset(self, name: str):
        """Rename one library asset, re-pointing EVERY reference at the new
        name so placements, edit history and saved presets all keep working."""
        path = self._sticker_asset_path(name)
        if path is None:
            QMessageBox.warning(self, "重命名素材", f"找不到素材「{name}」的文件。")
            return
        new, ok = QInputDialog.getText(
            self, "重命名素材", f"素材「{name}」的新名称：", text=name)
        if not ok:
            return
        new = (new or "").strip()
        if not new or new == name:
            return
        for ch in '\\/:*?"<>|':
            new = new.replace(ch, "_")
        dest = path.with_name(new + path.suffix)
        if dest.exists() or new in self.sticker_library:
            QMessageBox.warning(self, "重命名素材", f"「{new}」已存在。")
            return
        try:
            path.rename(dest)
        except Exception as e:
            QMessageBox.critical(self, "重命名素材失败", str(e))
            return
        placements, cfg_hits = self._repoint_sticker_assets(name, new)
        if cfg_hits:
            save_config(self.config)
        self._load_stickers()
        if self.current_path is not None:
            self.canvas._rebuild()
        extra = f"，{cfg_hits} 处方案/收藏引用" if cfg_hits else ""
        self.statusBar().showMessage(
            f"素材「{name}」→「{new}」"
            f"（{placements} 处贴纸已同步，含撤销历史{extra}）")

    def toggle_sticker_favorite(self, name: str):
        favs = list(self.config.get("sticker_favorites") or [])
        if name in favs:
            favs.remove(name)
        else:
            favs.append(name)
        self.config["sticker_favorites"] = favs
        save_config(self.config)
        self.tool_panel.refresh_sticker_list()

    def remove_watermarks_current(self):
        """Watermark tab: drop every watermark on the current image."""
        if self.canvas.image_np is None:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        n = self.canvas.remove_watermarks()
        if not n:
            self.statusBar().showMessage("当前图没有水印")
            return
        self.statusBar().showMessage(f"已移除 {n} 个水印（可撤销）")

    # ------------------------------------------------------------------
    # Arrow-key image navigation (v1.13 numpad, v1.14.4 plain arrows)
    # ------------------------------------------------------------------
    # Windows virtual-key codes of the numeric keypad with NumLock ON. Qt sets
    # KeypadModifier for these by itself (verified in Qt's qwindowskeymapper.cpp:
    # wParam VK_NUMPAD0-9 is outside '0'..'9', so the modifier is added; a REAL
    # arrow key has the extended bit set and gets none).
    _VK_NUMPAD = {0x68: -1, 0x64: -1, 0x62: 1, 0x66: 1}   # 8/4 prev, 2/6 next
    _PAD_PREV = (Qt.Key_Up, Qt.Key_Left, Qt.Key_8, Qt.Key_4)
    _PAD_NEXT = (Qt.Key_Down, Qt.Key_Right, Qt.Key_2, Qt.Key_6)

    def _pad_delta(self, event):
        """Which direction a key event means, or None if it is not one of ours.

        Recognised (v1.14.4 -- the user asked for the REAL arrow keys to work
        everywhere, having been given a numpad-only binding before):

        * **real arrow keys**: Up/Left = previous, Down/Right = next, claimed
          GLOBALLY. The canvas loses arrow-key scrolling as a result; the user
          accepted that trade (wheel zoom and drag-panning are unaffected);
        * numpad arrows through KeypadModifier (NumLock off) and the numpad
          digits Key_8/2/4/6 (NumLock on);
        * numpad keys identified by ``nativeVirtualKey()``.

        The v1.14.3 fallback that treated the TOP-ROW digits 8/2/4/6 as
        navigation is gone -- it only existed to guess at Fn-overlay keypads.
        """
        vk = event.nativeVirtualKey()
        if vk in self._VK_NUMPAD:
            return self._VK_NUMPAD[vk]
        key = event.key()
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.KeypadModifier:
            if key in self._PAD_PREV:
                return -1
            if key in self._PAD_NEXT:
                return 1
            return None
        if not mods:
            if key in (Qt.Key_Up, Qt.Key_Left):
                return -1
            if key in (Qt.Key_Down, Qt.Key_Right):
                return 1
        return None

    # widgets that legitimately use the arrow keys themselves; a key must never
    # be stolen from one of these
    _ARROW_OWNERS = (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox,
                     QComboBox, QAbstractItemView, QSlider)

    def eventFilter(self, obj, event):
        """Arrow keys (and the numeric keypad) switch images.

        Up/Left = previous, Down/Right = next. Guards, in order:

        * the window must be active, so a dialog of this app does not navigate
          the list behind itself;
        * no popup may be open (an open combo popup owns its arrow keys);
        * the focused widget must not be one that uses arrows itself -- text and
          number fields, combo boxes, item views (the thumbnail list still
          navigates natively, which lands on the same image) and sliders.

        The canvas (a QGraphicsView, deliberately NOT an item view) is the main
        case this exists for: clicking the image and then pressing the arrow
        keys must switch images."""
        if event.type() == QEvent.Type.KeyPress and self.isActiveWindow():
            delta = self._pad_delta(event)
            if delta is not None:
                foc = self.focusWidget()
                skip = (isinstance(foc, self._ARROW_OWNERS)
                        or QApplication.activePopupWidget() is not None
                        or QApplication.activeModalWidget() is not None)
                if not skip:
                    self.goto_relative_image(delta)
                    return True
        return super().eventFilter(obj, event)

    def goto_relative_image(self, delta: int):
        """Move the thumbnail selection by delta, clamped to the list."""
        if self._any_worker_running():
            return
        n = self.thumbnail_list.count()
        if n == 0:
            return
        row = self.thumbnail_list.currentRow()
        if row < 0:
            row = 0
        target = min(max(row + delta, 0), n - 1)
        if target == row:
            return
        self.thumbnail_list.setCurrentRow(target)

    def _apply_edit_defaults(self):
        """Read edit defaults from config and apply to UI/canvas."""
        ed = self.config.get("edit_defaults", {})
        effect = ed.get("effect", "blur")
        strength = int(ed.get("strength", 15))
        brush = int(ed.get("brush_size", 30))
        tool = ed.get("tool", "pointer")

        # effect combo
        idx = self.tool_panel.effect_combo.findData(effect)
        if idx >= 0:
            self.tool_panel.effect_combo.setCurrentIndex(idx)
        self.canvas.current_effect = effect
        self.canvas.current_strength = strength
        self.tool_panel.strength_spin.setValue(strength)
        self.canvas.brush_size = brush
        self.tool_panel.brush_spin.setValue(brush)
        if tool in self.tool_panel.tool_buttons:
            self.tool_panel._set_tool(tool)

    def _collect_edit_defaults(self):
        self.config["edit_defaults"] = {
            "effect": self.canvas.current_effect,
            "strength": self.canvas.current_strength,
            "brush_size": self.canvas.brush_size,
            "tool": self.canvas.tool,
        }
        save_config(self.config)

    def _collect_auto_defaults(self):
        """Persist the 自动打码 tab's widgets on exit (v1.14.5).

        These were only ever written by the Settings dialog, so anything the
        user changed in the main panel -- auto strength above all -- was
        forgotten on restart even though _apply_auto_defaults reads it back.
        Seven keys are covered; the panel is the only place they are edited.

        Reads through ``auto_settings()`` so the panel keeps owning the mapping
        from widget to config key."""
        try:
            s = self.tool_panel.auto_settings()
        except Exception:
            return
        self.config["conf_threshold"] = float(s["conf"])
        self.config["iou_threshold"] = float(s["iou"])
        self.config["selected_regions"] = list(s["classes"])
        self.config["auto_effect"] = s["auto_effect"]
        self.config["auto_strength"] = int(s["auto_strength"])
        self.config["auto_mask_dilate"] = int(s["auto_mask_dilate"])
        self.config["detect_size"] = int(s["input_size"])
        save_config(self.config)

    def _unsaved_paths(self) -> set[str]:
        """Dirty paths that STILL carry censor content: a path whose
        edits were undone/cleared must not trigger the close warning."""
        return {p for p in self._dirty_paths
                if p in self.image_states and state_has_edits(self.image_states[p])}

    def closeEvent(self, event):
        dirty = self._unsaved_paths()
        if dirty:
            ret = QMessageBox.question(
                self, "未保存的修改",
                f"有 {len(dirty)} 张图包含未导出的打码内容，关闭将丢失。\n确定退出吗？",
            )
            if ret != QMessageBox.Yes:
                event.ignore()
                return
        # Stop workers and wait for them: they hold image_states and ONNX
        # sessions, so writing/tearing down while a thread is mid-inference
        # can corrupt state or crash on exit. cancel() is checked between
        # items, so the wait normally returns after the current item.
        for _w in (self.auto_worker, self.batch_worker, self.job_worker,
                   self._pipeline_worker):
            try:
                if _w and _w.isRunning():
                    _w.cancel()
                    _w.wait(15000)
            except Exception:
                pass
        try:
            self._save_current_state()
            self._collect_edit_defaults()
            self._collect_auto_defaults()
            self.config["window_geometry"] = bytes(
                self.saveGeometry().toBase64()).decode("ascii")
            save_config(self.config)
        except Exception:
            pass
        # drop the app-level numpad filter, otherwise a closed window would keep
        # receiving (and answering for) every key event
        _app = QApplication.instance()
        if _app is not None:
            _app.removeEventFilter(self)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp *.gif);;所有文件 (*.*)"
        )
        if files:
            self._add_paths(files)

    def import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
            files = sorted(
                (
                    str(p) for p in Path(folder).rglob("*")
                    if p.suffix.lower() in exts and p.is_file()
                ),
                key=lambda p: natural_key(Path(p).name),
            )
            self._add_paths(files)

    def _add_paths(self, paths: list[str]):
        paths = sorted(set(paths), key=lambda p: natural_key(Path(p).name))
        # v1.16.0: skip files that are already in the list (the old code let
        # the same file enter twice -> duplicate thumbnails and exports).
        # normcase so `D:\pics\a.png` and `d:\PICS\A.PNG` count as one file.
        known = {os.path.normcase(p) for p in self.thumbnail_list.item_paths()}
        fresh = [p for p in paths if os.path.normcase(p) not in known]
        skipped = len(paths) - len(fresh)
        if not fresh:
            if skipped:
                self.statusBar().showMessage(f"所选 {skipped} 张图片都已在列表中")
            return
        self.thumbnail_list.add_paths(fresh)
        self._update_thumb_stats()
        if self.current_path is None and fresh:
            self._load_path(fresh[0])
        if skipped:
            self.statusBar().showMessage(
                f"已导入 {len(fresh)} 张，跳过 {skipped} 张已在列表中")

    def _load_path(self, path: str, save_current: bool = True):
        if save_current:
            self._save_current_state()
        state = self.image_states.get(path)
        # A state may exist but be INCOMPLETE in two ways: batch operations
        # (e.g. "批量加水印到全部") append content to states of images that
        # were never opened, and the LRU eviction (v1.16.0) drops the bitmaps
        # of not-recently-used images while keeping everything else. Either
        # way "pil" is missing: reload the bitmap from disk and carry over
        # everything the state already holds (stickers/selections AND the
        # detection/erase masks -- an evicted image must not lose its
        # censoring just because it was reopened).
        if state is None or "pil" not in state or "orig_size" not in state:
            prev = self.image_states.pop(path, None) or {}
            try:
                orig_pil, _, _ = load_resized(path)
                rgb = np.array(orig_pil.convert("RGB"))
                orig_size = original_size(path)
                prev_auto = prev.get("auto_mask")
                prev_erase = prev.get("erase_mask")
                state = {
                    "pil": orig_pil,
                    "selections": [],
                    "stickers": [],
                    "auto_mask": (prev_auto if isinstance(prev_auto, np.ndarray)
                                  and prev_auto.shape == rgb.shape[:2]
                                  else np.zeros(rgb.shape[:2], dtype=bool)),
                    "erase_mask": (prev_erase if isinstance(prev_erase, np.ndarray)
                                   and prev_erase.shape == rgb.shape[:2]
                                   else np.zeros(rgb.shape[:2], dtype=bool)),
                    "orig_size": orig_size,
                    "preview_size": (rgb.shape[1], rgb.shape[0]),
                }
                # carry over whatever the partial state held (watermarks
                # appended by a batch action must survive the first open)
                state["stickers"] = list(prev.get("stickers") or [])
                state["selections"] = list(prev.get("selections") or [])
                self.image_states[path] = state
            except Exception as e:
                if prev:
                    self.image_states[path] = prev
                QMessageBox.critical(self, "加载失败", str(e))
                return
        # Only switch after the state exists: assigning current_path earlier
        # made a failed load leave the old canvas registered under the new
        # path (the next _save_current_state wrote wrong pixels into it).
        self.current_path = path
        self._touch_state(path)
        # Block signals while loading so set_image() doesn't erase saved state
        self.canvas.blockSignals(True)
        self.canvas.set_image(state["rgb"] if "rgb" in state else np.array(state["pil"].convert("RGB")))
        self.canvas.selections = list(state.get("selections", []))
        self.canvas.stickers = list(state.get("stickers", []))
        if self.canvas.image_np is not None:
            self.canvas.auto_mask = state.get(
                "auto_mask", np.zeros(self.canvas.image_np.shape[:2], dtype=bool)
            )
            self.canvas.erase_mask = state.get(
                "erase_mask", np.zeros(self.canvas.image_np.shape[:2], dtype=bool)
            )
        self.canvas._rebuild(fit=True)
        self.canvas.blockSignals(False)
        # v1.22.0：换图时重置自动打码页的结果（上一张的命中数不能留到这一张）
        panel = getattr(self, "tool_panel", None)
        if panel is not None and hasattr(panel, "on_image_changed"):
            panel.on_image_changed()
        self.statusBar().showMessage(f"当前: {path}")

    def _on_current_path_changed(self, path: str):
        self._load_path(path)

    def _save_current_state(self):
        if self.current_path is None or self.canvas.image_np is None:
            return
        if self.auto_worker and self.auto_worker.isRunning():
            # 批量自动打码线程正在写 image_states；此刻回写画布状态会把它
            # 刚写入的 auto_mask 用画布上的旧值覆盖回去。
            return
        state = self.image_states.setdefault(self.current_path, {})
        state["rgb"] = self.canvas.image_np.copy()
        state["pil"] = Image.fromarray(self.canvas.image_np).convert("RGB")
        state["preview_size"] = (
            self.canvas.image_np.shape[1], self.canvas.image_np.shape[0])
        state["selections"] = list(self.canvas.selections)
        state["stickers"] = list(self.canvas.stickers)
        state["auto_mask"] = (
            self.canvas.auto_mask.copy() if self.canvas.auto_mask is not None else None
        )
        state["erase_mask"] = (
            self.canvas.erase_mask.copy() if self.canvas.erase_mask is not None else None
        )
        self._touch_state(self.current_path)
        self._evict_image_states()

    # ------------------------------------------------------------------
    # LRU bitmap eviction (v1.16.0): image_states used to keep the edit-
    # size bitmaps (rgb + pil, ~19MB at 2048²) of EVERY imported image for
    # the whole session -- 300 photos ≈ 3.8GB. Only the current image and
    # the most recently viewed ones need their bitmaps; every state also
    # records `preview_size`, so exports/pipelines can rescale selections
    # and reload the original from disk (compose_from_state handles it).
    # ------------------------------------------------------------------
    EVICT_KEEP = 6   # current image + 5 most recently viewed

    def _touch_state(self, path: str):
        """Mark ``path`` as recently used in the LRU order."""
        if path in self._state_lru:
            self._state_lru.remove(path)
        self._state_lru.append(path)

    def _evict_image_states(self):
        """Drop the bitmaps of states beyond the LRU window. Everything else
        (selections, stickers, masks, orig_size, preview_size) stays, and
        compose_from_state reloads the original from disk when needed."""
        keep = set(self._state_lru[-self.EVICT_KEEP:])
        keep.add(self.current_path)
        for path, state in self.image_states.items():
            if path in keep:
                continue
            if "pil" in state or "rgb" in state:
                state.pop("pil", None)
                state.pop("rgb", None)

    def _on_canvas_zoom(self, ratio: float):
        self.zoom_label.setText(f"缩放 {round(ratio * 100)}%")

    # ------------------------------------------------------------------
    # Left panel snaps to standard widths: 1..5 thumbnail columns
    # ------------------------------------------------------------------
    def _snap_left_panel_width(self, pos: int, index: int):
        """Step the left panel through NINE standard widths: 0 (fully
        collapsed) plus 1..8 thumbnail columns. Crossing the midpoint
        between two standards jumps straight to that standard - an explicit
        switch, not a continuous follow."""
        if index != 0:
            return
        grid = self.thumbnail_list.gridSize().width()
        overhead = 16  # scrollbar + margins
        widths = [0] + [n * grid + overhead for n in range(1, 9)]
        step = 0
        for i in range(len(widths) - 1):
            mid = (widths[i] + widths[i + 1]) / 2
            if pos > mid:
                step = i + 1
        target = widths[step]
        sizes = self.splitter.sizes()
        if sizes[0] == target:
            return
        if step == 0:
            # fully collapsed: everything to the canvas, right panel intact
            self.splitter.setSizes([0, max(60, sum(sizes[1:])), sizes[2]])
        else:
            rest = max(60, sum(sizes) - target - sizes[2])
            self.splitter.setSizes([target, rest, sizes[2]])

    def _thumb_marks(self, path: str) -> dict:
        """Badge state for the thumbnail list: auto/manual/sticker/watermark."""
        state = self.image_states.get(path)
        if not state:
            return {}
        am = state.get("auto_mask")
        stickers = state.get("stickers") or []
        return {
            "auto": bool(am is not None and am.any()),
            "manual": bool(state.get("selections")),
            "sticker": any(not getattr(s, "watermark", False) for s in stickers),
            "watermark": any(getattr(s, "watermark", False) for s in stickers),
        }

    def _on_canvas_state_changed(self):
        if self.current_path:
            self._dirty_paths.add(self.current_path)
        panel = getattr(self, "tool_panel", None)
        if panel is not None and hasattr(panel, "refresh_auto_state"):
            panel.refresh_auto_state()      # 「清除自动掩码」跟随掩码有无
        if self.current_path and self.canvas.image_np is not None:
            self.thumbnail_list.set_badge(
                self.current_path, self._thumb_marks(self.current_path)
            )
            self._update_thumb_stats()

    def _update_thumb_stats(self):
        total = self.thumbnail_list.count()
        done = wm = 0
        for p in self.thumbnail_list.item_paths():
            marks = self._thumb_marks(p)
            if marks.get("auto") or marks.get("manual") or marks.get("sticker"):
                done += 1
            if marks.get("watermark"):
                wm += 1
        if wm:
            self.thumb_stats_label.setText(
                f"共 {total} 张 ｜ 已打码 {done} ｜ 未打码 {total - done} ｜ 已水印 {wm}"
            )
        else:
            self.thumb_stats_label.setText(
                f"共 {total} 张 ｜ 已打码 {done} ｜ 未打码 {total - done}"
            )

    # ------------------------------------------------------------------
    # Remove images from the thumbnail list (never touches disk files)
    # ------------------------------------------------------------------
    def _on_thumb_context_menu(self, pos):
        item = self.thumbnail_list.itemAt(pos)
        if item is None:
            # blank area: offer clearing the whole working list
            total = self.thumbnail_list.count()
            if total == 0:
                return
            menu = QMenu(self)
            act = menu.addAction(f"清空当前列表（{total} 张）")
            act.triggered.connect(self.clear_all_images)
            menu.exec(self.thumbnail_list.mapToGlobal(pos))
            return
        paths = self.thumbnail_list.selected_paths() or [self.thumbnail_list.current_path()]
        paths = [p for p in paths if p]
        if not paths:
            return
        menu = QMenu(self)
        action = menu.addAction(f"从列表移除 {len(paths)} 张")
        action.triggered.connect(lambda: self._remove_selected_images(paths))
        menu.addSeparator()
        if len(paths) == 1:
            rename_act = menu.addAction("重命名")
            rename_act.triggered.connect(
                lambda: self.thumbnail_list.rename_item(paths[0])
            )
        else:
            rename_act = menu.addAction(f"批量重命名…（{len(paths)} 张）")
            rename_act.triggered.connect(
                lambda: self._batch_rename_dialog(paths)
            )
        menu.addSeparator()
        sort_up = menu.addAction("按名称增序排序")
        sort_up.triggered.connect(
            lambda: self.thumbnail_list.sort_paths(
                lambda p: natural_key(self.thumbnail_list.display_name(p))
            )
        )
        sort_down = menu.addAction("按名称降序排序")
        sort_down.triggered.connect(
            lambda: self.thumbnail_list.sort_paths(
                lambda p: natural_key(self.thumbnail_list.display_name(p)),
                reverse=True,
            )
        )
        menu.addSeparator()
        time_up = menu.addAction("按修改时间增序")
        time_up.triggered.connect(
            lambda: self.thumbnail_list.sort_paths(lambda p: Path(p).stat().st_mtime)
        )
        time_down = menu.addAction("按修改时间降序")
        time_down.triggered.connect(
            lambda: self.thumbnail_list.sort_paths(
                lambda p: Path(p).stat().st_mtime, reverse=True
            )
        )
        menu.exec(self.thumbnail_list.mapToGlobal(pos))

    def _batch_rename_dialog(self, paths_ordered: list[str]):
        """In-app batch display-name rename (disk files are never touched)."""
        last = self.config.get("last_rename", {})
        dlg = QDialog(self)
        dlg.setWindowTitle(f"批量重命名（{len(paths_ordered)} 张，仅程序内显示名）")
        form = QFormLayout(dlg)
        prefix_edit = QLineEdit(str(last.get("prefix", "IMG")))
        start_spin = QSpinBox()
        start_spin.setRange(0, 99999)
        start_spin.setValue(int(last.get("start", 1)))
        digits_spin = QSpinBox()
        digits_spin.setRange(1, 10)
        digits_spin.setValue(int(last.get("digits", 3)))
        form.addRow("前缀", prefix_edit)
        start_row = QHBoxLayout()
        start_row.addWidget(start_spin)
        digits_row_label = QLabel("位数")
        start_row.addWidget(digits_row_label)
        start_row.addWidget(digits_spin)
        form.addRow("起始编号", start_row)
        preview = QLabel()
        preview.setObjectName("hint")
        preview.setWordWrap(True)
        form.addRow("预览", preview)
        clear_check = QCheckBox("清除这些图片的显示名（恢复原文件名显示）")
        form.addRow("", clear_check)

        def update_preview():
            if clear_check.isChecked():
                preview.setText("将恢复为原文件名显示")
                return
            pfx = prefix_edit.text()
            sample = ", ".join(
                f"{pfx}{str(start_spin.value() + i).zfill(digits_spin.value())}"
                for i in range(min(3, len(paths_ordered)))
            )
            preview.setText(f"示例: {sample} …")

        prefix_edit.textChanged.connect(update_preview)
        start_spin.valueChanged.connect(update_preview)
        digits_spin.valueChanged.connect(update_preview)
        clear_check.toggled.connect(update_preview)
        update_preview()

        btns = QHBoxLayout()
        ok = QPushButton("应用")
        cancel = QPushButton("取消")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        form.addRow(btns)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._apply_batch_rename(
            paths_ordered,
            prefix_edit.text(), start_spin.value(), digits_spin.value(),
            clear_check.isChecked(),
        )

    def _apply_batch_rename(self, paths_ordered, prefix, start, digits, clear):
        for i, p in enumerate(paths_ordered):
            if clear:
                self.thumbnail_list.set_display_name(p, "")
            else:
                name = f"{prefix}{str(start + i).zfill(digits)}"
                self.thumbnail_list.set_display_name(p, name)
        self._save_display_names()

    @staticmethod
    def _stored_geometry(config) -> bytes | None:
        raw = str(config.get("window_geometry") or "")
        if not raw:
            return None
        from PySide6.QtCore import QByteArray
        return QByteArray.fromBase64(raw.encode("ascii"))

    def _save_display_names(self):
        self.config["display_names"] = self.thumbnail_list.get_display_names()
        save_config(self.config)

    def clear_all_images(self):
        """Empty the working list (never touches files on disk)."""
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        total = self.thumbnail_list.count()
        if total == 0:
            return
        ret = QMessageBox.question(
            self, "清空列表",
            f"清空当前列表 {total} 张？（不影响磁盘上的原文件）",
        )
        if ret != QMessageBox.Yes:
            return
        self.thumbnail_list.clear_all()
        self.image_states.clear()
        self._state_lru.clear()
        self.current_path = None
        self.canvas.clear_image()
        self._dirty_paths.clear()
        self.config["display_names"] = {}
        save_config(self.config)
        self._update_thumb_stats()
        panel = getattr(self, "tool_panel", None)
        if panel is not None and hasattr(panel, "on_image_changed"):
            panel.on_image_changed()        # 列表空了 → 结果条回到「先导入一张图片」
        self.statusBar().showMessage("列表已清空")

    def _remove_selected_images(self, paths: list[str]):
        paths = [p for p in paths if p]
        if not paths:
            return
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        ret = QMessageBox.question(
            self, "移除图片",
            f"从列表移除 {len(paths)} 张？\n（不影响磁盘上的原文件）",
        )
        if ret != QMessageBox.Yes:
            return
        removed = self.thumbnail_list.remove_paths(paths)
        for p in removed:
            self.image_states.pop(p, None)
            if p in self._state_lru:
                self._state_lru.remove(p)
            # purge the display name from config so a same-named file
            # re-imported later doesn't resurrect the old caption
            self.config.get("display_names", {}).pop(p, None)
        if removed:
            save_config(self.config)
        if self.current_path in removed:
            remaining = self.thumbnail_list.item_paths()
            if remaining:
                # navigate explicitly; save_current=False so the already
                # removed old state is not resurrected by _save_current_state
                self.thumbnail_list.blockSignals(True)
                self.thumbnail_list.setCurrentRow(0)
                self.thumbnail_list.blockSignals(False)
                self._load_path(remaining[0], save_current=False)
            else:
                self.current_path = None
                self.canvas.clear_image()
        self._update_thumb_stats()
        panel = getattr(self, "tool_panel", None)
        if panel is not None and hasattr(panel, "on_image_changed"):
            panel.on_image_changed()        # 当前图被移除时结果条要跟着退

    # ------------------------------------------------------------------
    def auto_detect_current(self):
        """单张检测。v1.22.0：结果落到右侧面板的结果条，不再用弹窗打断。

        实测耗时（本机 DirectML，1024–1280 输入）0.12–0.18s，纯 CPU 回退
        0.96–1.97s —— 因此仍然同步执行，不引入线程（用户 2026-10-18 定）。
        """
        if self.current_path is None or self.canvas.image_np is None:
            self.tool_panel.on_image_changed()   # 页内提示「先导入一张图片」
            return
        settings = self.tool_panel.auto_settings()
        self.canvas.auto_params = settings
        try:
            rgb = self.canvas.image_np
            dets = self.detector.detect(
                rgb[:, :, ::-1],
                conf_threshold=settings["conf"],
                iou_threshold=settings["iou"],
                allowed_classes=settings["classes"],
                input_size=settings["input_size"],
            )
            if not dets:
                # 不清空旧掩码：检测不到 ≠ 用户想把已有的码擦掉
                self.tool_panel.show_detect_result([])
                self.statusBar().showMessage("未检测到需要打码的区域")
                return
            union = np.zeros(rgb.shape[:2], dtype=bool)
            for d in dets:
                union |= d.mask
            self.canvas.set_auto_mask(union)
            self.tool_panel.show_detect_result(dets)
            self.statusBar().showMessage(f"自动检测到 {len(dets)} 个目标")
        except Exception as e:
            self.tool_panel.show_detect_error(str(e))
            QMessageBox.critical(self, "自动打码失败", str(e))

    def batch_auto_censor(self):
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        paths = self.thumbnail_list.item_paths()
        if not paths:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        settings = self.tool_panel.auto_settings()
        self.canvas.auto_params = settings
        self.auto_worker = BatchAutoCensorWorker(
            paths, self.image_states, self.detector, settings, self
        )
        self.auto_worker.progress.connect(self._on_batch_progress)
        self.auto_worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.auto_worker.done.connect(self._on_auto_censor_done)
        self._set_auto_running_ui(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_cancel_btn.setVisible(True)
        self.auto_worker.start()

    def _set_auto_running_ui(self, running: bool):
        self.thumbnail_list.setEnabled(not running)
        btn = self.top_buttons.get("🤖 批量自动打码")
        if btn is not None:
            btn.setEnabled(not running)

    def _on_auto_censor_done(self, result):
        self._set_auto_running_ui(False)
        self.progress_bar.setVisible(False)
        self.progress_cancel_btn.setVisible(False)
        if self.current_path and self.current_path in self.image_states:
            # save_current=False: the batch worker already wrote the fresh
            # auto_mask into the state; saving the stale canvas state back
            # would overwrite it (the reason the first image kept losing its
            # censor result).
            self._load_path(self.current_path, save_current=False)
        self.thumbnail_list.refresh_badges(
            self.thumbnail_list.item_paths(), self._thumb_marks
        )
        self._update_thumb_stats()
        if result.get("cancelled"):
            QMessageBox.information(
                self, "批量自动打码",
                f"已取消：完成 {result.get('ok', 0)}/{result.get('total', 0)} 张",
            )
        else:
            QMessageBox.information(
                self, "批量自动打码",
                f"完成：{result.get('ok', 0)}/{result.get('total', 0)} 张检测到需要打码的区域",
            )

    def _any_worker_running(self) -> bool:
        return bool(
            (self.auto_worker and self.auto_worker.isRunning())
            or (self.batch_worker and self.batch_worker.isRunning())
            or (self.job_worker and self.job_worker.isRunning())
        )

    def clear_manual_selections(self):
        self.canvas._push_undo()
        self.canvas.selections = []
        self.canvas._rebuild()
        self.canvas.state_changed.emit()

    def add_current_sticker(self):
        item = self.tool_panel.sticker_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "提示", "请先选择贴纸。")
            return
        self.canvas.add_sticker(item.data(Qt.UserRole))

    # ------------------------------------------------------------------
    # Watermark helpers (watermarks are stickers with anchor+fractions)
    # ------------------------------------------------------------------
    def create_text_watermark(self, text, color, font_size=64, opacity=1.0,
                              stroke=0, stroke_fill=None):
        """Render text into a transparent RGBA image, add it to the sticker
        library (so it can be pasted as a watermark/sticker) and return the
        asset name. Returns None on empty text.

        The asset is ALSO saved into the sticker library folder, so it
        survives restarts exactly like imported stickers (one system)."""
        from ..core.image_ops import render_text_watermark
        text = (text or "").strip()
        if not text:
            return None
        # deterministic, file-friendly, collision-safe asset name
        import re as _re
        base = _re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text)[:24] or "watermark"
        stickers_dir = resource_path("app/stickers")
        name = base
        n = 1
        # collide against BOTH the in-memory library and the on-disk folder:
        # a name free in memory may still exist from a previous run
        while name in self.sticker_library or (stickers_dir / f"{name}.png").exists():
            n += 1
            name = f"{base}_{n}"
        layer = render_text_watermark(
            text, font_size=font_size, color=color, opacity=opacity,
            stroke=stroke,
            stroke_fill=stroke_fill or tuple(color[:3]) + (255,),
        )
        self.sticker_library[name] = layer
        self.canvas.sticker_library = self.sticker_library
        # persist like any library sticker (best effort: the in-memory asset
        # keeps working even if the folder is read-only)
        try:
            stickers_dir = resource_path("app/stickers")
            stickers_dir.mkdir(parents=True, exist_ok=True)
            layer.save(str(stickers_dir / f"{name}.png"))
        except Exception:
            pass
        return name

    def apply_watermark(self, scope="current", name=None, anchor="bottom-right",
                        size_frac=0.15, opacity=1.0):
        """Add a watermark (library image ``name``) anchored to each image.

        ``scope`` "current" adds it only to the current image; "all" adds it
        to every image in the list. Each image gets its own independent
        Sticker so a later manual move/resize only affects that one image."""
        if name is None or name not in self.sticker_library:
            raise ValueError("请先选择水印素材")
        targets = None
        if scope == "all":
            paths = self.thumbnail_list.item_paths()
            if not paths:
                raise ValueError("请先导入图片")
            targets = [(p, self.image_states.setdefault(p, {}))
                       for p in paths]
        elif scope == "current":
            if self.current_path is None:
                raise ValueError("当前没有图片")
            state = self.image_states.setdefault(self.current_path, {})
            targets = [(self.current_path, state)]
        else:
            raise ValueError(f"未知范围: {scope}")

        # Anchor pixel placement derives from the PREVIEW image size (the
        # canvas works in preview pixels); place() re-derives everything from
        # fractions at paste time, so the exact stored pixel size is only a
        # fallback for plain (non-watermark) stickers.
        from ..core.selection import Sticker
        for path, state in targets:
            h = w = 1
            if state.get("pil") is not None:
                w, h = state["pil"].size
            elif state.get("rgb") is not None:
                h, w = state["rgb"].shape[:2]
            short = min(w, h)
            wm_w = max(1.0, short * size_frac)
            # anchor=center places by nx/ny fractions: start at the middle.
            # Other anchors derive the position from the edge, nx/ny unused.
            start_nx = 0.5 if anchor == "center" else 1.0
            start_ny = 0.5 if anchor == "center" else 1.0
            st = Sticker(
                name, 0, 0, wm_w, wm_w,  # place() overrides when watermark
                nx=start_nx, ny=start_ny,
                size_frac=size_frac, anchor=anchor, opacity=opacity,
                watermark=True,
            )
            state.setdefault("stickers", []).append(st)

        # After a batch apply the current canvas must reflect its own new
        # watermark. _load_path replaces the canvas wholesale (clearing the
        # per-image undo stacks) - for a per-image watermark that is desired
        # (undo scope == image, not the batch).
        if scope == "current":
            self._load_path(self.current_path, save_current=False)
        elif self.current_path in self.image_states:
            self._load_path(self.current_path, save_current=False)
        self.refresh_badges_all()

    def refresh_badges_all(self):
        self.thumbnail_list.refresh_badges(
            self.thumbnail_list.item_paths(), self._thumb_marks
        )
        self._update_thumb_stats()

    def create_text_watermark_from_panel(self):
        """Called by the 水印 tab's 生成文字 button."""
        tp = self.tool_panel
        text = tp.wm_text.text().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请先输入文字内容。")
            return
        color = tp._wm_color_rgba()
        name = self.create_text_watermark(
            text, color=color,
            font_size=tp.wm_font.value(),
            opacity=tp.wm_opacity.value() / 100.0,
        )
        if name is None:
            return
        tp.refresh_watermark_assets()
        # also surface in the sticker library picker so it can be placed
        # as a normal sticker too
        ix = tp.wm_asset.findData(name)
        if ix >= 0:
            tp.wm_asset.setCurrentIndex(ix)
        self.statusBar().showMessage(f"已生成文字水印素材：{name}")

    # ------------------------------------------------------------------
    # Text watermark templates + watermark presets (config-backed)
    # ------------------------------------------------------------------
    def save_text_wm_template(self):
        tp = self.tool_panel
        text = tp.wm_text.text().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请先输入文字内容再保存模板。")
            return
        name, ok = QInputDialog.getText(self, "存为模板", "模板名称：",
                                        text=text[:20])
        if not ok or not name.strip():
            return
        name = name.strip()
        templates = list(self.config.get("text_wm_templates") or [])
        entry = {
            "name": name,
            "text": text,
            "color": list(tp._wm_color_rgba()),
            "font_size": tp.wm_font.value(),
            "opacity": tp.wm_opacity.value() / 100.0,
        }
        templates = [t for t in templates if t["name"] != name] + [entry]
        self.config["text_wm_templates"] = templates
        save_config(self.config)
        tp.refresh_text_wm_templates()
        ix = tp.wm_tpl.findData(name)
        if ix >= 0:
            tp.wm_tpl.setCurrentIndex(ix)
        self.statusBar().showMessage(f"文字模板已保存：{name}")

    def delete_text_wm_template(self):
        tp = self.tool_panel
        name = tp.wm_tpl.currentData()
        if not name:
            QMessageBox.information(self, "提示", "请先在下拉中选择要删除的模板。")
            return
        ret = QMessageBox.question(self, "删除模板", f"删除文字模板「{name}」？")
        if ret != QMessageBox.Yes:
            return
        templates = [t for t in (self.config.get("text_wm_templates") or [])
                     if t["name"] != name]
        self.config["text_wm_templates"] = templates
        save_config(self.config)
        tp.refresh_text_wm_templates()

    def _find_preset(self, name):
        for p in (self.config.get("watermark_presets") or []):
            if p["name"] == name:
                return p
        return None

    def save_watermark_preset(self):
        """Save the current page config as a NEW single-item preset."""
        tp = self.tool_panel
        if not tp.wm_asset.currentData():
            QMessageBox.warning(self, "提示", "请先选择水印素材。")
            return
        name, ok = QInputDialog.getText(self, "存为方案", "方案名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = [p for p in (self.config.get("watermark_presets") or [])
                   if p["name"] != name]
        presets.append({"name": name, "items": [tp.current_watermark_config()]})
        self.config["watermark_presets"] = presets
        save_config(self.config)
        tp.refresh_watermark_presets()
        ix = tp.wm_preset.findData(name)
        if ix >= 0:
            tp.wm_preset.setCurrentIndex(ix)
        self.statusBar().showMessage(f"水印方案已保存：{name}")

    def append_watermark_preset(self):
        """Append the current page config as an EXTRA item of the selected
        preset (multi-watermark presets: e.g. logo bottom-right + name
        top-left)."""
        tp = self.tool_panel
        name = tp.wm_preset.currentData()
        if not name:
            QMessageBox.information(self, "提示", "请先在方案下拉中选择一个方案。")
            return
        if not tp.wm_asset.currentData():
            QMessageBox.warning(self, "提示", "请先选择水印素材。")
            return
        preset = self._find_preset(name)
        if preset is None:
            return
        preset.setdefault("items", []).append(tp.current_watermark_config())
        save_config(self.config)
        tp.refresh_watermark_presets()
        self.statusBar().showMessage(
            f"已追加到方案「{name}」，现共 {len(preset['items'])} 项")

    def delete_watermark_preset(self):
        tp = self.tool_panel
        name = tp.wm_preset.currentData()
        if not name:
            QMessageBox.information(self, "提示", "请先在方案下拉中选择要删除的方案。")
            return
        ret = QMessageBox.question(self, "删除方案", f"删除水印方案「{name}」？")
        if ret != QMessageBox.Yes:
            return
        presets = [p for p in (self.config.get("watermark_presets") or [])
                   if p["name"] != name]
        self.config["watermark_presets"] = presets
        save_config(self.config)
        tp.refresh_watermark_presets()

    def apply_watermark_preset(self):
        """Apply every item of the selected preset to ALL images, in order."""
        tp = self.tool_panel
        name = tp.wm_preset.currentData()
        if not name:
            QMessageBox.information(self, "提示", "请先在方案下拉中选择一个方案。")
            return
        preset = self._find_preset(name)
        if not preset or not preset.get("items"):
            QMessageBox.information(self, "提示", "该方案为空。")
            return
        paths = self.thumbnail_list.item_paths()
        if not paths:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        missing = [it["asset"] for it in preset["items"]
                   if it.get("asset") not in self.sticker_library]
        if missing:
            QMessageBox.warning(
                self, "提示",
                f"方案里的素材不存在（可能已删除）：{', '.join(missing)}")
            return
        for it in preset["items"]:
            self.apply_watermark(scope="all", name=it["asset"],
                                 anchor=it.get("anchor", "bottom-right"),
                                 size_frac=it.get("size_frac", 0.15),
                                 opacity=it.get("opacity", 1.0))
        QMessageBox.information(
            self, "水印方案",
            f"方案「{name}」已应用：{len(preset['items'])} 项 × {len(paths)} 张。\n"
            "可切换到「指针」工具逐张拖动/缩放调整。")

    # ------------------------------------------------------------------
    def open_save_dialog(self):
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        self._save_current_state()
        if not self.thumbnail_list.item_paths():
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        last_export = self.config.get("last_export") or {}
        # 记住上次导出目录优先；没有记录才用当前图所在目录
        default_out = (str(last_export.get("out_dir"))
                       if last_export.get("out_dir") else
                       (str(Path(self.current_path).parent) if self.current_path else ""))
        dlg = SaveDialog(
            self,
            default_out=default_out,
            default_fmt=self.config.get("default_save_fmt", "PNG"),
            default_strip_meta=self.config.get("default_strip_meta", "strip"),
            last_rename=self.config.get("last_rename"),
            last_export=last_export,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.result_config()
        # remember the rename settings for next time
        self.config["last_rename"] = {
            "name_mode": cfg["name_mode"],
            "prefix": cfg["prefix"],
            "start": cfg["start"],
            "digits": cfg["digits"],
        }
        self.config["last_export"] = {
            "out_dir": cfg["out_dir"],
            "fmt": cfg["fmt"],
            "quality": cfg["quality"],
            "scale_mode": cfg["scale_mode"],
            "percent": cfg["percent"],
            "width": cfg["width"],
            "height": cfg["height"],
            "strip_metadata": cfg["strip_metadata"],
        }
        save_config(self.config)
        # preserve 模式下同名文件（不同文件夹的同名图）会互相覆盖——让用户决定
        if cfg["name_mode"] == "preserve":
            from ..core.export import find_name_conflicts
            conflicts = find_name_conflicts(
                self.thumbnail_list.item_paths(), cfg["fmt"]
            )
            if conflicts:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Warning)
                box.setWindowTitle("检测到同名文件")
                box.setText(
                    f"列表中有 {sum(len(v) for v in conflicts.values())} 张图片"
                    f"共用 {len(conflicts)} 个相同文件名（来自不同文件夹）。\n"
                    "按原名称导出时它们会互相覆盖，只保留最后一张。如何处理？"
                )
                dedupe_btn = box.addButton("自动去重（加 _2/_3 后缀）",
                                           QMessageBox.AcceptRole)
                overwrite_btn = box.addButton("仍然覆盖", QMessageBox.DestructiveRole)
                box.addButton("取消导出", QMessageBox.RejectRole)
                box.exec()
                clicked = box.clickedButton()
                if clicked is dedupe_btn:
                    cfg["on_name_conflict"] = "dedupe"
                elif clicked is overwrite_btn:
                    cfg["on_name_conflict"] = "overwrite"
                else:
                    return
        if not cfg["out_dir"]:
            if self.current_path:
                cfg["out_dir"] = str(Path(self.current_path).parent / "导出")
            else:
                cfg["out_dir"] = str(Path(self.thumbnail_list.item_paths()[0]).parent / "导出")
            self.statusBar().showMessage(f"输出目录：{cfg['out_dir']}")
        self._last_export_dir = cfg["out_dir"]
        auto_settings = self.tool_panel.auto_settings()
        self.batch_worker = BatchExportWorker(
            self.thumbnail_list.item_paths(),
            self.image_states,
            self.config,
            auto_settings,
            cfg,
            self.sticker_library,
            self,
            self.thumbnail_list.get_display_names(),
        )
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.batch_worker.done.connect(self._on_batch_done)
        self.batch_worker.start()
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def cancel_batch(self):
        if self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.cancel()
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
        if self.job_worker and self.job_worker.isRunning():
            self.job_worker.cancel()

    def _on_batch_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))

    def _on_batch_done(self, result):
        self.progress_bar.setVisible(False)
        # v1.17.0: clear only the paths that ACTUALLY exported -- a cancel
        # or partial failure used to disarm the warning for the rest
        for p in result.get("exported_paths") or []:
            self._dirty_paths.discard(p)
        if (result.get("processed", 0) > 0
                and self.config.get("auto_open_export_folder")
                and self._last_export_dir):
            self._open_export_folder(self._last_export_dir)
        lines = [
            f"共 {result.get('total', 0)} | 成功 {result.get('processed', 0)} "
            f"| 跳过 {result.get('skipped', 0)} | 失败 {result.get('failed', 0)}",
            "",
        ]
        for d in result.get("details", []):
            lines.append(f"[{d.get('status')}] {d.get('file')} - {d.get('info')}")
        dlg = QDialog(self)
        dlg.setWindowTitle("批量处理结果")
        dlg.resize(600, 400)
        lay = QVBoxLayout(dlg)
        txt = QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText("\n".join(lines))
        lay.addWidget(txt)
        btns = QHBoxLayout()
        out_dir = getattr(self, "_last_export_dir", "")
        btn_open = QPushButton("打开输出文件夹")
        btn_open.setEnabled(bool(out_dir) and os.path.isdir(out_dir))
        btn_open.clicked.connect(lambda: self._open_export_folder(out_dir))
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.accept)
        btns.addWidget(btn_open)
        btns.addStretch()
        btns.addWidget(btn_close)
        lay.addLayout(btns)
        dlg.exec()

    def _open_export_folder(self, out_dir: str):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(out_dir)))

    def _compose_path_image(self, path: str) -> Image.Image:
        """Return composed image for a path (main thread; reads UI settings)."""
        return compose_from_state(
            path, self.image_states, self.sticker_library,
            self.tool_panel.auto_settings(), self.config.get("params", {}),
        )

    # ------------------------------------------------------------------
    def _start_job(self, paths, job_fn, title):
        self.job_worker = BatchJobWorker(paths, job_fn, title, self)
        self.job_worker.progress.connect(self._on_batch_progress)
        self.job_worker.status.connect(lambda s: self.statusBar().showMessage(s))
        self.job_worker.done.connect(self._on_job_done)
        self._set_batch_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_cancel_btn.setVisible(True)
        self.job_worker.start()

    def _set_batch_ui_enabled(self, enabled: bool):
        for btn in (self.tool_panel.convert_btn,
                    self.tool_panel.resize_btn,
                    self.tool_panel.rename_btn):
            if btn is not None:
                btn.setEnabled(enabled)

    def _on_job_done(self, result):
        self._set_batch_ui_enabled(True)
        self.progress_bar.setVisible(False)
        self.progress_cancel_btn.setVisible(False)
        for p in result.get("done_paths") or []:
            self._dirty_paths.discard(p)  # these edits were saved out
        if result.get("cancelled"):
            QMessageBox.information(
                self, "批量处理",
                f"已取消：完成 {result.get('ok', 0)}/{result.get('total', 0)} 张",
            )
        else:
            QMessageBox.information(
                self, "批量处理",
                f"完成：{result.get('ok', 0)}/{result.get('total', 0)} 张"
                f"（失败 {result.get('failed', 0)}）",
            )

    # ------------------------------------------------------------------
    # Batch pipeline presets (config-backed action lists)
    # ------------------------------------------------------------------
    def save_batch_pipeline(self):
        tp = self.tool_panel
        cfg = tp.pipeline_config()
        if not cfg["actions"]:
            QMessageBox.warning(self, "提示", "请先添加动作再保存方案。")
            return
        name, ok = QInputDialog.getText(self, "存方案", "方案名称：")
        if not ok or not name.strip():
            return
        name = name.strip()
        presets = [p for p in (self.config.get("batch_pipelines") or [])
                   if p["name"] != name]
        presets.append({
            "name": name,
            "actions": cfg["actions"],
            "out_dir": cfg["out_dir"],
            "on_conflict": cfg["on_conflict"],
        })
        self.config["batch_pipelines"] = presets
        save_config(self.config)
        tp.refresh_pipeline_presets()
        ix = tp.pipe_preset.findData(name)
        if ix >= 0:
            tp.pipe_preset.setCurrentIndex(ix)
        self.statusBar().showMessage(f"批处理方案已保存：{name}")

    def delete_batch_pipeline(self):
        tp = self.tool_panel
        name = tp.pipe_preset.currentData()
        if not name:
            QMessageBox.information(self, "提示", "请先选择要删除的方案。")
            return
        ret = QMessageBox.question(self, "删方案", f"删除批处理方案「{name}」？")
        if ret != QMessageBox.Yes:
            return
        presets = [p for p in (self.config.get("batch_pipelines") or [])
                   if p["name"] != name]
        self.config["batch_pipelines"] = presets
        save_config(self.config)
        tp.refresh_pipeline_presets()

    def run_batch_pipeline(self):
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        files = self.thumbnail_list.item_paths()
        if not files:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        cfg = self.tool_panel.pipeline_config()
        if not cfg["actions"]:
            QMessageBox.warning(self, "提示", "请先添加至少一个处理动作。")
            return
        if not cfg["out_dir"]:
            QMessageBox.warning(self, "提示", "请选择输出目录。")
            return
        missing = [a["params"].get("asset") for a in cfg["actions"]
                   if a["type"] == "watermark"
                   and a["params"].get("asset") not in self.sticker_library]
        if missing:
            QMessageBox.warning(self, "提示",
                                f"水印素材不存在: {', '.join(missing)}")
            return
        # 执行前预览：逐张渲染全动作链的效果（不写工作区状态、不落盘），
        # 翻页确认后才真正执行。渲染用状态的独立副本。
        import copy as _copy
        from .widgets import PipelinePreviewDialog

        def _render_preview(preview_path):
            try:
                single = {preview_path: _copy.deepcopy(
                    self.image_states.get(preview_path, {}))}
                conv, _, _, _, _ = run_pipeline_on_image(
                    preview_path, single, cfg["actions"], self.sticker_library,
                    self.tool_panel.auto_settings(),
                    self.config.get("params", {}), self.detector, index=0,
                )
                return conv
            except Exception as e:
                print(f"预览失败 {preview_path}: {e}")
                return None

        dlg = PipelinePreviewDialog(files, _render_preview, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._last_export_dir = cfg["out_dir"]
        self._pipeline_worker = BatchPipelineWorker(
            files, self.image_states, cfg["actions"], self.sticker_library,
            self.tool_panel.auto_settings(), self.config.get("params", {}),
            self.detector, cfg["out_dir"], cfg["on_conflict"], self,
        )
        self._pipeline_worker.progress.connect(self._on_batch_progress)
        self._pipeline_worker.status.connect(
            lambda s: self.statusBar().showMessage(s))
        self._pipeline_worker.done.connect(self._on_pipeline_done)
        self._set_batch_ui_enabled(False)
        self.tool_panel.pipeline_btn.setEnabled(False)
        self._set_auto_running_ui(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_cancel_btn.setVisible(True)
        self._pipeline_worker.start()

    def _on_pipeline_done(self, result):
        self._set_batch_ui_enabled(True)
        self.tool_panel.pipeline_btn.setEnabled(True)
        self._set_auto_running_ui(False)
        self.progress_bar.setVisible(False)
        self.progress_cancel_btn.setVisible(False)
        for p in result.get("done_paths") or []:
            self._dirty_paths.discard(p)  # the pipeline exported these
        # worker mutated states (auto masks / watermarks): refresh the view
        if self.current_path:
            self._load_path(self.current_path, save_current=False)
        self.refresh_badges_all()
        if self.config.get("auto_open_export_folder") and self._last_export_dir:
            self._open_export_folder(self._last_export_dir)
        if result.get("cancelled"):
            QMessageBox.information(
                self, "批处理流水线",
                f"已取消：完成 {result.get('ok', 0)}/{result.get('total', 0)} 张")
            return
        msg = (f"完成：{result.get('ok', 0)}/{result.get('total', 0)} 张"
               f"（失败 {result.get('failed', 0)}）")
        errors = result.get("errors") or []
        detail = ""
        if errors:
            detail = "\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                detail += f"\n… 等共 {len(errors)} 个错误"
        QMessageBox.information(self, "批处理流水线", msg + detail)

    def batch_convert(self):
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        files = self.thumbnail_list.item_paths()
        if not files:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        out = self.tool_panel.convert_out.text().strip()
        if not out:
            QMessageBox.warning(self, "提示", "请选择输出目录。")
            return
        fmt = self.tool_panel.convert_fmt.currentText()
        quality = self.tool_panel.convert_quality.value()
        strip_meta = self.config.get("default_strip_meta", "strip") == "strip"
        params = self.config.get("params", {})
        auto_s = self.tool_panel.auto_settings()
        states = self.image_states
        sticker_lib = self.sticker_library
        os.makedirs(out, exist_ok=True)

        def job_fn(path, i):
            composed = compose_from_state(path, states, sticker_lib, auto_s, params)
            converted, _, pil_fmt, kwargs = save_processed(
                composed, fmt, quality, strip_meta
            )
            name = build_filename("IMG", 1, 3, i, pil_fmt)
            converted.save(os.path.join(out, name), **kwargs)
            converted.close()

        self._start_job(files, job_fn, "转格式")

    def batch_resize(self):
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        files = self.thumbnail_list.item_paths()
        if not files:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        out = self.tool_panel.resize_out.text().strip()
        if not out:
            QMessageBox.warning(self, "提示", "请选择输出目录。")
            return
        mode = self.tool_panel.resize_mode.currentData()
        percent = self.tool_panel.resize_percent.value()
        width = self.tool_panel.resize_width.value()
        height = self.tool_panel.resize_height.value()
        params = self.config.get("params", {})
        auto_s = self.tool_panel.auto_settings()
        states = self.image_states
        sticker_lib = self.sticker_library
        os.makedirs(out, exist_ok=True)

        def job_fn(path, i):
            img = compose_from_state(path, states, sticker_lib, auto_s, params)
            if mode == "percent":
                img = resize_image(img, percent=percent)
            else:
                img = resize_image(img, width=width, height=height)
            name = build_filename("RESIZED", 1, 3, i, "png")
            img.save(os.path.join(out, name))
            img.close()

        self._start_job(files, job_fn, "缩放")

    def batch_rename(self):
        if self._any_worker_running():
            QMessageBox.warning(self, "提示", "有后台任务正在进行，请稍候。")
            return
        files = self.thumbnail_list.item_paths()
        if not files:
            QMessageBox.warning(self, "提示", "请先导入图片。")
            return
        out = self.tool_panel.rename_out.text().strip()
        if not out:
            QMessageBox.warning(self, "提示", "请选择输出目录。")
            return
        prefix = self.tool_panel.rename_prefix.text().strip()
        start = self.tool_panel.rename_start.value()
        digits = self.tool_panel.rename_digits.value()
        # 预览：确认后才动任何文件（模板变量 {原名}/{日期}/{时间}/{序号}）
        rows = build_rename_plan(files, prefix, start, digits)
        dlg = RenamePreviewDialog(rows, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        params = self.config.get("params", {})
        auto_s = self.tool_panel.auto_settings()
        states = self.image_states
        sticker_lib = self.sticker_library
        quality = 100  # 重命名保持最高质量
        strip_meta = False
        os.makedirs(out, exist_ok=True)

        def job_fn(path, i):
            ext = Path(path).suffix.lstrip(".").lower() or "png"
            name = expand_rename_template(prefix, path, i, start, digits)
            dest = os.path.join(out, f"{name}.{ext}")
            state = states.get(path)
            if not state_has_edits(state):
                # 未编辑：字节级复制原文件（大小/质量/元数据 100% 保留）
                shutil.copyfile(path, dest)
                return
            img = compose_from_state(path, states, sticker_lib, auto_s, params)
            converted, _, pil_fmt, save_kwargs = save_processed(
                img, fmt_from_ext(ext), quality, strip_meta)
            converted.save(dest, pil_fmt, **save_kwargs)
            converted.close()

        self._start_job(files, job_fn, "重命名")

    # ------------------------------------------------------------------
    def open_settings(self):
        dlg = SettingsDialog(self.config, self)
        dlg.clearListRequested.connect(self.clear_all_images)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.config = dlg.config
            save_config(self.config)
            apply_theme(QApplication.instance(), self.config.get("theme", "xiaodu"))
            self.canvas.show_mask_overlay = self.config.get("show_mask_overlay", True)
            self.canvas.double_click_fit = self.config.get("double_click_fit", True)
            self.canvas._rebuild()
            self.tool_panel._apply_auto_defaults()
            self.tool_panel._update_auto_preview()
            want_providers = pick_providers(self.config.get("use_gpu", True))
            if (self.detector.model_path != self.config.get("model_path")
                    or self.detector.providers != want_providers):
                try:
                    self.detector = ONNXDetector(
                        self.config["model_path"], providers=want_providers
                    )
                    self.statusBar().showMessage("模型/推理设备已切换")
                except Exception as e:
                    QMessageBox.critical(self, "模型加载失败", str(e))

    def open_mini_window(self):
        if self._mini_window is None:
            from .mini_window import MiniWindow
            self._mini_window = MiniWindow(self.detector, self.config)
        else:
            self._mini_window.config = self.config
        self._mini_window.show()
        self._mini_window.raise_()

    # ------------------------------------------------------------------
    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() or md.hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        local_files = []
        remote_urls = []
        if md.hasUrls():
            for u in md.urls():
                local = u.toLocalFile()
                if local:
                    local_files.append(local)
                else:
                    url = u.toString()
                    if url.startswith(("http://", "https://")):
                        remote_urls.append(url)

        if local_files:
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
            files = []
            for u in local_files:
                p = Path(u)
                if p.is_dir():
                    files.extend(
                        str(x) for x in p.rglob("*")
                        if x.suffix.lower() in exts and x.is_file()
                    )
                elif p.suffix.lower() in exts:
                    files.append(u)
            files = sorted(files, key=lambda p: natural_key(Path(p).name))
            self._add_paths(files)

        for url in remote_urls:
            self._import_url(url)

        if md.hasText() and not local_files and not remote_urls:
            self._import_text(md.text())

        event.acceptProposedAction()

    def _import_url(self, url: str):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            ext = ".png"
            if "jpeg" in ctype or "jpg" in ctype:
                ext = ".jpg"
            elif "webp" in ctype:
                ext = ".webp"
            elif "gif" in ctype:
                ext = ".gif"
            tmp = Path(tempfile.gettempdir()) / "UnifiedImageTool_drop"
            tmp.mkdir(parents=True, exist_ok=True)
            path = tmp / ("download" + ext)
            path.write_bytes(resp.content)
            self._add_paths([str(path)])
        except Exception as e:
            QMessageBox.warning(self, "下载图片失败", str(e))

    def _import_text(self, text: str):
        text = text.strip()
        try:
            if text.startswith("data:image"):
                header, encoded = text.split(",", 1)
                raw = base64.b64decode(encoded)
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(raw))
                tmp = Path(tempfile.gettempdir()) / "UnifiedImageTool_drop"
                tmp.mkdir(parents=True, exist_ok=True)
                path = tmp / "base64.png"
                img.save(path)
                self._add_paths([str(path)])
                return
            if text.startswith(("http://", "https://")):
                self._import_url(text)
                return
            # try HTML img
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "html.parser")
            img = soup.find("img")
            src = img.get("src") if img else None
            if src:
                if src.startswith("data:image"):
                    self._import_text(src)
                elif src.startswith(("http://", "https://")):
                    self._import_url(src)
        except Exception as e:
            QMessageBox.warning(self, "导入失败", str(e))