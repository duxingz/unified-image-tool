# -*- coding: utf-8 -*-
"""Thumbnail list widget."""
from __future__ import annotations

import os

from PIL import Image
from PySide6.QtCore import QPointF, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem

# Badge colors: auto censor / manual edits / stickers / watermarks
_BADGE_COLORS = {
    "manual": QColor("#FF9800"),
    "auto": QColor("#2196F3"),
    "sticker": QColor("#4CAF50"),
    "watermark": QColor("#9C27B0"),
}
_ORDER_COLOR = QColor("#1565C0")
_HIGHLIGHT_COLOR = QColor("#1E88E5")


def _pil_to_qpixmap(img: Image.Image, size: int) -> QPixmap:
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ThumbnailList(QListWidget):
    currentPathChanged = Signal(str)
    deleteRequested = Signal(list)
    displayNamesChanged = Signal()

    def __init__(self, parent=None, thumb_size=96):
        super().__init__(parent)
        self._thumb_size = thumb_size
        self._paths: list[str] = []
        self._base_pixmaps: dict[str, QPixmap] = {}
        self._badges: dict[str, dict] = {}
        self._highlight: set = set()  # paths briefly outlined after a drop
        # In-app display names (config-persisted; disk files are never renamed)
        self._display_names: dict[str, str] = {}
        self._suspend_item_changed = False
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setSelectionRectVisible(True)  # rubber-band multi select
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(thumb_size, thumb_size))
        self.setGridSize(QSize(thumb_size + 24, thumb_size + 44))
        self.setResizeMode(QListWidget.Adjust)
        # Drag-reorder is fully hand-rolled (mousePress/Move/Release below):
        # Qt's InternalMove lost the selection on drop, showed a forbidden
        # cursor over empty areas and gave no positional feedback at all.
        self.setMovement(QListWidget.Static)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.setDragEnabled(False)
        # rename goes through the context menu (editItem), never double-click
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.itemChanged.connect(self._on_item_changed)
        self._reorder_active = False
        self._press_item = None
        self._press_pos = None
        self._press_selected: set = set()
        self._reorder_item = None
        self._reorder_group: list = []
        self._mouse_pos = None
        self._autoscroll_dir = 0
        self._autoscroll_timer = QTimer(self)
        self._autoscroll_timer.setInterval(16)
        self._autoscroll_timer.timeout.connect(self._autoscroll_tick)
        self.setSpacing(6)
        self.setWordWrap(True)
        self.currentItemChanged.connect(self._on_current_changed)

    def set_paths(self, paths: list[str]):
        self._paths = list(paths)
        self._base_pixmaps.clear()
        self._badges.clear()
        self.clear()
        for p in paths:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, p)
            self.addItem(item)
            self._load_thumb(item, p)
        if self.count() > 0:
            self.setCurrentRow(0)

    def _load_thumb(self, item: QListWidgetItem, path: str):
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        display = self._display_names.get(path) or os.path.basename(path)
        self._suspend_item_changed = True
        try:
            with Image.open(path) as img:
                img.load()
                pix = _pil_to_qpixmap(img, self._thumb_size)
                item.setIcon(pix)
                item.setText(display)
                item.setToolTip(path)
                self._base_pixmaps[path] = pix
        except Exception:
            item.setText(display + "\n[无法加载]")
            item.setToolTip(path)
        finally:
            self._suspend_item_changed = False

    def _on_item_changed(self, item):
        """Inline rename finished: store the display name (empty restores the
        disk file name). Display names never touch the files on disk."""
        if self._suspend_item_changed or item is None:
            return
        path = item.data(Qt.UserRole)
        if not path:
            return
        new = (item.text() or "").split("\n")[0].strip()
        base = os.path.basename(path)
        if not new or new == base:
            self._display_names.pop(path, None)
        else:
            self._display_names[path] = new
        self._suspend_item_changed = True
        item.setText(self._display_names.get(path) or base)
        self._suspend_item_changed = False
        self.displayNamesChanged.emit()

    # ------------------------------------------------------------------
    # Display names (program-memory-only names for the thumbnail captions)
    # ------------------------------------------------------------------
    def display_name(self, path: str) -> str:
        return self._display_names.get(path) or os.path.basename(path)

    def get_display_names(self) -> dict:
        return dict(self._display_names)

    def set_display_names(self, names: dict):
        self._display_names = {
            k: v for k, v in (names or {}).items() if str(v).strip()
        }
        self._suspend_item_changed = True
        for i in range(self.count()):
            item = self.item(i)
            p = item.data(Qt.UserRole) if item is not None else None
            if p:
                item.setText(self._display_names.get(p) or os.path.basename(p))
        self._suspend_item_changed = False

    def rename_item(self, path: str):
        """Start inline rename for the given path (context-menu entry)."""
        item = self._item_for_path(path)
        if item is None:
            return
        self.setCurrentItem(item)
        # editItem() works even with NoEditTriggers (programmatic call)
        self.editItem(item)

    def set_display_name(self, path: str, name: str):
        name = (name or "").strip()
        item = self._item_for_path(path)
        if not name:
            self._display_names.pop(path, None)
        else:
            self._display_names[path] = name
        if item is not None:
            self._suspend_item_changed = True
            item.setText(name or os.path.basename(path))
            self._suspend_item_changed = False
        self.displayNamesChanged.emit()

    def clear_all(self):
        """Empty the whole working list (disk files are never touched)."""
        self._autoscroll_timer.stop()
        self._paths = []
        self._base_pixmaps.clear()
        self._badges.clear()
        self._display_names.clear()
        self._highlight.clear()
        self._reorder_active = False
        self._reorder_group = []
        self._press_item = None
        self._reorder_item = None
        self.clear()

    def item_paths(self) -> list[str]:
        return list(self._paths)

    def _sync_paths_from_items(self, *_args):
        """Keep _paths in item order after a drag-reorder (rowsMoved)."""
        paths = []
        for i in range(self.count()):
            item = self.item(i)
            p = item.data(Qt.UserRole) if item is not None else None
            if p:
                paths.append(p)
        if paths:
            self._paths = paths
        self._refresh_icons()
        moved = self.selected_paths()
        if moved:
            self._flash_highlight(moved)

    def _refresh_icons(self):
        """Repaint every item icon: base thumbnail + censor badges + order
        number (from 1, shown when 2+ images) + drag highlight."""
        show_order = self.count() >= 2
        for i in range(self.count()):
            item = self.item(i)
            p = item.data(Qt.UserRole) if item is not None else None
            base = self._base_pixmaps.get(p) if p else None
            if base is None:
                continue
            item.setIcon(self._paint_badge(
                base, self._badges.get(p, {}),
                order=(i + 1) if show_order else None,
                highlight=p in self._highlight,
            ))

    def _flash_highlight(self, paths):
        """Briefly outline the given items (drop feedback for drag-reorder)."""
        self._highlight = set(paths)
        self._refresh_icons()
        QTimer.singleShot(350, self._clear_highlight)

    def _clear_highlight(self):
        if self._highlight:
            self._highlight = set()
            self._refresh_icons()

    def selected_paths(self) -> list[str]:
        return [
            item.data(Qt.UserRole)
            for item in self.selectedItems()
            if item.data(Qt.UserRole)
        ]

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.SelectAll):
            self.selectAll()
            event.accept()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            paths = self.selected_paths()
            if paths:
                self.deleteRequested.emit(paths)
                event.accept()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Hand-rolled drag reorder: the dragged thumbnail physically moves
    # through the grid (other items get pushed aside in real time), the
    # order badges renumber live, and the item stays selected on release.
    # ------------------------------------------------------------------
    _REORDER_THRESHOLD = 4.0

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not (
            event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)
        ):
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                # remember the whole selection BEFORE the native press collapses
                # it - if the pressed item was part of a multi-selection, the
                # whole group moves together
                self._press_selected = set(self.selected_paths())
                self._press_item = item
                self._press_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._reorder_active:
            self._update_reorder(event.position())
            event.accept()
            return
        if (self._press_item is not None
                and not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier))):
            d = event.position() - self._press_pos
            if abs(d.x()) > self._REORDER_THRESHOLD or abs(d.y()) > self._REORDER_THRESHOLD:
                self._start_reorder()
                if self._reorder_active:
                    self._update_reorder(event.position())
                    event.accept()
                    return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._reorder_active:
            self._reorder_active = False
            self._autoscroll_timer.stop()
            item, self._reorder_item = self._reorder_item, None
            group, self._reorder_group = self._reorder_group, []
            self._press_item = None
            if item is not None:
                # the whole moved group stays selected (InternalMove dropped it)
                for it in group:
                    it.setSelected(True)
                self.setCurrentItem(item)
                self._sync_paths_from_items()  # also flashes the moved items
            event.accept()
            return
        self._press_item = None
        super().mouseReleaseEvent(event)

    def _start_reorder(self):
        if self._press_item is None:
            return
        self._reorder_active = True
        self._autoscroll_timer.start()
        self._reorder_item = self._press_item
        pressed_path = self._reorder_item.data(Qt.UserRole)
        if pressed_path and pressed_path in self._press_selected:
            # group drag: every selected item moves as one contiguous block
            wanted = set(self._press_selected)
            self._reorder_group = [
                self.item(i) for i in range(self.count())
                if (self.item(i).data(Qt.UserRole) or "") in wanted
            ]
        else:
            self._reorder_group = [self._reorder_item]
        self._highlight = {
            it.data(Qt.UserRole) for it in self._reorder_group
            if it.data(Qt.UserRole)
        }
        self._refresh_icons()

    def _update_reorder(self, pos):
        """Move the dragged block (one item or the whole multi-selection) to
        the slot under the cursor: it lands where the cursor points and
        everything in between is pushed aside in real time."""
        pos = QPointF(pos)  # tolerate QPoint callers (tests, programmatic use)
        self._mouse_pos = pos
        self._update_autoscroll(pos)
        group_set = {id(it) for it in self._reorder_group}
        items_order = [self.item(i) for i in range(self.count())]
        group = [it for it in items_order if id(it) in group_set]
        others = [it for it in items_order if id(it) not in group_set]
        k = self._insert_index_in_others(pos, items_order, others)
        new_order = others[:k] + group + others[k:]
        if [id(x) for x in new_order] == [id(x) for x in items_order]:
            return
        # incremental resequence (takeItem+insertItem) - a full clear+re-add
        # breaks the grid flow and lays everything out on one line
        for target_pos in range(len(new_order)):
            it = new_order[target_pos]
            cur = self.row(it)
            if cur != target_pos:
                taken = self.takeItem(cur)
                self.insertItem(target_pos, taken)
        for it in group:
            it.setSelected(True)
        self.setCurrentItem(group[-1])
        self._refresh_icons()  # order badges renumber live

    def _insert_index_in_others(self, pos, items_order, others) -> int:
        """Insertion slot within ``others`` for the cursor position.

        Over a thumbnail: left half inserts before it, right half after it
        (single-column layout uses top/bottom instead). Over a gap: the
        nearest grid cell decides. Past the end: the very end."""
        if not others:
            return 0
        others_ids = {id(x) for x in others}
        hit = self.itemAt(pos.toPoint())
        if hit is not None and id(hit) not in others_ids:
            hit = None
        if hit is not None:
            rect = self.visualItemRect(hit)
            multi_col = rect.width() < self.viewport().width() * 0.8
            after = (pos.x() > rect.center().x()) if multi_col else (
                pos.y() > rect.center().y())
            return min(len(others), others.index(hit) + (1 if after else 0))
        # gaps between thumbnails: nearest grid cell
        grid = self.gridSize()
        cols = max(1, self.viewport().width() // grid.width())
        guess = (int(pos.y()) // grid.height()) * cols + int(pos.x()) // grid.width()
        if guess >= len(items_order):
            return len(others)
        before_ids = {id(x) for x in items_order[:guess + 1]}
        return sum(1 for x in others if id(x) in before_ids)

    def _update_autoscroll(self, pos):
        h = self.viewport().height()
        margin = 60
        self._autoscroll_dir = 1 if pos.y() > h - margin else (
            -1 if pos.y() < margin else 0)

    def _autoscroll_tick(self):
        if not self._reorder_active or self._autoscroll_dir == 0                 or self._mouse_pos is None:
            return
        sb = self.verticalScrollBar()
        sb.setValue(sb.value() + self._autoscroll_dir * 12)
        self._update_reorder(self._mouse_pos)

    def _target_row(self, pos, cur):
        """Insert row for the cursor position (None = keep current).

        Dragging onto the left half of a thumbnail inserts before it, onto
        the right half inserts after it - so the hovered item visibly gets
        pushed aside in the direction of the drag. In single-column layout
        the top/bottom half decides instead. Dropping below the last item
        moves the image to the end."""
        it = self.itemAt(pos.toPoint())
        count = self.count()
        if it is None:
            # gaps between thumbnails: locate the nearest grid cell so the
            # push-aside keeps working while the cursor travels the gaps
            grid = self.gridSize()
            cols = max(1, self.viewport().width() // grid.width())
            guess = (int(pos.y()) // grid.height()) * cols + int(pos.x()) // grid.width()
            if guess >= count:
                last = count - 1
                rect = self.visualItemRect(self.item(last)) if last >= 0 else None
                if rect is not None and pos.y() >= rect.bottom() and last != cur:
                    return last
                return cur
            it = self.item(min(guess, count - 1))
            if it is None or it is self._reorder_item:
                return cur
        if it is self._reorder_item:
            return cur
        row = self.row(it)
        rect = self.visualItemRect(it)
        multi_col = rect.width() < self.viewport().width() * 0.8
        if multi_col:
            after = pos.x() > rect.center().x()
        else:
            after = pos.y() > rect.center().y()
        target = row + 1 if after else row
        if cur < target:
            target -= 1  # removing the dragged item first shifts the slot
        return max(0, min(count - 1, target))

    def sort_paths(self, key, reverse: bool = False):
        """Reorder the list by ``key`` (e.g. natural_key); keeps the current
        item selected. Pixmaps/badges follow the items (path-keyed caches)."""
        if len(self._paths) < 2:
            return
        current = self.current_path()
        order = sorted(
            range(len(self._paths)), key=lambda i: key(self._paths[i]), reverse=reverse
        )
        items = [self.item(i) for i in range(self.count())]
        for i in range(self.count()):
            self.takeItem(0)
        for idx in order:
            self.addItem(items[idx])
        self._paths = [self._paths[idx] for idx in order]
        if current is not None:
            item = self._item_for_path(current)
            if item is not None:
                self.setCurrentItem(item)
        self._refresh_icons()

    def remove_paths(self, paths: list[str]) -> list[str]:
        """Remove images from the list (not from disk). Returns removed paths."""
        wanted = set(paths)
        if not wanted:
            return []
        removed = []
        for i in range(self.count() - 1, -1, -1):
            item = self.item(i)
            p = item.data(Qt.UserRole) if item is not None else None
            if p in wanted:
                self.takeItem(i)
                removed.append(p)
                self._base_pixmaps.pop(p, None)
                self._badges.pop(p, None)
        self._paths = [p for p in self._paths if p not in wanted]
        for p in removed:
            self._display_names.pop(p, None)
        self._refresh_icons()
        return removed

    def current_path(self) -> str | None:
        item = self.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _on_current_changed(self, current, _previous):
        if current is not None:
            path = current.data(Qt.UserRole)
            if path:
                self.currentPathChanged.emit(path)

    def add_paths(self, paths: list[str]):
        existing = set(self._paths)
        new_paths = [p for p in paths if p not in existing]
        if not new_paths:
            return
        self._paths.extend(new_paths)
        for p in new_paths:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, p)
            self.addItem(item)
            self._load_thumb(item, p)
        self._refresh_icons()

    # ------------------------------------------------------------------
    # Censor-status badges (auto / manual / sticker)
    # ------------------------------------------------------------------
    def set_badge(self, path: str, marks: dict):
        """Update the corner badge for one image.

        ``marks`` is a dict with optional bool keys ``auto`` (auto censor
        mask present), ``manual`` (manual selections present) and
        ``sticker`` (stickers placed)."""
        base = self._base_pixmaps.get(path)
        if base is None:
            return
        marks = {k: bool(v) for k, v in (marks or {}).items()}
        self._badges[path] = marks
        item = self._item_for_path(path)
        if item is None:
            return
        row = self.row(item)
        order = (row + 1) if self.count() >= 2 else None
        item.setIcon(self._paint_badge(
            base, marks, order=order, highlight=path in self._highlight
        ))
        active = [k for k in ("manual", "auto", "sticker") if marks.get(k)]
        if active:
            label = {"manual": "手动", "auto": "自动", "sticker": "贴纸"}
            item.setToolTip(f"{path}\n已打码: {'+'.join(label[k] for k in active)}")

    def refresh_badges(self, paths, status_fn):
        for p in paths:
            self.set_badge(p, status_fn(p))

    def _item_for_path(self, path: str):
        for i in range(self.count()):
            item = self.item(i)
            if item is not None and item.data(Qt.UserRole) == path:
                return item
        return None

    @staticmethod
    def _paint_badge(base: QPixmap, marks: dict, order=None, highlight=False) -> QPixmap:
        # copy the base pixmap (same format) and draw badges on top; with no
        # marks/order/highlight the result is pixel-identical to the original
        pm = QPixmap(base)
        painter = QPainter(pm)
        if highlight:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(_HIGHLIGHT_COLOR, 3))
            painter.drawRect(1, 1, pm.width() - 3, pm.height() - 3)
        # rightmost = manual, then auto, sticker, leftmost = watermark
        # (bottom-right corner; a watermark badge is purple)
        s = 9
        margin = 2
        x = pm.width() - margin
        for key in ("manual", "auto", "sticker", "watermark"):
            if not marks.get(key):
                continue
            painter.fillRect(
                QRect(x - s, pm.height() - margin - s, s, s), _BADGE_COLORS[key]
            )
            x -= s + 2
        if order is not None:
            # top-left: dark blue circle with the 1-based export order
            d = 16
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(_ORDER_COLOR)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(2, 2, d, d)
            painter.setPen(QColor("white"))
            font = QFont()
            font.setPointSize(7)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QRect(2, 2, d, d), Qt.AlignCenter, str(order))
        painter.end()
        return pm
        return pm