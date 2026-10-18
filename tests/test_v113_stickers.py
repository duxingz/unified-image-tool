# -*- coding: utf-8 -*-
"""v1.13.0 sticker-system tests.

Covers the gaps closed in this version:
  * rotation UI + the three render sites staying in sync (pixel-verified)
  * Delete key on the canvas removes the SELECTED sticker only
  * canvas context menu (delete / duplicate / rotate / flip / restack)
  * sticker library management (rename / delete / favourite / search)
  * watermark tab "remove every watermark"
  * numpad image navigation
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent, QKeyEvent
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu

from app.core.export import _scale_sticker
from app.core.image_ops import compose_image
from app.core.selection import Sticker
from app.ui.editor_canvas import EditorCanvas


@pytest.fixture()
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))


class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        return []


# ------------------------------------------------------------------ helpers

def _src_sticker_img(w=80, h=40):
    """Asymmetric source: red plate with a blue block in the TOP-LEFT, so a
    rotation or a mirror is detectable in the composited pixels."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w - 1, h - 1), fill=(255, 0, 0, 255))
    d.rectangle((0, 0, w // 5 - 1, h // 4 - 1), fill=(0, 0, 255, 255))
    return im


@pytest.fixture()
def canvas(qapp):
    c = EditorCanvas()
    base = Image.new("RGBA", (400, 300), (255, 255, 255, 255))
    c.image_np = np.array(base.convert("RGB"))
    c.original_pil = base
    c.sticker_library = {"s": _src_sticker_img()}
    return c


_OPEN_WINDOWS: list = []


@pytest.fixture(autouse=True)
def _close_windows():
    """Close every window a test opened, even when it failed.

    Each MainWindow installs an APPLICATION-level event filter for numpad
    navigation; a leftover one would answer -- and consume -- numpad keys for
    later tests, which makes failures look impossible to reproduce."""
    yield
    while _OPEN_WINDOWS:
        w = _OPEN_WINDOWS.pop()
        try:
            w.close()
        except Exception:
            pass


def _mk_window(qapp, tmp_path, monkeypatch, n=3):
    """MainWindow with the sticker library folder redirected to tmp_path."""
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow
    from app.core.config import default_config

    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: tmp_path / rel)
    w = MainWindow(StubDetector(), default_config())
    _OPEN_WINDOWS.append(w)
    paths = []
    for i in range(n):
        arr = np.full((240, 320, 3), 60 + i * 30, dtype=np.uint8)
        p = tmp_path / f"img{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    w.thumbnail_list.add_paths(paths)
    w._load_path(paths[0], save_current=False)
    return w, paths


def _painted_bbox(canvas, st):
    out = compose_image(canvas.original_pil, [], [], [st], canvas.sticker_library)
    a = np.array(out)
    ys, xs = np.where((a[:, :, :3] != 255).any(axis=2))
    assert len(xs), "nothing was painted"
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ============================================================ rotation

@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 180.0, 225.0])
def test_rotation_outline_matches_painted_pixels(canvas, angle):
    """The selection outline must track the rotated sticker, not just its
    unrotated box (the whole reason _sticker_corners_scene exists)."""
    st = Sticker("s", 200, 150, 80, 40, angle=angle)
    canvas.stickers = [st]
    box = canvas._sticker_bbox_scene(st)
    outline = (round(box.left()), round(box.top()),
               round(box.right()), round(box.bottom()))
    painted = _painted_bbox(canvas, st)
    err = max(abs(painted[i] - outline[i]) for i in range(4))
    assert err <= 2, f"angle={angle} painted={painted} outline={outline}"


def test_rotation_hit_test_uses_rotated_box(canvas):
    """A point that is only inside the ROTATED sticker must select it, and a
    point that is only inside the unrotated box must not."""
    st = Sticker("s", 200, 150, 80, 40, angle=90.0)
    canvas.stickers = [st]
    # rotated 90 deg -> 40 wide, 80 tall
    assert canvas._sticker_at(QPointF(200, 150 + 35)) == 0
    # (200+35, 150) is inside the UNROTATED 80x40 box but outside the rotated one
    assert canvas._sticker_at(QPointF(200 + 35, 150)) == -1


def test_rotate_handle_exists_for_stickers_only(canvas):
    st = Sticker("s", 200, 150, 80, 40)
    canvas.stickers = [st]
    canvas._sel_sticker = 0
    rp = canvas._rotate_handle_pos(st)
    assert rp is not None
    assert canvas._handle_at(rp) == "rot"

    # a watermark must never offer rotation (place() resolves it from the
    # image edge, so spinning it would break the anchor contract)
    wm = Sticker("s", 0, 0, 40, 20, nx=0.9, ny=0.9, size_frac=0.2,
                 anchor="bottom-right", watermark=True)
    canvas.stickers = [wm]
    canvas._sel_sticker = 0
    assert canvas._handle_at(canvas._rotate_handle_pos(wm)) != "rot"
    assert canvas.set_sticker_angle(45.0) is False
    assert wm.angle == 0.0


def test_rotate_drag_snaps_to_15_degrees(canvas):
    st = Sticker("s", 200, 150, 80, 40)
    canvas.stickers = [st]
    canvas._sel_sticker = 0
    rp = canvas._rotate_handle_pos(st)          # straight above the centre
    assert canvas._begin_sticker_drag(rp) is True
    assert canvas._sticker_drag == "rot"
    canvas._drag_sticker_to(QPointF(300, 150))  # 90 deg to the right
    canvas._sticker_drag = None
    assert st.angle == 90.0

    # 47 deg of pointer travel snaps to 45, not to 47
    st.angle = 0.0
    assert canvas._begin_sticker_drag(canvas._rotate_handle_pos(st)) is True
    canvas._drag_sticker_to(QPointF(200 + 73.1, 150 - 68.2))
    canvas._sticker_drag = None
    assert st.angle == 45.0, st.angle


def test_rotate_is_undoable(canvas):
    st = Sticker("s", 200, 150, 80, 40)
    canvas.stickers = [st]
    canvas._sel_sticker = 0
    assert canvas.set_sticker_angle(90.0) is True
    assert st.angle == 90.0
    canvas.undo()
    assert canvas.stickers[0].angle == 0.0


def test_resize_keeps_rotation_and_fixed_corner(canvas):
    """Corner resize must work in the sticker's own axes: at 90 deg the
    opposite corner must stay put instead of the box flying off."""
    st = Sticker("s", 200, 150, 80, 40, angle=90.0)
    canvas.stickers = [st]
    canvas._sel_sticker = 0
    pts = canvas._sticker_corners_scene(st)
    fixed_before = pts[2]  # "tl" drag keeps "br" fixed
    assert canvas._begin_sticker_drag(pts[0]) is True
    assert canvas._sticker_drag == "tl"
    canvas._drag_sticker_to(QPointF(pts[0].x() - 40, pts[0].y() - 40))
    canvas._sticker_drag = None
    assert st.angle == 90.0, "resizing must not change the angle"
    fixed_after = canvas._sticker_corners_scene(st)[2]
    assert abs(fixed_after.x() - fixed_before.x()) <= 1.5
    assert abs(fixed_after.y() - fixed_before.y()) <= 1.5


def test_rotation_grip_actually_renders(canvas):
    """Rendered-pixel check, not a data-layer one: this project has been bitten
    twice by checks that passed while the screen stayed blank."""
    from PySide6.QtGui import QImage

    canvas.resize(420, 320)
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    canvas._sel_sticker = 0
    canvas._rebuild(fit=True)
    img = canvas.grab().toImage().convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8)
    arr = buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
    # amber grip #FFC107
    amber = ((arr[:, :, 0] > 220) & (arr[:, :, 1] > 150)
             & (arr[:, :, 1] < 215) & (arr[:, :, 2] < 90))
    assert amber.any(), "the rotation grip was not rendered on screen"

    # ... and a watermark must NOT get one
    wm = Sticker("s", 0, 0, 40, 20, nx=0.9, ny=0.9, size_frac=0.2,
                 anchor="bottom-right", watermark=True)
    canvas.stickers = [wm]
    canvas._sel_sticker = 0
    canvas._rebuild(fit=True)
    img = canvas.grab().toImage().convertToFormat(QImage.Format_RGB888)
    buf = np.frombuffer(img.constBits(), np.uint8)
    arr = buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
    amber = ((arr[:, :, 0] > 220) & (arr[:, :, 1] > 150)
             & (arr[:, :, 1] < 215) & (arr[:, :, 2] < 90))
    assert not amber.any(), "a watermark must not offer rotation"


# ============================================================ flips

def test_flip_h_mirrors_painted_pixels(canvas):
    st = Sticker("s", 200, 150, 80, 40)
    out = np.array(compose_image(canvas.original_pil, [], [], [st],
                                 canvas.sticker_library))
    blue = np.where((out[:, :, 2] > 200) & (out[:, :, 0] < 100))
    red = np.where((out[:, :, 0] > 200) & (out[:, :, 2] < 100))
    assert blue[1].mean() < red[1].mean(), "blue block should start on the left"

    canvas.stickers = [st]
    canvas._sel_sticker = 0
    assert canvas.flip_selected_sticker(True) is True
    out = np.array(compose_image(canvas.original_pil, [], [], [st],
                                 canvas.sticker_library))
    blue = np.where((out[:, :, 2] > 200) & (out[:, :, 0] < 100))
    red = np.where((out[:, :, 0] > 200) & (out[:, :, 2] < 100))
    assert blue[1].mean() > red[1].mean(), "flip_h must move it to the right"


def test_flip_v_mirrors_painted_pixels(canvas):
    st = Sticker("s", 200, 150, 80, 40)
    canvas.stickers = [st]
    canvas._sel_sticker = 0
    assert canvas.flip_selected_sticker(False) is True
    out = np.array(compose_image(canvas.original_pil, [], [], [st],
                                 canvas.sticker_library))
    blue = np.where((out[:, :, 2] > 200) & (out[:, :, 0] < 100))
    red = np.where((out[:, :, 0] > 200) & (out[:, :, 2] < 100))
    assert blue[0].mean() > red[0].mean(), "flip_v must move it to the bottom"


def test_scale_sticker_keeps_flip_and_opacity():
    st = Sticker("s", 10, 20, 80, 40, angle=30.0, opacity=0.5,
                 flip_h=True, flip_v=True)
    scaled = _scale_sticker(st, 2.0, 3.0)
    assert (scaled.x, scaled.y) == (20, 60)
    assert (scaled.width, scaled.height) == (160, 120)
    assert scaled.angle == 30.0
    assert scaled.opacity == 0.5
    assert scaled.flip_h is True and scaled.flip_v is True


def test_export_keeps_flip(qapp, tmp_path):
    """End-to-end delivery check: the canvas shows a flipped sticker, so the
    EXPORTED file must too. compose_from_state rescales plain stickers through
    _scale_sticker, so a dropped flip there would only show in the bytes."""
    from app.core.export import compose_from_state

    src_p = tmp_path / "big.png"
    Image.new("RGB", (400, 200), (255, 255, 255)).save(src_p)
    preview = Image.new("RGB", (200, 100), (255, 255, 255))

    st = Sticker("s", 100, 50, 80, 40)
    st.flip_h = True
    state = {"pil": preview, "orig_size": (400, 200), "stickers": [st]}
    out = compose_from_state(str(src_p), {str(src_p): state},
                             {"s": _src_sticker_img()},
                             {"auto_effect": "blur", "auto_strength": 10}, {})
    assert out.size == (400, 200), "export must keep the original resolution"
    a = np.array(out.convert("RGB"))
    blue = np.where((a[:, :, 2] > 200) & (a[:, :, 0] < 100))
    red = np.where((a[:, :, 0] > 200) & (a[:, :, 2] < 100))
    assert len(blue[0]) and len(red[0]), "the sticker must be in the export"
    assert blue[1].mean() > red[1].mean(), "flip_h was lost in the export"


def test_export_keeps_rotation(qapp, tmp_path):
    """A 90 deg sticker is 40 wide x 80 tall on screen, so the exported pixels
    must be tall too (and the export must stay at full resolution)."""
    from app.core.export import compose_from_state

    src_p = tmp_path / "big2.png"
    Image.new("RGB", (400, 200), (255, 255, 255)).save(src_p)
    preview = Image.new("RGB", (200, 100), (255, 255, 255))

    st = Sticker("s", 100, 50, 80, 40, angle=90.0)
    state = {"pil": preview, "orig_size": (400, 200), "stickers": [st]}
    out = compose_from_state(str(src_p), {str(src_p): state},
                             {"s": _src_sticker_img()},
                             {"auto_effect": "blur", "auto_strength": 10}, {})
    a = np.array(out.convert("RGB"))
    ys, xs = np.where((a[:, :, :3] != 255).any(axis=2))
    w = int(xs.max()) - int(xs.min()) + 1
    h = int(ys.max()) - int(ys.min()) + 1
    assert h > w, f"expected a tall rotated sticker, got {w}x{h}"


# ============================================================ delete key

def test_delete_key_removes_only_selected_sticker(canvas):
    """Delete must take the sticker, NOT a manual selection (the top-bar
    button keeps its own layered order, locked by test_v153_features)."""
    from app.core.selection import RectSelection

    canvas.selections = [RectSelection(10, 10, 40, 40, effect="blur", strength=8)]
    st = Sticker("s", 200, 150, 80, 40)
    canvas.stickers = [st]
    canvas._sel_sticker = 0

    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Delete, Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(ev)
    assert canvas.stickers == [], "Delete must remove the selected sticker"
    assert len(canvas.selections) == 1, "manual selections must be untouched"

    canvas.undo()
    assert len(canvas.stickers) == 1, "delete must be undoable"


def test_delete_key_without_selection_does_nothing(canvas):
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    canvas._sel_sticker = -1
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    canvas.keyPressEvent(ev)
    assert len(canvas.stickers) == 1


# ============================================================ clipboard

def test_copy_paste_duplicates_sticker_with_offset(canvas):
    st = Sticker("s", 200, 150, 80, 40, angle=25.0)
    canvas.stickers = [st]
    canvas._sel_sticker = 0
    assert canvas.copy_selected_sticker() is True
    assert canvas.paste_sticker_from_clipboard() is True
    assert len(canvas.stickers) == 2
    dup = canvas.stickers[1]
    assert canvas._sel_sticker == 1
    assert (dup.x, dup.y) != (st.x, st.y), "the copy must not hide the original"
    assert dup.angle == 25.0
    canvas.undo()
    assert len(canvas.stickers) == 1


def test_paste_without_copy_is_a_noop(canvas):
    assert canvas.paste_sticker_from_clipboard() is False


# ============================================================ restacking

def test_restack_moves_selected_sticker(canvas):
    a = Sticker("s", 100, 100, 40, 20)
    b = Sticker("s", 200, 200, 40, 20)
    canvas.stickers = [a, b]
    canvas._sel_sticker = 0
    assert canvas.move_selected_sticker(True) is True
    assert canvas.stickers == [b, a]
    assert canvas._sel_sticker == 1, "selection must follow the sticker"
    assert canvas.move_selected_sticker(False) is True
    assert canvas.stickers == [a, b]
    assert canvas._sel_sticker == 0


# ============================================================ context menu

def _fire_context_menu(canvas, scene_xy, press_viewport=None):
    """Fire a context-menu event with the real exec neutralised.

    QMenu.exec cannot be monkeypatched (PySide6 resolves it from the native
    method table), so the canvas exposes sticker_context_menu() separately and
    this helper swaps it for a recorder."""
    calls = []
    canvas.sticker_context_menu = lambda idx: calls.append(idx)
    canvas._rmb_press_pos = (QPointF(*press_viewport) if press_viewport
                             else None)
    vp = canvas.mapFromScene(QPointF(*scene_xy))
    canvas.contextMenuEvent(
        QContextMenuEvent(QContextMenuEvent.Reason.Mouse, vp, vp))
    return calls


def test_context_menu_lists_sticker_actions(canvas):
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    menu = canvas.sticker_context_menu(0)
    assert menu is not None
    labels = [a.text() for a in menu.actions() if a.text()]
    for want in ("删除该贴纸", "复制一个", "旋转 90°", "水平翻转",
                 "垂直翻转", "移到最上层", "移到最下层"):
        assert want in labels, f"{want} missing from {labels}"
    assert canvas._sel_sticker == 0, "opening the menu must select the sticker"


def test_context_menu_disables_rotation_for_watermarks(canvas):
    wm = Sticker("s", 0, 0, 40, 20, nx=0.5, ny=0.5, size_frac=0.2,
                 anchor="center", watermark=True)
    canvas.stickers = [wm]
    menu = canvas.sticker_context_menu(0)
    assert menu is not None
    by_text = {a.text(): a for a in menu.actions() if a.text()}
    assert by_text["旋转 90°"].isEnabled() is False
    assert by_text["角度归零"].isEnabled() is False
    assert by_text["水平翻转"].isEnabled() is True


def test_context_menu_opens_only_on_a_sticker(canvas):
    """Right-clicking empty space must not offer sticker actions."""
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    assert _fire_context_menu(canvas, (10, 10)) == [], "empty space"
    assert _fire_context_menu(canvas, (200, 150)) == [0], "on the sticker"


def test_context_menu_suppressed_after_pan_drag(canvas):
    """Right-drag pans; the menu arrives on release, so a drag must not pop it."""
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    target = canvas.mapFromScene(QPointF(200, 150))
    # press 200px away in viewport terms -> that was a pan, not a click
    calls = _fire_context_menu(canvas, (200, 150),
                               press_viewport=(target.x() - 200,
                                               target.y() - 200))
    assert calls == [], "a pan drag must not open the sticker menu"


def test_context_menu_action_actually_deletes(canvas):
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    menu = canvas.sticker_context_menu(0)
    for act in menu.actions():
        if act.text() == "删除该贴纸":
            act.trigger()
            break
    assert canvas.stickers == []
    canvas.undo()
    assert len(canvas.stickers) == 1


def test_context_menu_rotate_action_works(canvas):
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    menu = canvas.sticker_context_menu(0)
    for act in menu.actions():
        if act.text() == "旋转 90°":
            act.trigger()
            break
    assert canvas.stickers[0].angle == 90.0


# ============================================================ remove watermarks

def test_remove_watermarks_keeps_plain_stickers(canvas):
    wm1 = Sticker("s", 0, 0, 40, 20, nx=0.5, ny=0.5, size_frac=0.2,
                  anchor="center", watermark=True)
    wm2 = Sticker("s", 0, 0, 40, 20, nx=0.4, ny=0.4, size_frac=0.2,
                  anchor="center", watermark=True)
    plain = Sticker("s", 200, 150, 80, 40)
    canvas.stickers = [wm1, wm2, plain]
    assert canvas.remove_watermarks() == 2
    assert canvas.stickers == [plain]
    canvas.undo()
    assert len(canvas.stickers) == 3


def test_remove_watermarks_noop_when_none(canvas):
    canvas.stickers = [Sticker("s", 200, 150, 80, 40)]
    assert canvas.remove_watermarks() == 0


# ============================================================ library mgmt

def _seed_library(w, tmp_path, names=("logo", "star")):
    """Write fake assets into the redirected sticker folder."""
    d = tmp_path / "app" / "stickers"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        _src_sticker_img().save(d / f"{n}.png")
    w._load_stickers()
    return d


def test_delete_sticker_asset_removes_file_and_entry(qapp, tmp_path,
                                                     monkeypatch, no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    d = _seed_library(w, tmp_path)
    assert "logo" in w.sticker_library
    w.delete_sticker_asset("logo")
    assert not (d / "logo.png").exists(), "the file must be gone"
    assert "logo" not in w.sticker_library
    assert "star" in w.sticker_library, "other assets must survive"


def test_delete_sticker_asset_unknown_name_is_safe(qapp, tmp_path, monkeypatch,
                                                   no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    _seed_library(w, tmp_path)
    w.delete_sticker_asset("does-not-exist")   # must not raise
    assert "logo" in w.sticker_library


def test_delete_sticker_asset_drops_favourite(qapp, tmp_path, monkeypatch,
                                              no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    _seed_library(w, tmp_path)
    w.toggle_sticker_favorite("logo")
    assert "logo" in w.config["sticker_favorites"]
    w.delete_sticker_asset("logo")
    assert "logo" not in (w.config.get("sticker_favorites") or [])


def test_rename_sticker_asset_repoints_placements(qapp, tmp_path, monkeypatch,
                                                  no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    d = _seed_library(w, tmp_path)
    st = Sticker("logo", 100, 100, 40, 20)
    w.canvas.stickers.append(st)
    w.image_states[w.current_path] = {"stickers": [st]}
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("brand", True)))

    w.rename_sticker_asset("logo")
    assert not (d / "logo.png").exists()
    assert (d / "brand.png").exists()
    assert "brand" in w.sticker_library and "logo" not in w.sticker_library
    assert st.name == "brand", "existing placements must follow the rename"


def test_rename_sticker_asset_rejects_duplicate(qapp, tmp_path, monkeypatch,
                                                no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    d = _seed_library(w, tmp_path)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("star", True)))
    w.rename_sticker_asset("logo")
    assert (d / "logo.png").exists(), "must not clobber the existing asset"
    assert (d / "star.png").exists()


def test_rename_sticker_asset_sanitises_illegal_chars(qapp, tmp_path,
                                                      monkeypatch, no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    d = _seed_library(w, tmp_path)
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("a/b:c", True)))
    w.rename_sticker_asset("logo")
    assert (d / "a_b_c.png").exists()


def test_toggle_sticker_favorite_persists(qapp, tmp_path, monkeypatch,
                                          no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    _seed_library(w, tmp_path)
    w.toggle_sticker_favorite("logo")
    assert w.config["sticker_favorites"] == ["logo"]
    w.toggle_sticker_favorite("logo")
    assert w.config["sticker_favorites"] == []


def test_sticker_search_and_favourites_filter(qapp, tmp_path, monkeypatch,
                                              no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    _seed_library(w, tmp_path, names=("logo", "logo_dark", "star"))

    def shown():
        return [w.tool_panel.sticker_list.item(i).data(Qt.UserRole)
                for i in range(w.tool_panel.sticker_list.count())]

    assert sorted(shown()) == ["logo", "logo_dark", "star"]
    w.tool_panel.sticker_filter.setText("logo")
    assert sorted(shown()) == ["logo", "logo_dark"]
    w.tool_panel.sticker_filter.setText("")
    w.toggle_sticker_favorite("star")
    w.tool_panel.sticker_fav_only.setChecked(True)
    assert shown() == ["star"], "favourites-only must hide the rest"


# ============================================================ numpad nav

def _numpad(monkeypatch, w, key):
    """Send a keypad key press and report whether it was consumed."""
    ev = QKeyEvent(QEvent.Type.KeyPress, key,
                   Qt.KeyboardModifier.KeypadModifier)
    monkeypatch.setattr(w, "isActiveWindow", lambda: True)
    return QApplication.instance().sendEvent(w.thumbnail_list, ev)


@pytest.mark.parametrize("key,expected_row", [
    (Qt.Key_Up, 0), (Qt.Key_4, 0),          # NumLock off / on -> previous
    (Qt.Key_Down, 2), (Qt.Key_2, 2),        # NumLock off / on -> next
    (Qt.Key_Left, 0), (Qt.Key_Right, 2),    # horizontal pad, same mapping
])
def test_numpad_navigates_images(qapp, tmp_path, monkeypatch, no_dialogs,
                                 key, expected_row):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, n=3)
    w.thumbnail_list.setCurrentRow(1)
    _numpad(monkeypatch, w, key)
    assert w.thumbnail_list.currentRow() == expected_row


def test_numpad_is_clamped_at_the_ends(qapp, tmp_path, monkeypatch, no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, n=2)
    w.thumbnail_list.setCurrentRow(0)
    _numpad(monkeypatch, w, Qt.Key_Up)
    assert w.thumbnail_list.currentRow() == 0, "no wrap-around at the top"
    w.thumbnail_list.setCurrentRow(1)
    _numpad(monkeypatch, w, Qt.Key_Down)
    assert w.thumbnail_list.currentRow() == 1, "no wrap-around at the bottom"


def test_plain_arrow_keys_are_claimed_globally(qapp, tmp_path, monkeypatch,
                                              no_dialogs):
    """v1.14.4: the REAL arrow keys now switch images.

    v1.13 deliberately left them alone so the canvas kept arrow-key scrolling;
    the user then reported "箭头键无法全局移动图片" and chose global navigation
    over canvas scrolling (wheel zoom + drag pan remain). The test asserts the
    new contract, and that widgets which legitimately own the arrows still keep
    them."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, n=3)
    monkeypatch.setattr(w, "isActiveWindow", lambda: True)
    w.thumbnail_list.setCurrentRow(1)

    plain = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up,
                      Qt.KeyboardModifier.NoModifier)
    assert w.eventFilter(w.canvas, plain) is True, "real arrows must navigate"
    w.thumbnail_list.setCurrentRow(1)
    pad = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up,
                    Qt.KeyboardModifier.KeypadModifier)
    assert w.eventFilter(w.canvas, pad) is True, "keypad arrows must still work"

    # a modifier makes it somebody else's key
    ctrl = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up,
                     Qt.KeyboardModifier.ControlModifier)
    assert w.eventFilter(w.canvas, ctrl) is False


@pytest.mark.parametrize("widget_attr", [
    "wm_font",            # QSpinBox
    "sticker_filter",     # QLineEdit
    "wm_asset",           # QComboBox
    "sticker_list",       # QListWidget (an item view owns its arrows)
])
def test_arrow_keys_are_left_to_widgets_that_own_them(qapp, tmp_path, monkeypatch,
                                                      no_dialogs, widget_attr):
    """Global navigation must not steal arrows from a field/list/combo."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, n=3)
    monkeypatch.setattr(w, "isActiveWindow", lambda: True)
    target = getattr(w.tool_panel, widget_attr)
    target.show()
    target.setFocus()
    qapp.processEvents()
    assert w.focusWidget() is target, "fixture sanity: the widget has focus"
    w.thumbnail_list.setCurrentRow(1)
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up,
                   Qt.KeyboardModifier.NoModifier)
    assert w.eventFilter(target, ev) is False, \
        f"{widget_attr} must keep its own arrow keys"


def test_numpad_ignored_when_window_is_not_active(qapp, tmp_path, monkeypatch,
                                                  no_dialogs):
    """A modal dialog of this app must not have the list navigating behind it."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, n=3)
    w.thumbnail_list.setCurrentRow(1)
    monkeypatch.setattr(w, "isActiveWindow", lambda: False)
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Up,
                   Qt.KeyboardModifier.KeypadModifier)
    QApplication.instance().sendEvent(w.thumbnail_list, ev)
    assert w.thumbnail_list.currentRow() == 1


def test_numpad_ignored_while_typing(qapp, tmp_path, monkeypatch, no_dialogs):
    """The numpad must still type into a text field."""
    from PySide6.QtWidgets import QLineEdit
    w, _ = _mk_window(qapp, tmp_path, monkeypatch, n=3)
    w.thumbnail_list.setCurrentRow(1)
    edit = QLineEdit(w.tool_panel)
    edit.setFocus()
    monkeypatch.setattr(w, "isActiveWindow", lambda: True)
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_2, Qt.KeyboardModifier.KeypadModifier)
    QApplication.instance().sendEvent(edit, ev)
    assert w.thumbnail_list.currentRow() == 1, "typing must not navigate"
