# -*- coding: utf-8 -*-
"""v1.14.0 UI restyle tests.

Locks the things a re-skin can silently break:
  * the two panels stay draggable to their narrow limits (v1.7.2 regression)
  * both themes stay complete and renderable
  * the new card / setting-row / preview components behave
  * the watermark page keeps saying WHAT each knob does (the 字号 vs 大小
    confusion that started this whole restyle must not come back)
"""
import re

import numpy as np
import pytest
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QLabel, QPushButton, QSizePolicy,
)

from app.ui.theme import SPACE, XIAODU_QSS, XIAODUNIGHT_QSS, current_palette, theme_qss
from app.ui.widgets import SectionCard, SettingRow, WatermarkPreview, set_variant


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


_OPEN_WINDOWS: list = []


@pytest.fixture(autouse=True)
def _close_windows():
    """Close windows even when a test fails: each MainWindow installs an
    APPLICATION-level event filter, and a leftover one interferes with later
    tests."""
    yield
    while _OPEN_WINDOWS:
        w = _OPEN_WINDOWS.pop()
        try:
            w.close()
        except Exception:
            pass


def _mk_window(qapp, tmp_path, monkeypatch, n=3):
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


def _rgb_array(widget):
    widget.resize(widget.size().expandedTo(widget.minimumSizeHint()))
    img = widget.grab().toImage().convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8)
    return buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)


# ==================================================== narrow-panel guard

def test_panels_still_drag_to_their_narrow_limits(qapp, tmp_path, monkeypatch,
                                                  no_dialogs):
    """The v1.7.2 fix: the left panel must reach ~220px and the right panel
    must stay narrowable. A restyle that adds padding is exactly what used to
    break this, so it gets an explicit test."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    w.show()
    QApplication.instance().processEvents()
    sp = w.splitter

    # the right panel keeps its Ignored policy: content clips instead of
    # blocking the splitter (the hard requirement for narrow dragging)
    assert w.tool_panel.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored

    sp.setSizes([220, 800, 340])
    QApplication.instance().processEvents()
    assert sp.sizes()[0] <= 260, f"left panel stuck at {sp.sizes()[0]}"

    sp.setSizes([900, 400, 220])
    QApplication.instance().processEvents()
    assert sp.sizes()[2] <= 260, f"right panel stuck at {sp.sizes()[2]}"


def test_right_panel_minimum_hint_is_small(qapp, tmp_path, monkeypatch,
                                           no_dialogs):
    """Even without relying on Ignored, the content must not demand a wide
    panel (the restyle replaced QFormLayout rows with titled cards)."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    for i in range(w.tool_panel.count()):
        page = w.tool_panel.widget(i)
        page_min = page.minimumSizeHint().width()
        assert page_min <= 420, f"tab {i} needs {page_min}px"


def test_pages_fit_without_scrolling(qapp, tmp_path, monkeypatch, no_dialogs):
    """No tool-panel page may need more height than the panel has.

    The right panel is deliberately a plain QTabWidget with NO scroll area
    (v1.6.2 rollback), so an over-tall page has its bottom controls silently
    clipped. The v1.14 card layout made 水印 34 px too tall at the app's
    designed default window and squeezed the 位置 anchor grid to 33 px with
    overlapping rows -- the fix was trimming the page, and this is the lock.

    Asserted on minimumSizeHint (what Qt cannot compress below) at the app's
    own default window size, so it holds under either capture font."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    w.resize(1360, 840)          # MainWindow's designed default
    w.show()
    for _ in range(3):
        QApplication.instance().processEvents()
    panel = w.tool_panel
    avail = panel.height()
    assert avail > 400, f"panel did not lay out (height={avail})"
    for i in range(panel.count()):
        page = panel.widget(i)
        need = page.minimumSizeHint().height()
        assert need <= avail, (
            f"tab {panel.tabText(i)!r} needs {need}px but only {avail}px "
            f"available -> its bottom would be clipped")


# ==================================================== theme integrity

@pytest.mark.parametrize("name", ["xiaodu", "xiaodunight"])
def test_theme_qss_is_complete(name):
    qss = theme_qss(name)
    assert len(qss) > 3000, "stylesheet looks truncated"
    assert "{" in qss and "}" in qss
    # a missed double-brace in the template would leak a literal '{{'
    assert "{{" not in qss and "}}" not in qss
    # every surface we actually style must be present
    for sel in ("QPushButton", "QComboBox", "QSpinBox", "QTabBar::tab",
                "QMenu", "QToolTip", "QScrollBar", "QProgressBar",
                "QFrame#card", "QLabel#hint", "QLabel#cardTitle"):
        assert sel in qss, f"{sel} missing from the {name} stylesheet"


def test_two_skins_are_actually_different():
    assert XIAODU_QSS != XIAODUNIGHT_QSS
    # each skin must be a real stylesheet (the light one was once a 3-line stub)
    assert XIAODU_QSS.count("QPushButton") >= 3
    assert XIAODUNIGHT_QSS.count("QPushButton") >= 3


def _qss_without_comments(qss: str) -> str:
    """Strip /* ... */ blocks so assertions test RULES, not the comments."""
    return re.sub(r"/\*.*?\*/", "", qss, flags=re.S)


@pytest.mark.parametrize("name", ["xiaodu", "xiaodunight"])
def test_qss_arrow_rules_absent_and_indicator_rules_present(name):
    """Two rules, both learned from visual passes (v1.14 → amended v1.20):

    * ``QComboBox::drop-down`` / ``::down-arrow``: v1.14 found that styling
      them WITHOUT an image resource kills the arrow, so they were banned.
      v1.20 replaced the native arrow with a chevron IMAGE
      (xdd_arrow_{light,dark}.png vendored next to the theme module), which
      removes the native square button box AND keeps the arrow — so now the
      CONTRACT is inverted: the drop-down MUST be restyled AND the
      down-arrow MUST carry ``image: url(...)`` (drop that image and the
      arrow vanishes again).
    * ``QCheckBox::indicator`` / ``QRadioButton::indicator`` MUST be styled:
      the app-wide QWidget rule already puts those widgets on the
      QStyleSheetStyle path, where the native CHECKED indicator loses its box
      and only a floating tick survives.

    Comments are stripped first so this tests RULES, not prose.
    """
    qss = _qss_without_comments(theme_qss(name))
    assert "QComboBox::drop-down" in qss
    assert "QComboBox::down-arrow" in qss
    assert re.search(r"QComboBox::down-arrow\s*\{[^}]*image:\s*url\(", qss), \
        "down-arrow must carry its image url, or the chevron vanishes"
    for needed in ("QCheckBox::indicator", "QRadioButton::indicator",
                   "QCheckBox::indicator:checked",
                   "QRadioButton::indicator:checked"):
        assert needed in qss, f"{needed} missing from the {name} stylesheet"


def test_combos_in_setting_rows_are_not_collapsed(qapp, tmp_path, monkeypatch,
                                                  no_dialogs):
    """Regression lock for the worst v1.14 defect: a combo with an Ignored
    horizontal policy placed AFTER a stretch in a SettingRow got width 0 and
    vanished, so 效果 / 检测分辨率 / 自动效果 were unsettable."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    w.show()
    for _ in range(3):
        QApplication.instance().processEvents()
    panel = w.tool_panel
    for tab in range(panel.count()):
        panel.setCurrentIndex(tab)
        for _ in range(2):
            QApplication.instance().processEvents()
        for combo in panel.widget(tab).findChildren(QComboBox):
            if not combo.isVisible():
                continue
            assert combo.width() > 40, (
                f"combo {combo.objectName() or combo.count()} collapsed to "
                f"{combo.width()}px on tab {tab}")


def test_current_palette_follows_apply_theme(qapp):
    from app.ui.theme import apply_theme
    apply_theme(qapp, "xiaodu")
    assert current_palette()["canvas"] == "#FAF6EF"
    apply_theme(qapp, "xiaodunight")
    assert current_palette()["canvas"] == "#211E1A"


@pytest.mark.parametrize("name", ["xiaodu", "xiaodunight"])
def test_combo_popup_styles_its_selected_row(name):
    """Regression lock: the current combo item once rendered as a BLANK row.

    The popup view set ``selection-color`` to white while Qt skipped painting
    the selection BACKGROUND for the current row of a non-focused popup, so the
    selected item was white-on-white -- reported by the user as "正在使用的选项
    会变成空白消失". Two things must hold:

    * the popup view must NOT set ``selection-color`` (a text colour that can
      end up painted on a background nobody drew), and
    * the selected/hovered row must be drawn explicitly so text and background
      always agree.
    """
    qss = _qss_without_comments(theme_qss(name))
    start = qss.index("QComboBox QAbstractItemView {")
    view_block = qss[start:qss.index("}", start)]
    assert "selection-color" not in view_block, (
        "selection-color in the combo popup view can paint white text on an "
        "unpainted background")
    for needed in ("QComboBox QAbstractItemView::item {",
                   "QComboBox QAbstractItemView::item:selected",
                   "QComboBox QAbstractItemView::item:hover"):
        assert needed in qss, f"{needed} missing from the {name} stylesheet"


# ==================================================== components

def test_section_card_exposes_title_and_body(qapp):
    card = SectionCard("标题", "副标题说明")
    titles = [lbl.text() for lbl in card.findChildren(QLabel)
              if lbl.objectName() == "cardTitle"]
    hints = [lbl.text() for lbl in card.findChildren(QLabel)
             if lbl.objectName() == "hint"]
    assert titles == ["标题"]
    assert hints == ["副标题说明"]
    assert card.body is not None
    # add_row() must actually land in the card body
    btn = QPushButton("x")
    card.add_row("设置名", btn, note="说明文字")
    rows = card.findChildren(SettingRow)
    assert len(rows) == 1
    assert rows[0].title_label.text() == "设置名"
    assert rows[0].note_label is not None
    assert rows[0].note_label.text() == "说明文字"


def test_setting_row_without_note_has_no_note_label(qapp):
    row = SettingRow("名称", QPushButton("y"))
    assert row.note_label is None


def test_set_variant_tags_widget(qapp):
    btn = QPushButton("ok")
    set_variant(btn, "primary")
    assert btn.property("variant") == "primary"
    assert btn.property("variant") in theme_qss("light")


@pytest.mark.parametrize("theme", ["xiaodu", "xiaodunight"])
def test_checked_checkbox_keeps_a_filled_box(qapp, theme):
    """Rendered-pixel lock for the checked indicator.

    A visual review caught that a CHECKED checkbox lost its box/fill under the
    app-wide stylesheet (only a floating tick remained). The fix draws the
    indicator states explicitly; this asserts the accent fill really appears.

    Measured full-widget pixel colours for reference: light unchecked has an
    accent OUTLINE (~41 px, drawn by the base style) vs ~240 px of solid accent
    when checked; dark ~44 vs ~236."""
    from app.ui.theme import apply_theme
    # apply the theme HERE: this test must not depend on some earlier test
    # having left a stylesheet behind (xdist distributes files across
    # workers, so a leftover from another file cannot be relied on --
    # without this line the checkbox rendered unstyled and the count was 0)
    apply_theme(qapp, theme)
    accent = {"xiaodu": (36, 120, 168), "xiaodunight": (78, 156, 203)}[theme]
    acc = np.array(accent)

    def accent_px(checked: bool) -> int:
        cb = QCheckBox("X")
        cb.setChecked(checked)
        cb.resize(140, 22)
        cb.show()
        for _ in range(3):
            qapp.processEvents()
        img = cb.grab().toImage().convertToFormat(QImage.Format_RGB888)
        w, h = img.width(), img.height()
        buf = np.frombuffer(img.constBits(), np.uint8)
        arr = buf.reshape(h, img.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
        strip = arr[:, :20, :].astype(int)      # indicator column
        n = int((np.abs(strip - acc).sum(axis=2) < 90).sum())
        cb.hide()
        cb.deleteLater()
        qapp.processEvents()
        return n

    try:
        off, on = accent_px(False), accent_px(True)
        assert on > 150, f"checked indicator has no accent fill ({on}px)"
        assert on >= off * 3, f"checked {on}px vs unchecked {off}px"
    finally:
        apply_theme(qapp, "light")


# ==================================================== watermark preview

def _red_asset():
    im = Image.new("RGBA", (120, 40), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((0, 0, 119, 39), fill=(255, 0, 0, 255))
    return im


def test_watermark_preview_scales_with_size_frac(qapp):
    """The whole point of the preview: 大小 visibly changes the on-image size
    while 字号 does not. Verified on real rendered pixels."""
    pv = WatermarkPreview()
    pv.resize(240, 160)

    def red_pixels(frac):
        pv.set_state(asset=_red_asset(), size_frac=frac,
                     anchor="bottom-right", opacity=1.0)
        arr = _rgb_array(pv)
        return int(((arr[:, :, 0] > 180) & (arr[:, :, 1] < 90)
                    & (arr[:, :, 2] < 90)).sum())

    small = red_pixels(0.15)
    big = red_pixels(0.90)
    assert small > 0, "the watermark rectangle was not painted at all"
    assert big > small * 3, f"size_frac did not resize it ({small} -> {big})"


def test_watermark_preview_placeholder_without_asset(qapp):
    pv = WatermarkPreview()
    pv.resize(240, 160)
    pv.set_state(asset=None)
    arr = _rgb_array(pv)          # must render, not crash
    assert arr.shape[0] > 0


def test_watermark_preview_respects_anchor(qapp):
    """bottom-right vs top-left must put the watermark on different sides."""
    pv = WatermarkPreview()
    pv.resize(240, 160)

    def centre(anchor):
        pv.set_state(asset=_red_asset(), size_frac=0.3, anchor=anchor)
        arr = _rgb_array(pv)
        mask = ((arr[:, :, 0] > 180) & (arr[:, :, 1] < 90) & (arr[:, :, 2] < 90))
        ys, xs = np.where(mask)
        assert len(xs), "nothing painted"
        return xs.mean(), ys.mean()

    rx, ry = centre("bottom-right")
    lx, ly = centre("top-left")
    assert rx > lx and ry > ly


# ==================================================== the original confusion

def test_watermark_tab_labels_the_two_size_knobs(qapp, tmp_path, monkeypatch,
                                                 no_dialogs):
    """Regression lock for the bug report that started this restyle: 字号 and
    大小(短边%) were two identical spin boxes. Each must now carry its own
    label AND a plain-language note saying what it does."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    panel = w.tool_panel
    wt = panel.widget(3)  # tab 3 = 水印

    titles = [lbl.text() for lbl in wt.findChildren(QLabel)]
    joined = " ".join(titles)
    assert "文字精度（字号）" in joined, joined
    assert "图上大小（占短边）" in joined, joined

    notes = [lbl.text() for lbl in wt.findChildren(QLabel)
             if lbl.objectName() == "hint"]
    notes_joined = " ".join(notes)
    assert "不改变水印在照片上的大小" in notes_joined, notes_joined
    # v1.22.0：「水印宽度 = 图片短边的百分比」与行标题 + 控件 " %" 后缀三重重复，
    # 页内那行 note 删掉、改用 tooltip（挪走的 15px 才让九宫格不被压扁）。
    assert "图片短边" in panel.wm_size_row.toolTip(), panel.wm_size_row.toolTip()
    # the preview must be wired in
    assert panel.wm_preview is not None


def test_watermark_preview_updates_on_panel_edits(qapp, tmp_path, monkeypatch,
                                                  no_dialogs):
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    panel = w.tool_panel
    panel.wm_size.setValue(70)
    assert abs(panel.wm_preview._size_frac - 0.70) < 1e-6
    panel.wm_opacity.setValue(50)
    assert abs(panel.wm_preview._opacity - 0.5) < 1e-6
    panel.wm_anchor_picker.set_anchor("top-left")
    panel._refresh_wm_preview()
    assert panel.wm_preview._anchor == "top-left"


def test_watermark_asset_combo_sits_in_image_card(qapp, tmp_path, monkeypatch,
                                                  no_dialogs):
    """素材 used to sit above the text section, so it read as part of the text
    generator. It must now live inside the 图片水印 card."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    wt = w.tool_panel.widget(3)
    cards = wt.findChildren(SectionCard)
    titles = []
    for c in cards:
        for lbl in c.findChildren(QLabel):
            if lbl.objectName() == "cardTitle":
                titles.append(lbl.text())
    assert "图片水印" in titles
    assert "文字水印" in titles
    img_card = [c for c in cards
                if any(lbl.objectName() == "cardTitle" and lbl.text() == "图片水印"
                       for lbl in c.findChildren(QLabel))][0]
    assert img_card.isAncestorOf(w.tool_panel.wm_asset), \
        "素材 must be inside the 图片水印 card"


# ==================================================== primary buttons

def test_primary_actions_are_tagged(qapp, tmp_path, monkeypatch, no_dialogs):
    """The main action on each page should be visually primary, otherwise the
    panel is a wall of identical buttons."""
    w, _ = _mk_window(qapp, tmp_path, monkeypatch)
    panel = w.tool_panel
    primaries = []
    for i in range(panel.count()):
        page = panel.widget(i)
        for btn in page.findChildren(QPushButton):
            if btn.property("variant") == "primary":
                primaries.append((i, btn.text()))
    assert len(primaries) >= 4, primaries
    # the watermark page's main action must be the one that applies it
    assert any(t == "加到当前图" for _, t in primaries), primaries
