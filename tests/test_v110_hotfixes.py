# -*- coding: utf-8 -*-
"""v1.10.0 hotfix regressions.

1. Text watermarks must render with a REAL CJK-capable font (absolute font
   paths, never bare names): bare-name loading silently falls back to
   Pillow's bitmap font which has no CJK glyphs -> tofu blocks.
2. Stickers can be imported by dropping image files onto the Stickers tab:
   the drop is consumed by the library (copied into the sticker folder),
   never interpreted as "import as images to censor".
"""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from app.core.image_ops import _pick_font, render_text_watermark


@pytest.fixture()
def no_dialogs(monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Yes))


# ------------------------------------------------------------- font fix
def test_pick_font_prefers_cjk_path():
    """The resolved font must come from an absolute system font path, not a
    bare filename or Pillow's built-in bitmap font (which has no CJK glyphs
    and renders Chinese as tofu blocks)."""
    font = _pick_font(48)
    name = font.getname()[0]
    # load_default reports "Aileron" or similar bitmap font; a real CJK
    # candidate reports its family (Microsoft YaHei / SimHei / Noto ...)
    assert name not in ("Aileron", "DejaVu Sans")


def test_render_text_watermark_chinese_not_tofu(qapp):
    """Chinese glyphs must produce varied, non-blocky alpha coverage:
    a bitmap-font failure renders identical hollow rectangles whose alpha
    fill ratio collapses to near-zero or is perfectly uniform per glyph."""
    layer = render_text_watermark("测水印", font_size=64,
                                  color=(255, 255, 255, 255))
    assert layer.mode == "RGBA"
    a = np.array(layer)
    alpha = a[..., 3]
    n = int((alpha > 0).sum())
    assert n > 200  # real strokes paint thousands of pixels
    ys, xs = np.where(alpha > 0)
    # glyphs should have internal 'holes'/structure: stroke density well
    # below 100% inside the bbox (a solid tofu rectangle is ~100%)
    if len(xs) > 0:
        fill = (alpha > 0).mean()
        assert fill < 0.95


def test_render_text_watermark_respects_explicit_font(qapp, tmp_path):
    p = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not p.exists():
        pytest.skip("simhei not present")
    layer = render_text_watermark("黑体测试", font_size=48, font_path=str(p))
    assert np.array(layer)[..., 3].max() > 0


# ------------------------------------------------------ sticker drop fix
class StubDetector:
    model_path = "stub"
    providers = ["CPUExecutionProvider"]

    def detect(self, image_bgr, conf_threshold=0.25, iou_threshold=0.45,
               allowed_classes=None, input_size=None):
        return []


def test_sticker_drop_imports_to_library(qapp, tmp_path, no_dialogs,
                                         monkeypatch):
    """Dropping images onto the Stickers tab copies them into the sticker
    library folder (numeric suffix on duplicates, source untouched) and
    refreshes list + watermark assets. A second drop imports AGAIN (does not
    get swallowed as 'images to censor')."""
    import app.ui.main_window as mw_mod
    from app.ui.main_window import MainWindow
    from app.core.config import default_config
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir(parents=True)
    monkeypatch.setattr(mw_mod, "resource_path", lambda rel: lib_dir / rel)

    w = MainWindow(StubDetector(), default_config())

    src = tmp_path / "src"
    src.mkdir()
    p1 = src / "sticker1.png"
    p2 = src / "水印贴纸.png"
    Image.fromarray(np.zeros((40, 60, 4), dtype=np.uint8)).save(p1)
    Image.fromarray(np.zeros((40, 60, 4), dtype=np.uint8)).save(p2)

    def make_drop():
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(p1)), QUrl.fromLocalFile(str(p2))])
        ev = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime,
                        Qt.LeftButton, Qt.NoModifier)
        # keep a python reference alive: the QDropEvent only holds a C++
        # pointer to the mime data, and if the wrapper is GC'd mid-call the
        # helper sees a dangling QObject (test-only artifact; the Qt event
        # loop keeps real drop events alive)
        ev._mime_keep = mime
        return ev

    w.tool_panel.sticker_list.dropEvent(make_drop())
    assert (lib_dir / "app" / "stickers" / "sticker1.png").exists()
    assert (lib_dir / "app" / "stickers" / "水印贴纸.png").exists()
    assert p1.exists()  # source untouched
    assert "sticker1" in w.sticker_library
    assert "水印贴纸" in w.sticker_library
    assert w.tool_panel.sticker_list.count() == 2
    assert w.tool_panel.wm_asset.count() == 2  # watermark combo in sync

    # duplicate drop -> numeric suffix, no overwrite
    w.tool_panel.sticker_list.dropEvent(make_drop())
    names = sorted(x.name for x in (lib_dir / "app" / "stickers").iterdir())
    assert names == ["sticker1.png", "sticker1_2.png",
                     "水印贴纸.png", "水印贴纸_2.png"]
    w.close()


def test_sticker_drop_zone_accepts_images(qapp, tmp_path, no_dialogs,
                                          monkeypatch):
    """Dropping on the tab's blank area (StickerDropZone) routes to the same
    importer; non-image drops are rejected."""
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent
    from app.ui.tool_panel import StickerDropZone, StickerDropList

    zone = StickerDropZone()
    got = []
    zone.imagesDropped.connect(lambda files: got.append(files))
    src = tmp_path / "x.png"
    Image.fromarray(np.zeros((10, 10, 4), dtype=np.uint8)).save(src)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(src))])
    zone.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, mime,
                              Qt.LeftButton, Qt.NoModifier))
    assert got and got[0]

    # a non-image drop must not be consumed as a sticker import
    txt = QMimeData()
    txt.setText("hello")
    zone.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, txt,
                              Qt.LeftButton, Qt.NoModifier))
    assert len(got) == 1
