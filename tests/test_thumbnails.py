# -*- coding: utf-8 -*-
"""Thumbnail badge tests (offscreen)."""
from PIL import Image

from app.ui.thumbnail_list import ThumbnailList


def test_badge_updates_icon_and_tooltip(qapp, tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (200, 150), (10, 20, 30)).save(p)
    tl = ThumbnailList()
    tl.add_paths([str(p)])
    item = tl.item(0)
    base_pm = tl._base_pixmaps[str(p)]

    def icon_matches_base():
        return item.icon().pixmap(base_pm.size()).toImage() == base_pm.toImage()

    assert icon_matches_base()

    tl.set_badge(str(p), {"auto": True, "manual": False, "sticker": True})
    assert not icon_matches_base()
    tip = item.toolTip()
    assert "已打码" in tip
    assert "自动" in tip and "贴纸" in tip and "手动" not in tip

    # clearing the badge restores the plain icon
    tl.set_badge(str(p), {})
    assert icon_matches_base()


def test_refresh_badges_applies_status_fn(qapp, tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    Image.new("RGB", (120, 120), (1, 2, 3)).save(p1)
    Image.new("RGB", (120, 120), (4, 5, 6)).save(p2)
    tl = ThumbnailList()
    tl.add_paths([str(p1), str(p2)])
    status_fn = lambda path: {"auto": path == str(p1), "manual": False, "sticker": False}
    tl.refresh_badges([str(p1), str(p2)], status_fn)
    assert "已打码" in tl.item(0).toolTip()
    assert "已打码" not in tl.item(1).toolTip()