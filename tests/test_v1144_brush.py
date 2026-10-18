# -*- coding: utf-8 -*-
"""v1.14.4 — brush/eraser WYSIWYG (three user-reported problems).

a) the brush cursor was a fixed 16..64 SCREEN-pixel circle while the brush paints
   in IMAGE pixels -> when zoomed out it looked up to 2x bigger than reality
   (users thought they had erased more than they had: "橡皮擦经常擦不干净"),
   and when zoomed in it looked smaller than the paint;
b) the in-drag preview differed from the released result -- it applied the
   effect repeatedly to already-affected pixels (accumulating, worst with
   light/white_mist) and used a DIFFERENT stroke geometry than the committed
   BrushSelection (circles per sampled point vs PIL line segments);
c) the eraser's own commit was already correct -- measured 100 % coverage -- so
   the "not clean" feeling came from (a).
"""
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from app.core.image_ops import apply_mask_effect
from app.core.selection import BrushSelection
from app.ui.editor_canvas import EditorCanvas

W, H = 400, 300


def texture():
    """Fine texture: any blur/block effect changes it measurably."""
    rng = np.random.default_rng(11)
    return Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), "RGB")


@pytest.fixture()
def canvas(qapp):
    cv = EditorCanvas()
    cv.image_np = np.array(texture())
    cv.original_pil = texture()
    cv.sticker_library = {}
    cv._rebuild()
    return cv


def _drag(cv, pts, render_each_frame=True):
    """Drive the real brush/eraser mouse path over ``pts``.

    The order matters: the real mouseMoveEvent appends the point and THEN the
    30ms throttle calls _render_drag_stroke, so the final point is rendered too
    (rendering before the append would leave the preview one frame behind and
    make this comparison meaningless)."""
    vp0 = cv.mapFromScene(QPointF(*pts[0]))
    cv.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(vp0),
                                   QPointF(vp0), Qt.MouseButton.LeftButton,
                                   Qt.MouseButton.LeftButton,
                                   Qt.KeyboardModifier.NoModifier))
    for p in pts[1:]:
        vp = cv.mapFromScene(QPointF(*p))
        cv.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(vp),
                                      QPointF(vp), Qt.MouseButton.NoButton,
                                      Qt.MouseButton.LeftButton,
                                      Qt.KeyboardModifier.NoModifier))
        if render_each_frame:
            cv._render_drag_stroke()
    if render_each_frame:
        cv._render_drag_stroke()          # flush the tail, as the throttle does
    vpe = cv.mapFromScene(QPointF(*pts[-1]))
    cv.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(vpe),
                                     QPointF(vpe), Qt.MouseButton.LeftButton,
                                     Qt.MouseButton.NoButton,
                                     Qt.KeyboardModifier.NoModifier))


STROKE = [(x, 150) for x in range(80, 321, 6)]


# ================================================== b: preview == final

def test_stroke_geometry_is_shared(canvas):
    """The preview mask and the committed mask must be the SAME shape.

    They used to be two independent implementations (circles per sampled point
    vs PIL line segments + end caps), so a stroke visibly changed at release."""
    canvas.set_brush_size(30)
    preview = canvas._stroke_mask(STROKE, 30)
    committed = np.array(BrushSelection(STROKE, 30).to_mask((W, H)).convert("L")) > 127
    assert preview.shape == committed.shape
    diff = int((preview ^ committed).sum())
    assert diff == 0, f"preview and committed stroke shapes differ by {diff}px"


def test_shared_geometry_draws_into_a_crop(canvas):
    """The bbox-aware form (used by the incremental preview) must agree with the
    full-image form inside the crop."""
    canvas.set_brush_size(24)
    full = canvas._stroke_mask(STROKE, 24)
    x1, y1, x2, y2 = 150, 100, 300, 200
    crop = canvas._stroke_mask(STROKE, 24, bbox=(x1, y1, x2, y2))
    assert crop.shape == (y2 - y1, x2 - x1)
    assert int((crop ^ full[y1:y2, x1:x2]).sum()) == 0


@pytest.mark.parametrize("effect", ["blur", "pixel", "black_lines", "white_mist"])
def test_preview_matches_released_result(canvas, effect):
    """What the drag shows must equal what mouse-release produces.

    Asserts on PIXELS: the old incremental code rendered onto the already
    affected live image, so every overlapping frame re-applied the effect."""
    canvas.tool = "brush"
    canvas.current_effect = effect
    canvas.current_strength = 30
    _drag(canvas, STROKE)
    live = getattr(canvas, "_live_img", None)
    final = canvas._composed_base.copy()
    band = (slice(120, 180), slice(80, 320))
    if live is None:                       # release leaves no live image
        return
    d = float(np.abs(live[band].astype(int) - final[band].astype(int)).mean())
    assert d < 1.0, f"{effect}: preview differs from the released result by {d:.2f}/255"


def test_light_preview_matches_released_result(canvas):
    """The light effect derives its radial centre from the mask's bounding box
    INSIDE the rendered image, so the preview must render the whole stroke's
    box -- otherwise the centre moves and the preview is visibly wrong."""
    canvas.tool = "brush"
    canvas.current_effect = "light"
    canvas.current_strength = 30
    _drag(canvas, STROKE)
    live = canvas._live_img
    final = canvas._composed_base.copy()
    band = (slice(120, 180), slice(80, 320))
    d = float(np.abs(live[band].astype(int) - final[band].astype(int)).mean())
    assert d < 1.0, f"light: preview differs from the released result by {d:.2f}/255"


def test_preview_is_not_more_aggressive_than_the_result(canvas):
    """The reported symptom: the drag looked much stronger than the release."""
    canvas.tool = "brush"
    canvas.current_effect = "light"
    canvas.current_strength = 30
    _drag(canvas, STROKE)
    band = (slice(120, 180), slice(80, 320))

    def detail(a):
        return float(np.abs(np.diff(a[band].astype(float), axis=1)).mean())

    before = detail(canvas._composed_base)
    live = detail(canvas._live_img)
    final = detail(canvas._composed_base)
    assert abs(live - final) < 1.0, \
        f"preview detail {live:.2f} vs released {final:.2f} (started at {before:.2f})"


def test_single_frame_drag_equals_many_frame_drag(canvas):
    """Rendering every frame must not accumulate: a stroke dragged in 40 steps
    has to land on the same pixels as the same stroke committed in one go."""
    canvas.tool = "brush"
    canvas.current_effect = "white_mist"
    canvas.current_strength = 30
    _drag(canvas, STROKE, render_each_frame=True)
    many = canvas._composed_base.copy()

    cv2 = EditorCanvas()
    cv2.image_np = np.array(texture())
    cv2.original_pil = texture()
    cv2.sticker_library = {}
    cv2._rebuild()
    cv2.tool = "brush"
    cv2.current_effect = "white_mist"
    cv2.current_strength = 30
    _drag(cv2, STROKE, render_each_frame=False)      # no intermediate frames
    one = cv2._composed_base
    d = float(np.abs(many.astype(int) - one.astype(int)).mean())
    assert d < 0.01, f"frame count changed the result by {d:.3f}/255"


# ================================================== a: cursor size

@pytest.mark.parametrize("zoom", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_brush_cursor_tracks_the_paint_size(canvas, zoom):
    """Cursor diameter must be brush_size * zoom (clamped to 8..200 px)."""
    canvas.set_brush_size(30)
    canvas.resetTransform()
    canvas.scale(zoom, zoom)
    pm, hotspot = canvas._make_brush_cursor()
    expected = max(8, min(200, int(round(30 * zoom))))
    assert pm.width() == expected, f"zoom {zoom}: cursor {pm.width()}px, want {expected}"
    assert hotspot == expected // 2


def test_brush_cursor_is_rebuilt_on_zoom_change(canvas):
    """zoomChanged must refresh the cursor (it is sized in screen pixels)."""
    canvas.set_tool("brush")
    canvas.set_brush_size(30)
    canvas.resetTransform()
    before = canvas.cursor().pixmap().width()
    canvas.scale(2.0, 2.0)
    canvas.zoomChanged.emit(2.0)          # what _apply_zoom/fit emits
    after = canvas.cursor().pixmap().width()
    assert after > before, f"cursor did not grow with zoom ({before} -> {after})"


def test_brush_cursor_is_not_silently_clamped_for_normal_sizes(canvas):
    """The old clamp 16..64 lied for small and large brushes."""
    for size, want in ((4, 8), (12, 12), (50, 50), (100, 100)):
        canvas.set_brush_size(size)
        canvas.resetTransform()
        pm, _ = canvas._make_brush_cursor()
        assert pm.width() == want, f"brush {size} -> cursor {pm.width()}px"


# ================================================== c: eraser coverage

def test_eraser_erases_exactly_what_the_stroke_covers(canvas):
    """The eraser commits the same geometry it previews, covering 100 % of the
    painted stroke (measured before: 8805/8805)."""
    canvas.auto_mask = np.zeros((H, W), dtype=bool)
    canvas.auto_mask[100:200, 40:360] = True
    canvas.auto_params = {"auto_effect": "blur", "auto_strength": 10,
                          "auto_mask_dilate": 10}
    canvas.erase_mask = np.zeros((H, W), dtype=bool)
    canvas._rebuild()
    canvas.tool = "eraser"
    canvas.set_brush_size(30)
    _drag(canvas, [(x, 150) for x in range(60, 341, 6)])

    stroke = canvas._stroke_mask([(x, 150) for x in range(60, 341, 6)], 30)
    covered = int(stroke.sum())
    erased = int((canvas.erase_mask & stroke).sum())
    assert covered > 0
    assert erased == covered, f"only {erased}/{covered} of the stroke was erased"

    # and the composed pixels really are the original there
    from app.core.image_ops import compose_image
    from app.core.selection import MaskSelection
    comp = np.array(compose_image(
        canvas.original_pil, [], [MaskSelection(canvas.auto_mask, effect="blur",
                                                strength=10)], [], {},
        auto_params=canvas.auto_params, erase_mask=canvas.erase_mask)
        .convert("RGB")).astype(int)
    orig = np.array(canvas.original_pil.convert("RGB")).astype(int)
    residual = float(np.abs(comp - orig).mean(axis=2)[stroke].mean())
    assert residual < 1.0, f"censoring still visible inside the erased stroke ({residual:.2f})"
