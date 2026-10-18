# -*- mode: python ; coding: utf-8 -*-

# Unused modules excluded to shrink the bundle. QtQuick/Qml/Pdf/Network/etc.
# are only pulled in by the full PySide6 QtGui/QtWidgets import graph and are
# never used by this widgets-only app. requests' optional deps (brotli,
# zstandard) and PyYAML/cryptography are likewise unused; if a smoke test of
# the packaged exe fails, remove the offending entry from this list.
pyinstaller_excludes = [
    # Qt modules this widgets-only app does not use
    'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtQmlModels',
    'PySide6.QtPdf', 'PySide6.QtPdfWidgets', 'PySide6.QtNetwork',
    'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
    'PySide6.QtCharts', 'PySide6.QtDataVisualization',
    'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DInput',
    'PySide6.Qt3DLogic', 'PySide6.Qt3DAnimation', 'PySide6.Qt3DExtras',
    'PySide6.QtDesigner', 'PySide6.QtUiTools', 'PySide6.QtHelp',
    'PySide6.QtSql', 'PySide6.QtTest', 'PySide6.QtXml',
    'PySide6.QtOpenGL', 'PySide6.QtOpenGLWidgets',
    # optional/transitive deps not imported by the app
    'brotli', 'brotlicffi', 'zstandard', 'cryptography', 'yaml', 'PyYAML',
]

a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=[],
    datas=[('app/assets', 'app/assets'), ('app/stickers', 'app/stickers'), ('models', 'models'),
          ('app/ui/xdd_arrow_light.png', 'app/ui'), ('app/ui/xdd_arrow_dark.png', 'app/ui'),
              ('app/ui/xdd_check_light.png', 'app/ui'), ('app/ui/xdd_check_dark.png', 'app/ui')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=pyinstaller_excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UnifiedImageTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPECPATH, 'app', 'assets', 'icon.ico'),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='UnifiedImageTool',
)

# ---- post-build cleanup: drop DLLs the app never loads ----
import glob  # noqa: E402
import os  # noqa: E402

_dist_dir = os.path.join(SPECPATH, 'dist', coll.name)
_post_remove_patterns = [
    # OpenCV bundles a video/ffmpeg backend (~30MB) inside the cv2 package
    # dir; this tool only processes still images.
    os.path.join('_internal', 'cv2', 'opencv_videoio_ffmpeg*.dll'),
    # Qt software OpenGL fallback (~20MB); the app uses raster 2D views.
    # PySide6 6.11 puts it under _internal/PySide6/, older layouts at
    # _internal/ directly -- cover both.
    os.path.join('_internal', 'opengl32sw.dll'),
    os.path.join('_internal', 'PySide6', 'opengl32sw.dll'),
    # Qt translations (.qm): the app ships hardcoded Chinese strings and
    # never installs a QTranslator, so all ~100 locales are dead weight.
    os.path.join('_internal', 'PySide6', 'translations', '*.qm'),
    # Qt DLLs for modules excluded above, in case a PySide6 hook still
    # copied them from the installed Qt distribution.
    os.path.join('_internal', 'PySide6', 'Qt6Quick*.dll'),
    os.path.join('_internal', 'PySide6', 'Qt6Qml*.dll'),
    os.path.join('_internal', 'PySide6', 'Qt6Pdf*.dll'),
    os.path.join('_internal', 'PySide6', 'Qt6Network*.dll'),
    os.path.join('_internal', 'PySide6', 'Qt6Multimedia*.dll'),
    os.path.join('_internal', 'PySide6', 'Qt6OpenGL*.dll'),
]
for pattern in _post_remove_patterns:
    for f in glob.glob(os.path.join(_dist_dir, pattern)):
        try:
            os.remove(f)
            print(f"[cleanup] removed {os.path.basename(f)}")
        except OSError as exc:
            print(f"[cleanup] skip {f}: {exc}")