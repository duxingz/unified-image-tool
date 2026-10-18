# -*- coding: utf-8 -*-
"""Clipboard helpers (Windows image copy)."""
from __future__ import annotations

import io
import platform

from PIL import Image


def copy_image_to_clipboard(pil_image: Image.Image) -> bool:
    """Copy a PIL image to the Windows clipboard as DIB."""
    if pil_image is None:
        return False
    try:
        if platform.system() == "Windows":
            import win32clipboard

            output = io.BytesIO()
            pil_image.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]  # strip BMP header
            output.close()

            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            finally:
                win32clipboard.CloseClipboard()
            return True
    except ImportError:
        print("Windows剪贴板操作需要 pywin32。请运行: pip install pywin32")
    except Exception as e:
        print(f"复制图片到剪贴板时发生错误: {e}")
    return False
