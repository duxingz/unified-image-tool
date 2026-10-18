# -*- coding: utf-8 -*-
"""Filename numbering logic."""
import os
import re
import time
from pathlib import Path

_NUM_SPLIT = re.compile(r"(\d+)")


def natural_key(name: str):
    """Windows-like natural sort key: img_2 < img_10, and 037 < 037-1.

    The file extension is excluded from the comparison - with the old key the
    '.' character sorted after '-' ('.' > '-' in ASCII), making '037.png'
    order AFTER '037-1.png' while Windows places it first."""
    text = str(name)
    stem, dot, ext = text.rpartition(".")
    if not dot:
        stem, ext = text, ""
    parts = _NUM_SPLIT.split(stem)
    key = [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in parts
    ]
    # extension acts as a tie-breaker only for identical stems
    key.append((1, "." + ext.lower()) if ext else (1, ""))
    return key


def build_filename(prefix, start, digits, index, ext):
    num = start + index
    num_str = str(num).zfill(digits) if digits > 0 else str(num)
    return f"{prefix}{num_str}.{ext}"


def build_preview(prefix, start, digits, ext="png"):
    try:
        start = int(start)
    except (TypeError, ValueError):
        return "起始编号无效"
    num_str = str(start).zfill(digits) if digits > 0 else str(start)
    return f"{prefix}{num_str}.{ext}"

# ----------------------------------------------------------------------
# Rename templates: {原名} {日期} {时间} {序号}
# ----------------------------------------------------------------------
INVALID_CHARS = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(name: str) -> str:
    """Replace Windows-illegal filename characters with '_'."""
    return INVALID_CHARS.sub("_", str(name))


def shot_date(path: str) -> str:
    """Best date for {日期}/{时间}: EXIF DateTimeOriginal when present,
    else the file modification time. Returns 'YYYYmmdd' / 'YYYYmmdd-HHMMSS'
    inside the returned dict."""
    date_s = time_s = ""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get(36867) or exif.get(306)  # DateTimeOriginal / DateTime
        if raw:
            # EXIF format 'YYYY:MM:DD HH:MM:SS' - ':' is illegal in filenames
            digits = re.sub(r"\D", "", str(raw))
            if len(digits) >= 14:
                date_s, time_s = digits[:8], f"{digits[:8]}-{digits[8:14]}"
            elif len(digits) >= 8:
                date_s = digits[:8]
    except Exception:
        pass
    if not date_s:
        try:
            mt = os.path.getmtime(path)
        except OSError:
            mt = time.time()
        date_s = time.strftime("%Y%m%d", time.localtime(mt))
        time_s = time.strftime("%Y%m%d-%H%M%S", time.localtime(mt))
    if not time_s:
        time_s = date_s
    return {"date": date_s, "time": time_s}


def expand_rename_template(template, src_path, index, start, digits,
                           date_info=None) -> str:
    """Expand a rename template. Supported variables:

    {原名} original file stem   {序号} zero-padded running number
    {日期} shoot/mtime date     {时间} date + time (filename-safe)

    A template without any variable keeps the legacy behavior: the text is
    a plain prefix followed by the number. The result is sanitized for
    Windows filenames."""
    template = str(template or "")
    if date_info is None:
        date_info = shot_date(src_path)
    stem = Path(src_path).stem

    def num() -> str:
        n = start + index
        return str(n).zfill(digits) if digits > 0 else str(n)

    if "{" not in template:
        return f"{template}{num()}"
    out = template
    out = out.replace("{原名}", stem)
    out = out.replace("{日期}", date_info["date"])
    out = out.replace("{时间}", date_info["time"])
    out = out.replace("{序号}", num())
    return sanitize_filename(out) or f"IMG{num()}"


def build_rename_plan(paths, template, start, digits):
    """[(original_name, new_name)] preview rows for the rename dialog."""
    rows = []
    for i, p in enumerate(paths):
        ext = Path(p).suffix.lstrip(".").lower() or "png"
        new = expand_rename_template(template, p, i, start, digits)
        rows.append((Path(p).name, f"{new}.{ext}"))
    return rows


def has_template_vars(template) -> bool:
    return "{" in str(template or "")
