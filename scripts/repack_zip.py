# -*- coding: utf-8 -*-
"""Zip dist/UnifiedImageTool into UnifiedImageTool.zip with STANDARD
forward-slash entries.

Why not PowerShell's Compress-Archive: it writes backslash-separated entry
names (non-standard). Windows Explorer copes, but third-party unpackers
(7-Zip / 360zip / some WinRAR versions) flatten the whole tree into a pile
of files whose names contain backslashes — the user "can't find the exe"
(2026-10-24 report). zipfile always writes forward slashes; the top-level
UnifiedImageTool/ folder is preserved so extraction stays tidy.

Usage: python repack_zip.py <dist_dir> <zip_path>
"""
import sys, time, zipfile
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
base = src.parent  # arcnames relative to dist/ keep the UnifiedImageTool/ root
files = sorted(p for p in src.rglob("*") if p.is_file())
t0 = time.time()
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for i, p in enumerate(files, 1):
        z.write(p, p.relative_to(base).as_posix())
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}")
print(f"packed {len(files)} files in {time.time() - t0:.1f}s")
