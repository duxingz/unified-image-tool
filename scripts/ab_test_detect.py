# -*- coding: utf-8 -*-
"""A/B quality check: same real images detected at 1280 / 1024 / 832.

Measures per-size: detection counts, class mix, masked pixel ratio and
timing. Exports censored previews to a separate output folder with rename
naming (S1280_xxx / S1024_xxx / S832_xxx) so the originals are never
touched or overwritten.

Usage: python scripts/ab_test_detect.py [图片目录] [输出目录]
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from app.core.config import default_config
from app.core.detector import ONNXDetector, pick_providers
from app.core.image_ops import load_resized

SIZES = [1280, 1024, 832]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def main() -> int:
    src_dir = Path(sys.argv[1] if len(sys.argv) > 1 else
                   r"C:\Users\qu\Desktop\nai26\杂图合集")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else
                   src_dir / "_ab_test_output")

    paths = sorted(
        p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()
    )
    if not paths:
        print(f"目录无图片: {src_dir}")
        return 1
    print(f"样图 {len(paths)} 张，来自 {src_dir}")

    detector = ONNXDetector(default_config()["model_path"],
                            providers=pick_providers(True))
    print(f"provider: {detector.session.get_providers()[0]}")

    stats = {s: {"dets": 0, "mask_px": 0, "total_px": 0, "sec": 0.0,
                 "imgs_with": 0, "class_cnt": {}} for s in SIZES}
    per_size_names = {s: [] for s in SIZES}
    summary_rows = []

    for idx, path in enumerate(paths):
        try:
            orig_pil, _, _ = load_resized(str(path))
        except Exception as e:
            print(f"  跳过 {path.name}: {e}")
            continue
        rgb = np.array(orig_pil.convert("RGB"))
        bgr = rgb[:, :, ::-1]
        h, w = rgb.shape[:2]

        row = {"file": path.name, "size": f"{w}x{h}"}
        consistent = True
        for size in SIZES:
            t0 = time.perf_counter()
            dets = detector.detect(bgr, conf_threshold=0.35, iou_threshold=0.5,
                                   allowed_classes=["anus", "nipple", "penis",
                                                    "vagina", "pubic_hair"],
                                   input_size=size)
            dt = time.perf_counter() - t0
            stats[size]["sec"] += dt
            stats[size]["dets"] += len(dets)
            stats[size]["total_px"] += h * w
            if dets:
                stats[size]["imgs_with"] += 1
            union = np.zeros((h, w), dtype=bool)
            for d in dets:
                union |= d.mask
                stats[size]["class_cnt"][d.class_name] = (
                    stats[size]["class_cnt"].get(d.class_name, 0) + 1
                )
            stats[size]["mask_px"] += int(union.sum())
            row[f"d{size}"] = len(dets)
            per_size_names[size].append(f"S{size}_{idx + 1:03d}.png")
        # cross-size agreement for this image
        pattern = tuple(row.get(f"d{s}", 0) for s in SIZES)
        if len(set(pattern)) > 1:
            consistent = False
        row["一致"] = "一致" if consistent else "不一致"
        summary_rows.append(row)

    # ---- export censored previews per size (rename naming) ----------------
    for size in SIZES:
        sub = out_dir / f"u{size}"
        sub.mkdir(parents=True, exist_ok=True)
        for idx, path in enumerate(paths):
            try:
                orig_pil, _, _ = load_resized(str(path))
            except Exception:
                continue
            rgb = np.array(orig_pil.convert("RGB"))
            dets = detector.detect(rgb[:, :, ::-1], conf_threshold=0.35,
                                   iou_threshold=0.5, input_size=size)
            union = np.zeros(rgb.shape[:2], dtype=bool)
            for d in dets:
                union |= d.mask
            from app.core.selection import MaskSelection
            from app.core.image_ops import compose_image
            auto_params = {"auto_effect": "blur", "auto_strength": 10,
                           **default_config().get("params", {})}
            composed = compose_image(
                orig_pil, [], [MaskSelection(union, effect="blur", strength=10)],
                [], {}, params=default_config().get("params", {}),
                auto_params=auto_params,
            )
            name = f"S{size}_{idx + 1:03d}.png"
            composed.save(sub / name)
            composed.close()
        print(f"预览导出: {sub} ({len(list(sub.glob('*.png')))} 张)")

    # ---- report -----------------------------------------------------------
    print()
    print(f"{'档位':>6} {'检出总数':>8} {'平均检出':>8} {'有检出图':>8} {'掩码像素占比':>10} {'耗时':>8}")
    for size in SIZES:
        st = stats[size]
        n_img = len(paths)
        avg = st["dets"] / n_img
        ratio = st["mask_px"] / st["total_px"] * 100 if st["total_px"] else 0
        print(f"{size:>6} {st['dets']:>8} {avg:>7.2f}个 {st['imgs_with']:>6}/{n_img} "
              f"{ratio:>9.2f}% {st['sec']:>6.1f}s")
    print()
    print("类别构成（1280/1024/832）:")
    all_classes = sorted({c for s in SIZES for c in stats[s]["class_cnt"]})
    for c in all_classes:
        print(f"  {c:>12}: " + " / ".join(str(stats[s]["class_cnt"].get(c, 0)) for s in SIZES))
    diff = sum(1 for r in summary_rows if r["一致"] == "不一致")
    print()
    print(f"跨档位检出数量不一致的图片: {diff}/{len(summary_rows)}")
    print(f"（不一致的图片名见下，用于人工判断小档位漏检是否不可接受）")
    for r in summary_rows:
        if r["一致"] == "不一致":
            print(f"  {r['file']}: 1280={r['d1280']} 1024={r['d1024']} 832={r['d832']}")
    print()
    print(f"导出目录: {out_dir}（重命名输出，原始文件未动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())