#!/usr/bin/env python3
"""Build a 3-column contact sheet: FP16 baseline | Packed only | Packed+LoRA.

Rows are selected prompts (worst-case ones where LoRA helped, plus a few easy
ones to check no regression). Each row is labeled with prompt index/category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_one(d: Path, pattern: str) -> Path | None:
    matches = list(d.glob(pattern))
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Has fp16_*.png + selected_prompts.json")
    ap.add_argument("--packed-dir", type=Path, required=True,
                    help="Has packed_only_*.png")
    ap.add_argument("--lora-dir", type=Path, required=True,
                    help="Has packed_lora_*.png")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--indices", type=int, nargs="+",
                    default=[19, 22, 55, 36, 56, 3, 6, 30, 2, 11, 17, 18],
                    help="Prompt indices to include (rows).")
    ap.add_argument("--thumb-width", type=int, default=384)
    args = ap.parse_args()

    sp = json.loads((args.baseline_dir / "selected_prompts.json").read_text(encoding="utf-8"))
    by_idx = {it["index"]: it for it in sp}

    rows = []
    for idx in args.indices:
        if idx not in by_idx:
            print(f"skip #{idx:02d} (not in baseline)")
            continue
        it = by_idx[idx]
        seed = it["seed"]
        fp16 = find_one(args.baseline_dir, f"fp16_{idx:02d}_seed{seed}_*.png")
        pak = find_one(args.packed_dir, f"packed_only_{idx:02d}_seed{seed}_*.png")
        lora = find_one(args.lora_dir, f"packed_lora_r16_{idx:02d}_seed{seed}_*.png")
        if not (fp16 and pak and lora):
            print(f"skip #{idx:02d} (missing: fp16={fp16 is None} pak={pak is None} lora={lora is None})")
            continue
        rows.append((it, fp16, pak, lora))
    print(f"{len(rows)} rows to render")

    # Determine cell size from first image
    sample = Image.open(rows[0][1])
    aspect = sample.height / sample.width
    cell_w = args.thumb_width
    cell_h = int(cell_w * aspect)

    label_h = 36       # per-row label (prompt info)
    header_h = 28      # column headers
    pad = 6

    n_cols = 3
    sheet_w = n_cols * cell_w + (n_cols + 1) * pad
    sheet_h = header_h + sum(label_h + cell_h + pad for _ in rows) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        bold = ImageFont.truetype("arialbd.ttf", 14)
    except OSError:
        font = bold = ImageFont.load_default()

    headers = ["FP16 baseline", "Packed only (no LoRA)", "Packed + LoRA r=16 α=8"]
    for c, h in enumerate(headers):
        x = pad + c * (cell_w + pad)
        draw.text((x + 6, 6), h, fill=(0, 0, 0), font=bold)

    y = header_h + pad
    for it, fp16, pak, lora in rows:
        label = f"#{it['index']:02d}  [{it.get('category','?')}]  seed={it['seed']}  | {it['prompt'][:120]}"
        draw.text((pad + 6, y - 2), label, fill=(40, 40, 40), font=font)
        for c, path in enumerate([fp16, pak, lora]):
            img = Image.open(path).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
            x = pad + c * (cell_w + pad)
            sheet.paste(img, (x, y + label_h - 12))
        y += label_h + cell_h + pad

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)
    print(f"Saved: {args.output}  ({sheet.size})")


if __name__ == "__main__":
    main()
