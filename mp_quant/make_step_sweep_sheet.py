#!/usr/bin/env python3
"""6-column contact sheet for step-count sweep.

Columns: FP16@4 | LoRA@4 | FP16@6 | LoRA@6 | FP16@8 | LoRA@8
Rows: one per prompt.
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
    ap.add_argument("--fp16-step4-dir", type=Path, required=True)
    ap.add_argument("--lora-step4-dir", type=Path, required=True)
    ap.add_argument("--fp16-step6-dir", type=Path, required=True)
    ap.add_argument("--lora-step6-dir", type=Path, required=True)
    ap.add_argument("--fp16-step8-dir", type=Path, required=True)
    ap.add_argument("--lora-step8-dir", type=Path, required=True)
    ap.add_argument("--prompts-from", type=Path, required=True,
                    help="Dir with selected_prompts.json defining which prompts to include")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prompt-count", type=int, default=12)
    ap.add_argument("--thumb-width", type=int, default=256)
    args = ap.parse_args()

    items = json.loads((args.prompts_from / "selected_prompts.json").read_text(encoding="utf-8"))[: args.prompt_count]

    cols = [
        ("FP16 @ 4 steps",  args.fp16_step4_dir,  "fp16_{idx:02d}_seed{seed}*.png"),
        ("LoRA @ 4 steps",  args.lora_step4_dir,  "packed_lora_r16_{idx:02d}_seed{seed}*.png"),
        ("FP16 @ 6 steps",  args.fp16_step6_dir,  "fp16_{idx:02d}_seed{seed}*.png"),
        ("LoRA @ 6 steps",  args.lora_step6_dir,  "packed_lora_r16_{idx:02d}_seed{seed}*.png"),
        ("FP16 @ 8 steps",  args.fp16_step8_dir,  "fp16_{idx:02d}_seed{seed}*.png"),
        ("LoRA @ 8 steps",  args.lora_step8_dir,  "packed_lora_r16_{idx:02d}_seed{seed}*.png"),
    ]

    rows = []
    for it in items:
        idx, seed = it["index"], it["seed"]
        paths = []
        for (label, d, pat) in cols:
            p = find_one(d, pat.format(idx=idx, seed=seed))
            paths.append(p)
        if any(p is None for p in paths):
            missing_labels = [cols[i][0] for i, p in enumerate(paths) if p is None]
            print(f"skip #{idx:02d} — missing in {missing_labels}")
            continue
        rows.append((it, paths))
    print(f"{len(rows)} rows to render")

    sample = Image.open(rows[0][1][0])
    aspect = sample.height / sample.width
    cell_w = args.thumb_width
    cell_h = int(cell_w * aspect)

    label_h = 36
    header_h = 32
    pad = 6
    n_cols = len(cols)
    sheet_w = n_cols * cell_w + (n_cols + 1) * pad
    sheet_h = header_h + sum(label_h + cell_h + pad for _ in rows) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        bold = ImageFont.truetype("arialbd.ttf", 13)
    except OSError:
        font = bold = ImageFont.load_default()

    for c, (label, _, _) in enumerate(cols):
        x = pad + c * (cell_w + pad)
        draw.text((x + 6, 8), label, fill=(0, 0, 0), font=bold)

    y = header_h + pad
    for it, paths in rows:
        label = f"#{it['index']:02d} [{it.get('category','?')}] seed={it['seed']}  |  {it['prompt'][:140]}"
        draw.text((pad + 6, y - 2), label, fill=(40, 40, 40), font=font)
        for c, p in enumerate(paths):
            img = Image.open(p).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
            x = pad + c * (cell_w + pad)
            sheet.paste(img, (x, y + label_h - 12))
        y += label_h + cell_h + pad

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)
    print(f"Saved: {args.output}  ({sheet.size})")


if __name__ == "__main__":
    main()
