#!/usr/bin/env python3
"""
Generate 5-model side-by-side comparison images.

Models compared (left to right):
  1. SD-1.5          (20-step, guidance=7.5)
  2. SD-Turbo 4-step (guidance=0.0)
  3. Pruned  4-step  (guidance=0.0)
  4. SD-Turbo 2-step (guidance=0.0)
  5. Pruned  2-step  (guidance=0.0)

Output: one PNG per prompt, with labelled columns.
"""

import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prune import sp_core
from prune.pruned_rebuild import create_unet_from_safetensors

# ── Config ────────────────────────────────────────────────────────────────────
SD15_MODEL    = os.environ.get('SD15_MODEL',    'models/sd-1-5')
BASE_MODEL    = os.environ.get('BASE_MODEL',    'models/sd-turbo')
PRUNED_ST     = os.environ.get('PRUNED_ST',     'models/distill_feat_attn/distill_final.safetensors')
PRUNED_CFG    = os.environ.get('PRUNED_CFG',    'models/distill_feat_attn/distill_final.config.json')
OUTPUT_DIR    = os.environ.get('OUTPUT_DIR',    'gen_test_output/all_models_compare')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

PROMPTS = [
    "a beautiful landscape with mountains and lake, highly detailed, 4k",
    "a cute cat sitting on a windowsill, digital art, warm lighting",
    "a cyberpunk city skyline at night, neon lights, futuristic",
    "a portrait of a woman with flowers in her hair, soft lighting, artstation",
    "a red sports car parked on a rainy street, reflections, cinematic",
    "an astronaut riding a horse on the moon, surreal, high quality",
    "a cozy coffee shop interior, morning light through windows, watercolor style",
    "a golden retriever playing in autumn leaves, photography, bokeh",
]
SEEDS = [42, 123, 2024, 7, 256, 999, 55, 314]

LABEL_H = 40  # pixels reserved for column header


# ── Inference helper ──────────────────────────────────────────────────────────

def gen_image(pipe, prompt, seed, steps, guidance):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        return pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512, width=512,
            generator=gen,
        ).images[0]


# ── Grid builder ──────────────────────────────────────────────────────────────

def make_grid(images, labels, prompt_text):
    """Stack images side by side with column labels and a bottom caption."""
    assert len(images) == len(labels)
    n   = len(images)
    w, h = images[0].size
    cap_h = 30

    canvas = Image.new('RGB', (w * n, h + LABEL_H + cap_h), (240, 240, 240))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_cap = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      13)
    except (OSError, IOError):
        font_lbl = font_cap = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * w, LABEL_H))
        bbox = draw.textbbox((0, 0), label, font=font_lbl)
        tw   = bbox[2] - bbox[0]
        x    = i * w + (w - tw) // 2
        draw.text((x, 8), label, fill=(30, 30, 30), font=font_lbl)

    # bottom caption with prompt
    short = prompt_text if len(prompt_text) <= 90 else prompt_text[:87] + "..."
    draw.text((8, h + LABEL_H + 6), short, fill=(60, 60, 60), font=font_cap)
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32

    base_model_source = sp_core.resolve_base_model_source(BASE_MODEL)

    configs = [
        # (label, steps, guidance, pipeline_key)
        ("SD-1.5\n20-step",    20, 7.5,  "sd15"),
        ("SD-Turbo\n4-step",    4, 0.0,  "turbo"),
        ("Pruned\n4-step",      4, 0.0,  "pruned"),
        ("SD-Turbo\n2-step",    2, 0.0,  "turbo"),
        ("Pruned\n2-step",      2, 0.0,  "pruned"),
    ]

    # ── Load SD-1.5 ───────────────────────────────────────────────────────────
    print("Loading SD-1.5...")
    pipe_sd15 = StableDiffusionPipeline.from_pretrained(
        SD15_MODEL, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe_sd15.set_progress_bar_config(disable=True)
    p15 = sum(p.numel() for p in pipe_sd15.unet.parameters()) / 1e6
    print(f"  SD-1.5 UNet: {p15:.0f}M params")

    # ── Load SD-Turbo ─────────────────────────────────────────────────────────
    print("Loading SD-Turbo...")
    pipe_turbo = StableDiffusionPipeline.from_pretrained(
        base_model_source, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe_turbo.set_progress_bar_config(disable=True)
    pt = sum(p.numel() for p in pipe_turbo.unet.parameters()) / 1e6
    print(f"  SD-Turbo UNet: {pt:.0f}M params")

    # ── Load Pruned ───────────────────────────────────────────────────────────
    print("Loading pruned UNet...")
    pruned_unet = create_unet_from_safetensors(PRUNED_ST, PRUNED_CFG)
    pp = sum(p.numel() for p in pruned_unet.parameters()) / 1e6
    print(f"  Pruned UNet: {pp:.0f}M params")
    pruned_unet = pruned_unet.to(device=DEVICE, dtype=dtype).eval()

    pipe_pruned = StableDiffusionPipeline.from_pretrained(
        base_model_source, unet=pruned_unet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False, local_files_only=True,
    ).to(DEVICE)
    pipe_pruned.set_progress_bar_config(disable=True)

    pipe_map = {"sd15": pipe_sd15, "turbo": pipe_turbo, "pruned": pipe_pruned}

    # ── Generate ──────────────────────────────────────────────────────────────
    for idx, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
        print(f"\n[{idx+1}/{len(PROMPTS)}] seed={seed}: {prompt[:60]}...")
        images = []
        labels = []
        for label, steps, guidance, pipe_key in configs:
            pipe = pipe_map[pipe_key]
            img  = gen_image(pipe, prompt, seed, steps, guidance)
            images.append(img)
            labels.append(label)
            print(f"  ✓ {label.replace(chr(10), ' ')}")

        grid = make_grid(images, labels, prompt)
        fname = f"compare_{idx+1:02d}.png"
        grid.save(os.path.join(OUTPUT_DIR, fname))
        print(f"  Saved: {fname}")

    print(f"\nAll grids saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
