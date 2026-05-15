#!/usr/bin/env python3
"""
Generate comparison images: baseline SD1.5 vs pruned UNet.

Produces side-by-side grids (baseline | pruned) using identical seeds
so differences are clearly visible.
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
from prune import sp_core

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = 'models'

BASE_MODEL = os.environ.get('BASE_MODEL', 'models/sd-turbo')
PRUNED_ST = os.environ.get('PRUNED_ST', os.path.join(MODELS_DIR, 'distill_step_40000.safetensors'))
PRUNED_CFG = os.environ.get('PRUNED_CFG', os.path.join(MODELS_DIR, 'distill_step_40000.config.json'))
ROUND = os.environ.get('ROUND', '1')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'gen_test_output')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
# SD-Turbo is an adversarial distillation model: designed for 1-4 steps with
# guidance_scale=0.0. Running it with 30 steps / CFG=7.5 mismatches the training
# schedule and produces pixelated results despite correct structure.
STEPS = int(os.environ.get('STEPS', 2))
GUIDANCE = 0.0


def make_side_by_side(img_left: Image.Image, img_right: Image.Image,
                      label_left: str, label_right: str) -> Image.Image:
    """Combine two images side by side with labels."""
    w, h = img_left.size
    label_h = 32
    canvas = Image.new('RGB', (w * 2, h + label_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw labels centered
    for i, label in enumerate([label_left, label_right]):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        x = i * w + (w - tw) // 2
        draw.text((x, 6), label, fill=(0, 0, 0), font=font)

    canvas.paste(img_left, (0, label_h))
    canvas.paste(img_right, (w, label_h))
    return canvas


def generate_with_pipe(pipe, prompt, seed):
    """Generate a single image with fixed seed. SD-Turbo: 4 steps, no CFG."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=512, width=512,
            generator=gen,
        )
    return result.images[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_model_source = sp_core.resolve_base_model_source(BASE_MODEL)

    # --- Baseline pipeline ---
    print("Loading baseline SD-Turbo pipeline...")
    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
    pipe_base = StableDiffusionPipeline.from_pretrained(
        base_model_source,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    pipe_base.enable_attention_slicing()

    # Generate baseline images
    print("Generating baseline images...")
    baseline_imgs = []
    for i, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
        print(f"  [{i+1}/{len(PROMPTS)}] seed={seed}: {prompt[:50]}...")
        img = generate_with_pipe(pipe_base, prompt, seed)
        img.save(os.path.join(OUTPUT_DIR, f'baseline_{i+1}.png'))
        baseline_imgs.append(img)

    # Free baseline pipeline
    del pipe_base
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # --- Pruned pipeline ---
    print("\nLoading pruned UNet from safetensors...")
    from prune.pruned_rebuild import create_unet_from_safetensors
    pruned_unet = create_unet_from_safetensors(PRUNED_ST, PRUNED_CFG)
    params_m = sum(p.numel() for p in pruned_unet.parameters()) / 1e6
    print(f"  Pruned UNet: {params_m:.1f}M params")

    pruned_unet = pruned_unet.to(dtype=dtype, device=DEVICE)

    pipe_pruned = StableDiffusionPipeline.from_pretrained(
        base_model_source,
        unet=pruned_unet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    ).to(DEVICE)
    pipe_pruned.enable_attention_slicing()

    # Generate pruned images — 8 separate files named pruned_round#_N.png
    print(f"Generating pruned model images (round {ROUND})...")
    for i, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS)):
        print(f"  [{i+1}/{len(PROMPTS)}] seed={seed}: {prompt[:50]}...")
        img = generate_with_pipe(pipe_pruned, prompt, seed)
        fname = f'pruned_round{ROUND}_{i+1}.png'
        img.save(os.path.join(OUTPUT_DIR, fname))
        print(f"  Saved: {fname}")

    print(f"\nAll outputs in: {OUTPUT_DIR}")
    print("DONE")


if __name__ == '__main__':
    main()
