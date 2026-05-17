#!/usr/bin/env python3
"""Generate SDXL-Lightning smoke-test images with the official Diffusers setup."""

import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

BASE_MODEL = os.environ.get("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
LIGHTNING_REPO = os.environ.get("LIGHTNING_REPO", "ByteDance/SDXL-Lightning")
STEP = int(os.environ.get("STEP", "4"))
HEIGHT = int(os.environ.get("HEIGHT", "1024"))
WIDTH = int(os.environ.get("WIDTH", "1024"))
GUIDANCE = float(os.environ.get("GUIDANCE", "0.0"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", f"gen_test_output/sdxl_lightning_{STEP}step"))
DTYPE_NAME = os.environ.get("DTYPE", "float16").lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "close-up portrait photo of a young woman, detailed eyes, natural skin texture, soft studio light, 85mm lens",
    "a fluffy golden retriever puppy sitting in autumn leaves, sharp fur detail, professional photography",
    "a cyberpunk city street at night, neon signs, rain reflections, cinematic, highly detailed",
    "a beautiful mountain lake at sunrise, mist, dramatic clouds, ultra detailed landscape photography",
]
SEEDS = [123, 7, 2024, 42]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if DTYPE_NAME == "float16" and DEVICE == "cuda" else torch.float32
    ckpt = f"sdxl_lightning_{STEP}step_unet.safetensors"
    print(
        f"SDXL-Lightning smoke: base={BASE_MODEL}, repo={LIGHTNING_REPO}, ckpt={ckpt}, "
        f"steps={STEP}, guidance={GUIDANCE}, size={WIDTH}x{HEIGHT}, dtype={dtype}, device={DEVICE}"
    )

    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(DEVICE, dtype)
    ckpt_path = hf_hub_download(LIGHTNING_REPO, ckpt)
    state = load_file(ckpt_path, device=DEVICE)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    print(f"Loaded Lightning UNet: missing={len(missing)}, unexpected={len(unexpected)}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        unet=unet,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(DEVICE)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    for i, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS), start=1):
        print(f"[{i}/{len(PROMPTS)}] seed={seed}: {prompt}")
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=STEP,
                guidance_scale=GUIDANCE,
                height=HEIGHT,
                width=WIDTH,
                generator=generator,
            ).images[0]
        out = OUTPUT_DIR / f"sdxl_lightning_{STEP}step_{i:02d}_seed{seed}.png"
        image.save(out)
        print(f"  saved {out}")

    print(f"DONE: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
