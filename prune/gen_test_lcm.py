#!/usr/bin/env python3
"""
Generate comparison images for an LCM base model vs a pruned UNet.

This uses the same native/manual LCM sampling path as gen_lcm_baseline.py.
The repo-local LCM pipeline was written for an older diffusers API, so we
avoid calling pipe(...) directly and run the denoising loop explicitly.
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from prune import sp_core


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_MODEL = Path(os.environ.get("BASE_MODEL", "models/lcm-dreamshaper-v7"))
PRUNED_ST = Path(os.environ.get("PRUNED_ST", "models/distill_step_40000.safetensors"))
PRUNED_CFG = Path(os.environ.get("PRUNED_CFG", "models/distill_step_40000.config.json"))
ROUND = os.environ.get("ROUND", "1")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "gen_test_output/lcm_compare"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STEPS = int(os.environ.get("STEPS", "8"))
GUIDANCE = float(os.environ.get("GUIDANCE", "8.0"))
LCM_ORIGIN_STEPS = int(os.environ.get("LCM_ORIGIN_STEPS", os.environ.get("ORIGINAL_INFERENCE_STEPS", "50")))
HEIGHT = int(os.environ.get("HEIGHT", "768"))
WIDTH = int(os.environ.get("WIDTH", "768"))
DTYPE_NAME = os.environ.get("DTYPE", "float32").lower()

PROMPTS = [
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "close-up portrait photo of a young woman, detailed eyes, natural skin texture, soft studio light, 85mm lens",
    "close-up portrait photo of an elderly man with grey beard, detailed face wrinkles, dramatic side lighting",
    "a fluffy golden retriever puppy sitting in autumn leaves, sharp fur detail, professional photography",
    "a tabby cat sitting on a wooden windowsill, detailed whiskers and eyes, warm morning light",
    "a red fox standing in snowy forest, detailed fur, wildlife photography",
    "a beautiful landscape with mountains and lake, sunrise, highly detailed, cinematic",
    "a cozy coffee shop interior, morning light through windows, detailed furniture, realistic",
]
SEEDS = [42, 123, 2024, 7, 256, 999, 55, 314]


def load_native_lcm_pipeline(base_model: Path, dtype: torch.dtype, unet=None):
    model_source = sp_core.resolve_base_model_source(str(base_model))
    model_dir = Path(model_source)
    sys.path.insert(0, str(model_dir.resolve()))
    from lcm_pipeline import LatentConsistencyModelPipeline

    kwargs = dict(
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    if unet is not None:
        kwargs["unet"] = unet

    pipe = LatentConsistencyModelPipeline.from_pretrained(str(model_dir), **kwargs)
    pipe = pipe.to(device=DEVICE, dtype=dtype)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)
    return pipe


def make_latents(pipe, generator: torch.Generator, dtype: torch.dtype):
    shape = (
        1,
        pipe.unet.config.in_channels,
        HEIGHT // pipe.vae_scale_factor,
        WIDTH // pipe.vae_scale_factor,
    )
    latents = torch.randn(shape, generator=generator, device=DEVICE, dtype=dtype)
    return latents * pipe.scheduler.init_noise_sigma


def generate(pipe, prompt: str, seed: int, dtype: torch.dtype):
    device = torch.device(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        prompt_embeds = pipe._encode_prompt(prompt, device, 1, None)
        latents = make_latents(pipe, generator, prompt_embeds.dtype)
        pipe.scheduler.set_timesteps(
            STEPS,
            device=device,
            original_inference_steps=LCM_ORIGIN_STEPS,
        )
        w = torch.tensor(GUIDANCE).repeat(1)
        w_embedding = pipe.get_w_embedding(
            w,
            embedding_dim=256,
            dtype=prompt_embeds.dtype,
        ).to(device=device, dtype=prompt_embeds.dtype)

        denoised = latents
        for t in pipe.scheduler.timesteps:
            ts = torch.full((1,), t, device=device, dtype=torch.long)
            model_pred = pipe.unet(
                latents,
                ts,
                timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            latents, denoised = pipe.scheduler.step(
                model_pred,
                t,
                latents,
                generator=generator,
                return_dict=False,
            )

        image = pipe.vae.decode(denoised / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = pipe.run_safety_checker(image, device, prompt_embeds.dtype)
        do_denormalize = [True] if has_nsfw_concept is None else [not has_nsfw_concept[0]]
        return pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)[0]


def save_contact_sheet(pairs, path: Path):
    if not pairs:
        return
    thumb_w = 256
    label_h = 28
    row_gap = 8
    thumb_h = int(pairs[0][0].size[1] * thumb_w / pairs[0][0].size[0])
    canvas = Image.new("RGB", (2 * thumb_w, len(pairs) * (thumb_h + label_h + row_gap)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for idx, (base_img, pruned_img) in enumerate(pairs):
        y = idx * (thumb_h + label_h + row_gap)
        draw.text((5, y + 6), f"{idx + 1:02d} baseline", fill=(0, 0, 0), font=font)
        draw.text((thumb_w + 5, y + 6), f"{idx + 1:02d} pruned {ROUND}", fill=(0, 0, 0), font=font)
        canvas.paste(base_img.resize((thumb_w, thumb_h)), (0, y + label_h))
        canvas.paste(pruned_img.resize((thumb_w, thumb_h)), (thumb_w, y + label_h))
    canvas.save(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if DTYPE_NAME == "float16" and DEVICE == "cuda" else torch.float32
    print(
        "LCM compare settings: "
        f"base={BASE_MODEL}, steps={STEPS}, guidance={GUIDANCE}, "
        f"origin_steps={LCM_ORIGIN_STEPS}, size={WIDTH}x{HEIGHT}, dtype={dtype}"
    )

    print("Loading native baseline LCM pipeline...")
    pipe_base = load_native_lcm_pipeline(BASE_MODEL, dtype)
    baseline_imgs = []
    for i, (prompt, seed) in enumerate(zip(PROMPTS, SEEDS), start=1):
        print(f"baseline [{i}/{len(PROMPTS)}] seed={seed}: {prompt}")
        img = generate(pipe_base, prompt, seed, dtype)
        img.save(OUTPUT_DIR / f"baseline_{i}.png")
        baseline_imgs.append(img)

    del pipe_base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("Loading pruned UNet from safetensors...")
    from prune.pruned_rebuild import create_unet_from_safetensors

    pruned_unet = create_unet_from_safetensors(str(PRUNED_ST), str(PRUNED_CFG))
    params_m = sum(p.numel() for p in pruned_unet.parameters()) / 1e6
    print(f"Pruned UNet: {params_m:.1f}M params")
    pruned_unet = pruned_unet.to(device=DEVICE, dtype=dtype)

    print("Loading native LCM pipeline with pruned UNet...")
    pipe_pruned = load_native_lcm_pipeline(BASE_MODEL, dtype, unet=pruned_unet)
    pairs = []
    for i, (prompt, seed, base_img) in enumerate(zip(PROMPTS, SEEDS, baseline_imgs), start=1):
        print(f"pruned [{i}/{len(PROMPTS)}] seed={seed}: {prompt}")
        img = generate(pipe_pruned, prompt, seed, dtype)
        img.save(OUTPUT_DIR / f"pruned_round{ROUND}_{i}.png")
        pairs.append((base_img, img))

    save_contact_sheet(pairs, OUTPUT_DIR / "contact_sheet.png")
    print(f"DONE: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
