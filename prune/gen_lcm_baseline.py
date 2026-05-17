#!/usr/bin/env python3
"""
Generate baseline-only images for LCM_Dreamshaper_v7 using its repo-local
pipeline implementation.

This deliberately avoids the pruning/rebuild path and the generic gen_test
script so LCM image quality can be checked in isolation.
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_MODEL = Path(os.environ.get("BASE_MODEL", "models/lcm-dreamshaper-v7"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "gen_test_output/lcm_baseline_native"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STEPS_LIST = [
    int(x.strip())
    for x in os.environ.get("STEPS_LIST", os.environ.get("STEPS", "4,8")).split(",")
    if x.strip()
]
GUIDANCE = float(os.environ.get("GUIDANCE", "8.0"))
LCM_ORIGIN_STEPS = int(os.environ.get("LCM_ORIGIN_STEPS", "50"))
HEIGHT = int(os.environ.get("HEIGHT", "768"))
WIDTH = int(os.environ.get("WIDTH", "768"))
DTYPE_NAME = os.environ.get("DTYPE", "float32").lower()
LIMIT = int(os.environ.get("LIMIT", "0"))

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


def load_native_lcm_pipeline(base_model: Path, dtype: torch.dtype):
    model_dir = (ROOT_DIR / base_model).resolve() if not base_model.is_absolute() else base_model
    sys.path.insert(0, str(model_dir))
    from lcm_pipeline import LatentConsistencyModelPipeline

    pipe = LatentConsistencyModelPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe = pipe.to(device=DEVICE, dtype=dtype)
    original_set_timesteps = pipe.scheduler.set_timesteps

    def set_timesteps_on_device(num_inference_steps, lcm_origin_steps=None, *args, **kwargs):
        kwargs["device"] = torch.device(DEVICE)
        if lcm_origin_steps is not None:
            kwargs["original_inference_steps"] = lcm_origin_steps
        return original_set_timesteps(num_inference_steps, *args, **kwargs)

    pipe.scheduler.set_timesteps = set_timesteps_on_device
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


def generate(pipe, prompt: str, seed: int, steps: int, dtype: torch.dtype):
    device = torch.device(DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        prompt_embeds = pipe._encode_prompt(prompt, device, 1, None)
        latents = make_latents(pipe, generator, prompt_embeds.dtype)
        pipe.scheduler.set_timesteps(
            steps,
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


def save_contact_sheet(image_rows, path: Path):
    if not image_rows:
        return
    thumb_w = 256
    label_h = 28
    src_w, src_h = image_rows[0][1].size
    thumb_h = int(src_h * thumb_w / src_w)
    cols = len(STEPS_LIST)
    rows = len(image_rows) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, img) in enumerate(image_rows):
        row = idx // cols
        col = idx % cols
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        draw.text((x + 5, y + 6), label, fill=(0, 0, 0), font=font)
        canvas.paste(img.resize((thumb_w, thumb_h)), (x, y + label_h))
    canvas.save(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if DTYPE_NAME == "float16" and DEVICE == "cuda" else torch.float32

    prompts = PROMPTS[:LIMIT] if LIMIT > 0 else PROMPTS
    seeds = SEEDS[: len(prompts)]

    print(
        "Loading native LCM pipeline: "
        f"base={BASE_MODEL}, steps={STEPS_LIST}, guidance={GUIDANCE}, "
        f"origin_steps={LCM_ORIGIN_STEPS}, size={WIDTH}x{HEIGHT}, dtype={dtype}"
    )
    pipe = load_native_lcm_pipeline(BASE_MODEL, dtype)

    contact_rows = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds), start=1):
        safe_prefix = f"{i:02d}"
        print(f"[{i}/{len(prompts)}] seed={seed}: {prompt}")
        for steps in STEPS_LIST:
            img = generate(pipe, prompt, seed, steps, dtype)
            out_path = OUTPUT_DIR / f"{safe_prefix}_steps{steps}_seed{seed}.png"
            img.save(out_path)
            contact_rows.append((f"{safe_prefix} steps={steps}", img))
            print(f"  saved {out_path}")

    save_contact_sheet(contact_rows, OUTPUT_DIR / "contact_sheet.png")
    print(f"DONE: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
