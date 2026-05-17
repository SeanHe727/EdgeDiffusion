#!/usr/bin/env python3
"""Weight-only fake-quant smoke test for SDXL-Lightning.

Separate from the older SD-Turbo mp_quant pipeline. This script tests visual
quality after fake quantization: weights are rounded to INT grids but stored
and executed as fp16 tensors. It is a quality smoke before real packing/kernels.
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

BASE_MODEL = os.environ.get("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
LIGHTNING_REPO = os.environ.get("LIGHTNING_REPO", "ByteDance/SDXL-Lightning")
STEP = int(os.environ.get("STEP", "4"))
HEIGHT = int(os.environ.get("HEIGHT", "1024"))
WIDTH = int(os.environ.get("WIDTH", "1024"))
GUIDANCE = float(os.environ.get("GUIDANCE", "0.0"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "gen_test_output/sdxl_lightning_weight_quant_smoke"))
SAVE_UNET = os.environ.get("SAVE_UNET", "0") == "1"
DTYPE_NAME = os.environ.get("DTYPE", "float16").lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LINEAR_BITS = int(os.environ.get("LINEAR_BITS", os.environ.get("BITS", "4")))
CONV_BITS_RAW = os.environ.get("CONV_BITS", "none").lower()
CONV_BITS = None if CONV_BITS_RAW in ("", "none", "fp16", "off") else int(CONV_BITS_RAW)
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "128"))
TARGET = os.environ.get("TARGET", "all")  # all | attention | ff | linear | conv
PROMPT_LIMIT = int(os.environ.get("PROMPT_LIMIT", "32"))

PROMPTS = [
    "close-up portrait photo of a young woman, detailed eyes, natural skin texture, soft studio light, 85mm lens",
    "a fluffy golden retriever puppy sitting in autumn leaves, sharp fur detail, professional photography",
    "a cyberpunk city street at night, neon signs, rain reflections, cinematic, highly detailed",
    "a beautiful mountain lake at sunrise, mist, dramatic clouds, ultra detailed landscape photography",
    "a tabby cat sitting on a wooden windowsill, detailed whiskers and eyes, warm morning light, photography",
    "close-up portrait photo of an elderly man with grey beard, detailed wrinkles, dramatic side lighting",
    "a red fox standing in a snowy forest, detailed fur, wildlife photography, shallow depth of field",
    "a cozy coffee shop interior, morning light through windows, detailed furniture, realistic photography",
    "a futuristic robot chef cooking in a small kitchen, cinematic lighting, highly detailed",
    "a medieval castle on a cliff above the ocean, storm clouds, epic fantasy concept art",
    "a macro photo of a red apple with water droplets, sharp focus, studio lighting",
    "an astronaut riding a horse on the moon, surreal, high quality, detailed suit",
    "a realistic photo of a golden retriever running through a meadow, motion, sunlight",
    "a fashion editorial portrait, black dress, clean background, detailed fabric, studio flash",
    "a wooden cabin in a pine forest at night, warm windows, snow, cinematic realism",
    "a bowl of ramen on a wooden table, steam, realistic food photography, rich details",
    "a vintage red sports car parked on a rainy street, reflections, cinematic photography",
    "a fantasy dragon flying above a mountain valley, detailed scales, golden sunset",
    "a modern living room with plants and bookshelves, natural light, interior photography",
    "a close-up photo of a tiger face, detailed fur and eyes, wildlife photography",
    "a watercolor painting of a small village by a river, soft colors, detailed houses",
    "a product photo of wireless headphones on a black background, rim lighting, sharp detail",
    "a street portrait of a man wearing a leather jacket, neon city background, 50mm lens",
    "a detailed isometric city block, tiny cars and people, clean daylight, high detail",
    "a white ceramic teapot with blue flowers, soft studio light, product photography",
    "a sailboat on a calm lake at golden hour, reflections, realistic landscape photography",
    "a close-up portrait of a child laughing, natural skin texture, soft outdoor light",
    "an ancient library with tall shelves and ladders, warm lamps, cinematic detail",
    "a black horse running on a beach, ocean waves, dramatic sky, photography",
    "a sci-fi spaceship interior cockpit, glowing panels, realistic, highly detailed",
    "a bouquet of wildflowers in a glass vase, sunlight, macro photography, delicate petals",
    "a panda eating bamboo in a green forest, detailed fur, wildlife photography",
]
SEEDS = [123, 7, 2024, 42, 256, 999, 55, 314, 808, 17, 64, 91, 777, 1001, 222, 333,
         444, 555, 666, 888, 111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999,
         101, 202, 303]

SKIP_PATTERNS = (
    "time_embedding",
    "time_emb_proj",
    "add_embedding",
    "conv_in",
    "conv_out",
)


def is_skipped(name: str) -> bool:
    return any(p in name for p in SKIP_PATTERNS)


def should_quantize_linear(name: str, mod: nn.Module) -> bool:
    if not isinstance(mod, nn.Linear) or is_skipped(name):
        return False
    if TARGET == "conv":
        return False
    if TARGET == "attention":
        return ".attn" in name or "attentions" in name
    if TARGET == "ff":
        return ".ff." in name or "ff.net" in name
    return TARGET in ("all", "linear")


def should_quantize_conv(name: str, mod: nn.Module) -> bool:
    if not isinstance(mod, nn.Conv2d) or is_skipped(name):
        return False
    return TARGET in ("all", "conv")


@torch.no_grad()
def fake_quant_2d_(weight_2d: torch.Tensor, bits: int, group_size: int) -> dict:
    orig_dtype = weight_2d.dtype
    w = weight_2d.detach().float()
    out_f, in_f = w.shape
    if group_size <= 0:
        group_size = in_f
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    out = torch.empty_like(w)
    total_mse = 0.0
    for start in range(0, in_f, group_size):
        end = min(start + group_size, in_f)
        block = w[:, start:end]
        scale = block.abs().amax(dim=1, keepdim=True) / qmax
        scale = scale.clamp(min=1e-8)
        q = torch.round(block / scale).clamp(qmin, qmax)
        deq = q * scale
        out[:, start:end] = deq
        total_mse += torch.mean((block - deq) ** 2).item() * (end - start)
    weight_2d.copy_(out.to(orig_dtype))
    return {"mse": total_mse / in_f, "out_features": out_f, "in_features": in_f}


@torch.no_grad()
def fake_quant_module_weight_(mod: nn.Module, bits: int, group_size: int) -> dict:
    w = mod.weight
    if isinstance(mod, nn.Linear):
        return fake_quant_2d_(w, bits, group_size)
    if isinstance(mod, nn.Conv2d):
        original_shape = w.shape
        flat = w.view(w.shape[0], -1)
        stats = fake_quant_2d_(flat, bits, group_size)
        w.copy_(flat.view(original_shape).to(w.dtype))
        stats["kernel_shape"] = list(original_shape)
        return stats
    raise TypeError(type(mod).__name__)


def load_lightning_unet(dtype):
    ckpt = f"sdxl_lightning_{STEP}step_unet.safetensors"
    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(DEVICE, dtype)
    ckpt_path = hf_hub_download(LIGHTNING_REPO, ckpt)
    state = load_file(ckpt_path, device=DEVICE)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    print(f"Loaded Lightning UNet: {ckpt}, missing={len(missing)}, unexpected={len(unexpected)}")
    return unet


def build_pipe(unet, dtype):
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
    return pipe


@torch.no_grad()
def generate(pipe, prompt, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(
        prompt,
        num_inference_steps=STEP,
        guidance_scale=GUIDANCE,
        height=HEIGHT,
        width=WIDTH,
        generator=gen,
    ).images[0]


def apply_fake_quant(unet):
    stats = []
    before_params = 0
    after_bits = 0
    print(
        f"Applying fake quant: linear_bits={LINEAR_BITS}, conv_bits={CONV_BITS}, "
        f"group_size={GROUP_SIZE}, target={TARGET}"
    )
    for name, mod in unet.named_modules():
        bit = None
        kind = None
        if should_quantize_linear(name, mod):
            bit = LINEAR_BITS
            kind = "Linear"
        elif CONV_BITS is not None and should_quantize_conv(name, mod):
            bit = CONV_BITS
            kind = "Conv2d"
        else:
            continue
        n = mod.weight.numel()
        before_params += n
        after_bits += n * bit
        s = fake_quant_module_weight_(mod, bit, GROUP_SIZE)
        stats.append({"name": name, "type": kind, "bits": bit, "params": n, **s})
        if len(stats) <= 5 or len(stats) % 50 == 0:
            print(f"  [{len(stats)}] {kind} {name}: W{bit}, {n/1e6:.2f}M params, mse={s['mse']:.3e}")
    print(f"Quantized {len(stats)} modules, {before_params/1e6:.1f} weights")
    fp16_bits = before_params * 16
    print(f"Theoretical reduction on targeted weights: {1 - after_bits / max(1, fp16_bits):.1%}")
    return stats


def save_contact_sheet(pairs, out_path):
    if not pairs:
        return
    thumb_w = 256
    label_h = 28
    row_gap = 8
    thumb_h = int(pairs[0][0].height * thumb_w / pairs[0][0].width)
    cols = 2
    sheet = Image.new("RGB", (cols * thumb_w, len(pairs) * (thumb_h + label_h + row_gap)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    q_label = f"Linear W{LINEAR_BITS}" + (f" + Conv W{CONV_BITS}" if CONV_BITS is not None else "")
    for idx, (base_img, q_img) in enumerate(pairs):
        y = idx * (thumb_h + label_h + row_gap)
        draw.text((6, y + 6), f"{idx+1:02d} fp16", fill=(0, 0, 0), font=font)
        draw.text((thumb_w + 6, y + 6), f"{idx+1:02d} {q_label}", fill=(0, 0, 0), font=font)
        sheet.paste(base_img.resize((thumb_w, thumb_h)), (0, y + label_h))
        sheet.paste(q_img.resize((thumb_w, thumb_h)), (thumb_w, y + label_h))
    sheet.save(out_path)


def compute_image_metrics(pairs):
    mses = []
    maes = []
    for a_img, b_img in pairs:
        a = np.asarray(a_img.convert("RGB")).astype(np.float32) / 255.0
        b = np.asarray(b_img.convert("RGB")).astype(np.float32) / 255.0
        mses.append(float(np.mean((a - b) ** 2)))
        maes.append(float(np.mean(np.abs(a - b))))
    return {"mse_each": mses, "mae_each": maes, "mse_mean": float(np.mean(mses)), "mae_mean": float(np.mean(maes))}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if DTYPE_NAME == "float16" and DEVICE == "cuda" else torch.float32
    prompts = PROMPTS[:PROMPT_LIMIT]
    seeds = SEEDS[: len(prompts)]
    print(
        f"SDXL-Lightning quant smoke: step={STEP}, linear_bits={LINEAR_BITS}, conv_bits={CONV_BITS}, "
        f"target={TARGET}, group={GROUP_SIZE}, prompts={len(prompts)}, size={WIDTH}x{HEIGHT}, "
        f"guidance={GUIDANCE}, dtype={dtype}, device={DEVICE}"
    )
    if LINEAR_BITS not in (4, 8):
        raise ValueError("LINEAR_BITS must be 4 or 8")
    if CONV_BITS is not None and CONV_BITS not in (4, 8):
        raise ValueError("CONV_BITS must be 4, 8, or none")

    unet = load_lightning_unet(dtype)
    pipe = build_pipe(unet, dtype)

    baseline_imgs = []
    print("Generating fp16 baseline images...")
    for i, (prompt, seed) in enumerate(zip(prompts, seeds), start=1):
        print(f"  fp16 [{i}/{len(prompts)}] seed={seed}: {prompt[:70]}")
        img = generate(pipe, prompt, seed)
        img.save(OUTPUT_DIR / f"fp16_{i:02d}_seed{seed}.png")
        baseline_imgs.append(img)

    stats = apply_fake_quant(pipe.unet)

    quant_imgs = []
    q_prefix = f"linearw{LINEAR_BITS}" + (f"_convw{CONV_BITS}" if CONV_BITS is not None else "")
    print("Generating quantized images...")
    for i, (prompt, seed) in enumerate(zip(prompts, seeds), start=1):
        print(f"  {q_prefix} [{i}/{len(prompts)}] seed={seed}: {prompt[:70]}")
        img = generate(pipe, prompt, seed)
        img.save(OUTPUT_DIR / f"{q_prefix}_{i:02d}_seed{seed}.png")
        quant_imgs.append(img)

    pairs = list(zip(baseline_imgs, quant_imgs))
    metrics = compute_image_metrics(pairs)
    save_contact_sheet(pairs, OUTPUT_DIR / "contact_sheet.png")

    with open(OUTPUT_DIR / "quant_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "lightning_repo": LIGHTNING_REPO,
            "step": STEP,
            "linear_bits": LINEAR_BITS,
            "conv_bits": CONV_BITS,
            "target": TARGET,
            "group_size": GROUP_SIZE,
            "prompt_limit": len(prompts),
            "height": HEIGHT,
            "width": WIDTH,
            "guidance": GUIDANCE,
            "skip_patterns": list(SKIP_PATTERNS),
            "image_metrics": metrics,
            "layers": stats,
        }, f, indent=2)
    print(f"Image MSE mean: {metrics['mse_mean']:.6f}; MAE mean: {metrics['mae_mean']:.6f}")

    if SAVE_UNET:
        out_st = OUTPUT_DIR / f"sdxl_lightning_{STEP}step_{q_prefix}_fakequant_unet.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in pipe.unet.state_dict().items()}, out_st)
        print(f"Saved fake-quant UNet: {out_st}")

    print(f"DONE: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
