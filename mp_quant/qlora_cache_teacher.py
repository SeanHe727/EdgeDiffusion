#!/usr/bin/env python3
"""Cache FP16 SDXL-Lightning teacher outputs for Q-LoRA student training.

For each calibration prompt, run the full 4-step denoising trajectory with
the fp16 teacher UNet. At every timestep, record:
  - latent_in        : scheduler.scale_model_input(latent, t)  [1, 4, H/8, W/8]
  - timestep         : scalar
  - prompt_embeds    : encoder_hidden_states [1, 77, 2048]
  - pooled_embeds    : added_cond_kwargs['text_embeds'] [1, 1280]
  - time_ids         : added_cond_kwargs['time_ids'] [1, 6]
  - teacher_noise_pred: fp16 UNet output (the distillation target)

Then advance scheduler with teacher noise_pred so the next step's latent is on
the teacher's trajectory.

Cache is a flat directory of .safetensors files (one per sample) plus an
index.json listing them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"


def load_prompts(dataset_dir: Path, n: int, seed: int) -> list[str]:
    import random
    rng = random.Random(seed)
    rows = []
    for p in sorted(dataset_dir.glob("*.txt")):
        t = " ".join(p.read_text(encoding="utf-8", errors="ignore").replace("\n", " ").split())
        if 15 <= len(t) <= 300:
            rows.append(t)
    rng.shuffle(rows)
    return rows[:n]


@torch.no_grad()
def encode_prompts(pipe, prompts: list[str], device: str, dtype: torch.dtype):
    """Use the pipeline's built-in dual-CLIP encoding."""
    enc = []
    for p in prompts:
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=p, prompt_2=None, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        enc.append((prompt_embeds.to(dtype), pooled.to(dtype)))
    return enc


def make_time_ids(height: int, width: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    # SDXL: time_ids = [orig_h, orig_w, crop_top=0, crop_left=0, target_h, target_w]
    return torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype)


@torch.no_grad()
def cache_one_prompt(pipe, unet, prompt_embeds, pooled, time_ids, args, device, dtype, prompt_idx, out_dir):
    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    init_noise_sigma = pipe.scheduler.init_noise_sigma

    # Initial latent: random per prompt (seeded)
    torch.manual_seed(args.seed_base + prompt_idx * 9973)
    latent = torch.randn(
        (1, 4, args.height // 8, args.width // 8),
        device=device, dtype=dtype,
    ) * init_noise_sigma

    added_cond = {"text_embeds": pooled, "time_ids": time_ids}
    samples = []
    for step_i, t in enumerate(timesteps):
        pipe.scheduler._step_index = step_i
        latent_in = pipe.scheduler.scale_model_input(latent, t)
        noise_pred = unet(latent_in, t,
                          encoder_hidden_states=prompt_embeds,
                          added_cond_kwargs=added_cond).sample
        # Save (input, target) tuple
        sample_fname = f"prompt{prompt_idx:04d}_step{step_i}.safetensors"
        save_file({
            "latent_in":        latent_in.detach().contiguous().cpu(),
            "timestep":         (t if torch.is_tensor(t) else torch.tensor(t)).detach().contiguous().cpu(),
            "prompt_embeds":    prompt_embeds.detach().contiguous().cpu(),
            "pooled_embeds":    pooled.detach().contiguous().cpu(),
            "time_ids":         time_ids.detach().contiguous().cpu(),
            "teacher_pred":     noise_pred.detach().contiguous().cpu(),
        }, str(out_dir / sample_fname))
        samples.append(sample_fname)
        # Advance teacher latent
        pipe.scheduler._step_index = step_i
        latent = pipe.scheduler.step(noise_pred, t, latent).prev_sample
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt-count", type=int, default=128)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=20260518)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(args.repo_root / ".tmp"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.float16

    print(f"Loading FP16 SDXL-Lightning UNet ({args.steps}-step)...")
    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(device, dtype)
    ckpt = hf_hub_download(LIGHTNING_REPO, f"sdxl_lightning_{args.steps}step_unet.safetensors")
    unet.load_state_dict(load_file(ckpt, device=device), strict=False)
    unet.eval()

    print("Building pipeline (for text encoders + scheduler)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, unet=unet, torch_dtype=dtype,
        variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    prompts = load_prompts(args.repo_root / "dataset", args.prompt_count, args.seed_base)
    print(f"Loaded {len(prompts)} prompts from dataset/")

    print("Encoding prompts via dual CLIP...")
    encoded = encode_prompts(pipe, prompts, device, dtype)

    time_ids = make_time_ids(args.height, args.width, device, dtype)

    print(f"\nCaching teacher trajectories: {len(prompts)} prompts x {args.steps} steps = "
          f"{len(prompts)*args.steps} samples")
    print(f"Output: {args.output_dir}")
    all_samples: list[dict] = []
    for i, (prompt_text, (pe, pl)) in enumerate(zip(prompts, encoded)):
        files = cache_one_prompt(pipe, unet, pe, pl, time_ids, args, device, dtype, i, args.output_dir)
        for step_i, fname in enumerate(files):
            all_samples.append({"prompt_idx": i, "step_idx": step_i, "file": fname, "prompt": prompt_text})
        if (i + 1) % 16 == 0 or i == len(prompts) - 1:
            print(f"  {i+1}/{len(prompts)}: '{prompt_text[:60]}'")

    index = {
        "base_model": BASE_MODEL, "lightning_repo": LIGHTNING_REPO,
        "steps": args.steps, "height": args.height, "width": args.width,
        "prompt_count": len(prompts),
        "samples": all_samples,
    }
    (args.output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    n_bytes = sum((args.output_dir / s["file"]).stat().st_size for s in all_samples)
    print(f"\nDone. {len(all_samples)} samples, {n_bytes/1024/1024:.1f} MB total.")
    print(f"Index: {args.output_dir / 'index.json'}")


if __name__ == "__main__":
    main()
