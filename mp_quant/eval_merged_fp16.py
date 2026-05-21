#!/usr/bin/env python3
"""Phase A verification — load the merged fp16 UNet (from trt_prepare_fp16_unet.py)
into a vanilla UNet2DConditionModel and run the 64-prompt eval vs FP16 baseline.

Pass condition: MSE mean ≈ 0.00794 (matches the packed+LoRA α=4 deployment).
A large deviation indicates the LoRA merge or per-group dequant is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-safetensors", type=Path, required=True,
                    help="The output of trt_prepare_fp16_unet.py")
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Has fp16_*.png + selected_prompts.json (FP16 baseline)")
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--prompt-count", type=int, default=64)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"; dtype = torch.float16

    print(f"Loading merged fp16 UNet from {args.merged_safetensors.name}...")
    state = load_file(str(args.merged_safetensors), device=str(device))
    unet = UNet2DConditionModel.from_config(args.base_model, subfolder="unet").to(device, dtype)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    print(f"  loaded {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  first 3 missing keys: {missing[:3]}")
    if unexpected:
        print(f"  first 3 unexpected keys: {unexpected[:3]}")
    del state
    torch.cuda.empty_cache()
    after_load_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"Building SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, unet=unet, torch_dtype=dtype, variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    sp = args.baseline_dir / "selected_prompts.json"
    if not sp.exists():
        sys.exit(f"Missing {sp}")
    items = json.loads(sp.read_text(encoding="utf-8"))[: args.prompt_count]
    print(f"Eval set: {len(items)} prompts at {args.width}x{args.height}, {args.steps} steps")

    torch.cuda.reset_peak_memory_stats()
    metrics = []
    for it in items:
        idx, seed, prompt, src = it["index"], it["seed"], it["prompt"], it["source"]
        gen = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                   height=args.height, width=args.width, generator=gen).images[0]
        out_fname = args.output_dir / f"merged_fp16_{idx:02d}_seed{seed}_{Path(src).stem}.png"
        img.save(out_fname)
        fp16_cands = list(args.baseline_dir.glob(f"fp16_{idx:02d}_seed{seed}_*.png"))
        if fp16_cands:
            ref = np.asarray(Image.open(fp16_cands[0]).convert("RGB"), dtype=np.float32) / 255.0
            cur = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            mse = float(np.mean((ref - cur) ** 2))
            mae = float(np.mean(np.abs(ref - cur)))
            metrics.append({"index": idx, "seed": seed, "mse": mse, "mae": mae,
                            "category": it.get("category"), "prompt": prompt[:80]})
        if idx % 8 == 0 or idx == len(items):
            tail = f"  mse={metrics[-1]['mse']:.5f}" if metrics else ""
            print(f"  [{idx:02d}/{len(items)}] {prompt[:60]}{tail}")
    gen_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    summary = None
    if metrics:
        mses = [m["mse"] for m in metrics]
        maes = [m["mae"] for m in metrics]
        summary = {
            "mode": "merged_fp16",
            "n_images": len(metrics),
            "mse_mean": float(np.mean(mses)), "mae_mean": float(np.mean(maes)),
            "mse_max":  float(np.max(mses)),  "mae_max":  float(np.max(maes)),
            "after_load_peak_mb": after_load_peak,
            "generation_peak_mb": gen_peak,
            "metrics": metrics,
        }
        (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n=== Merged FP16 RESULTS ({len(metrics)} prompts) ===")
        print(f"  MSE mean: {summary['mse_mean']:.5f}  (target ≈ 0.00794 from packed+LoRA α=4)")
        print(f"  MAE mean: {summary['mae_mean']:.5f}  (target ≈ 0.04610)")
        print(f"  MSE max : {summary['mse_max']:.5f}")
        print(f"  After-load VRAM peak: {after_load_peak:.0f} MB")
        print(f"  Generation peak VRAM: {gen_peak:.0f} MB")
        delta = summary['mse_mean'] - 0.00794
        if abs(delta) < 0.0005:
            print(f"  ✓ PASS — MSE within ±0.0005 of expected. LoRA merge is correct.")
        else:
            print(f"  ✗ FAIL — MSE delta = {delta:+.5f}. Inspect LoRA scaling / dequant order.")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
