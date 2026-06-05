#!/usr/bin/env python3
"""Sprint 2 end-to-end gate — swap diffusers UNet for our TRT engine, generate
4 images, and compare to FP16 baseline by per-pixel MSE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image

from quantization.main.TRT.trt_unet_wrapper import TRTUNetWrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--prompt-count", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", "/opt/dlami/nvme/.tmp")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path("/opt/dlami/nvme/.tmp").mkdir(parents=True, exist_ok=True)
    device = "cuda"; dtype = torch.float16

    print("Reading SDXL UNet config to seed TRTUNetWrapper.config...")
    ref_unet_cfg = UNet2DConditionModel.load_config(args.base_model, subfolder="unet")

    print(f"Loading TRT engine {args.engine}...")
    t0 = time.time()
    trt_unet = TRTUNetWrapper(args.engine, ref_unet_cfg, device=device)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("Building SDXL pipeline with TRT-backed UNet...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.unet = trt_unet
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)

    sp = args.baseline_dir / "selected_prompts.json"
    if not sp.exists():
        sys.exit(f"Missing {sp}")
    items = json.loads(sp.read_text(encoding="utf-8"))[: args.prompt_count]

    print(f"\nGenerating {len(items)} images via TRT INT8+FP16 engine...")
    metrics = []
    for it in items:
        idx, seed, prompt, src = it["index"], it["seed"], it["prompt"], it["source"]
        gen = torch.Generator(device=device).manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=args.steps,
                       guidance_scale=args.guidance,
                       height=args.height, width=args.width,
                       generator=gen).images[0]
        elapsed = time.time() - t0
        out_fname = args.output_dir / f"trt_e2e_{idx:02d}_seed{seed}_{Path(src).stem}.png"
        img.save(out_fname)
        fp16_cands = list(args.baseline_dir.glob(f"fp16_{idx:02d}_seed{seed}_*.png"))
        line = f"  [{idx:02d}/{len(items)}] {elapsed:.2f}s  {prompt[:55]}"
        if fp16_cands:
            ref = np.asarray(Image.open(fp16_cands[0]).convert("RGB"),
                             dtype=np.float32) / 255.0
            cur = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            mse = float(np.mean((ref - cur) ** 2))
            mae = float(np.mean(np.abs(ref - cur)))
            metrics.append({"index": idx, "mse": mse, "mae": mae,
                            "latency_s": elapsed})
            line += f"  mse_vs_fp16={mse:.5f}"
        print(line)

    if metrics:
        mses = [m["mse"] for m in metrics]
        lats = [m["latency_s"] for m in metrics]
        print(f"\n=== TRT INT8+FP16 vs FP16 baseline ({len(metrics)} images) ===")
        print(f"  MSE mean: {np.mean(mses):.5f}  (gate: <= 0.02)")
        print(f"  MSE max : {np.max(mses):.5f}")
        print(f"  Latency : {np.mean(lats):.2f} s/image (mean over {len(lats)})")
        if np.mean(mses) <= 0.02:
            print(f"  PASS — within quality gate.")
        else:
            print(f"  FAIL — MSE > 0.02 gate.")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
