#!/usr/bin/env python3
"""Stage 4: end-to-end eval for torchao Int8WeightOnly on SDXL-Lightning UNet.

Loads fp16 UNet, applies torchao Int8WeightOnlyConfig to all Linear layers
(except SKIP_PATTERNS). Generates the same 64 prompts as our baseline and
reports per-image MSE/MAE vs the existing fp16 baseline + latency + VRAM.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
SKIP_PATTERNS = ("time_embedding", "time_emb_proj", "add_embedding", "conv_in", "conv_out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Dir with fp16_*.png + selected_prompts.json (existing FP16 baseline)")
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt-count", type=int, default=64)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--group-size", type=int, default=128,
                    help="Int8 per-group size (None = per-channel which fails on SDXL).")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"; dtype = torch.float16

    torch.cuda.reset_peak_memory_stats()
    print(f"Loading FP16 SDXL-Lightning UNet ({args.steps}-step)...")
    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(device, dtype)
    ckpt = hf_hub_download(LIGHTNING_REPO, f"sdxl_lightning_{args.steps}step_unet.safetensors")
    unet.load_state_dict(load_file(ckpt, device=device), strict=False)

    print("Applying torchao Int8WeightOnlyConfig...")
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    target = {n for n, m in unet.named_modules()
              if isinstance(m, nn.Linear) and not any(p in n for p in SKIP_PATTERNS)}
    print(f"  targets: {len(target)} Linears (skipping conv_in/conv_out/embeddings)")
    cfg = Int8WeightOnlyConfig(group_size=args.group_size, version=1)
    print(f"  config: Int8WeightOnlyConfig(group_size={args.group_size}, version=1)")
    quantize_(unet, cfg, filter_fn=lambda m, n: n in target)
    torch.cuda.empty_cache()
    after_quant_alloc = torch.cuda.memory_allocated() / 1024 / 1024
    after_quant_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  after quant: current={after_quant_alloc:.0f} MB, peak={after_quant_peak:.0f} MB")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, unet=unet, torch_dtype=dtype, variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    items = json.loads((args.baseline_dir / "selected_prompts.json").read_text(encoding="utf-8"))[: args.prompt_count]
    print(f"Eval set: {len(items)} prompts at {args.width}x{args.height}, {args.steps} steps")

    torch.cuda.reset_peak_memory_stats()
    metrics = []
    latencies = []
    for it in items:
        idx, seed, prompt, src = it["index"], it["seed"], it["prompt"], it["source"]
        gen = torch.Generator(device=device).manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                       height=args.height, width=args.width, generator=gen).images[0]
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        out_fname = args.output_dir / f"torchao_int8_{idx:02d}_seed{seed}_{Path(src).stem}.png"
        img.save(out_fname)
        fp16_cands = list(args.baseline_dir.glob(f"fp16_{idx:02d}_seed{seed}_*.png"))
        if fp16_cands:
            ref = np.asarray(Image.open(fp16_cands[0]).convert("RGB"), dtype=np.float32) / 255.0
            cur = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            mse = float(np.mean((ref - cur) ** 2)); mae = float(np.mean(np.abs(ref - cur)))
            metrics.append({"index": idx, "seed": seed, "mse": mse, "mae": mae,
                            "category": it.get("category"), "prompt": prompt[:80]})
        if idx % 8 == 0 or idx == len(items):
            print(f"  [{idx:02d}/{len(items)}] {latencies[-1]:.0f}ms  {prompt[:60]}"
                  + (f"  mse={metrics[-1]['mse']:.5f}" if metrics else ""))
    gen_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    if metrics:
        mses = [m["mse"] for m in metrics]
        maes = [m["mae"] for m in metrics]
        summary = {
            "mode": "torchao_int8_weight_only",
            "n_images": len(metrics),
            "mse_mean": float(np.mean(mses)), "mae_mean": float(np.mean(maes)),
            "mse_max":  float(np.max(mses)),  "mae_max":  float(np.max(maes)),
            "latency_ms_median": statistics.median(latencies),
            "latency_ms_mean":   statistics.mean(latencies),
            "latency_ms_min":    min(latencies),
            "latency_ms_max":    max(latencies),
            "after_quant_alloc_mb": after_quant_alloc,
            "generation_peak_mb":   gen_peak,
            "metrics": metrics,
        }
        (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n=== torchao Int8 RESULTS ({len(metrics)} prompts) ===")
        print(f"  MSE mean: {summary['mse_mean']:.5f}  MAE mean: {summary['mae_mean']:.5f}")
        print(f"  MSE max:  {summary['mse_max']:.5f}   MAE max:  {summary['mae_max']:.5f}")
        print(f"  Latency:  median={summary['latency_ms_median']:.0f}ms  "
              f"mean={summary['latency_ms_mean']:.0f}ms  "
              f"min={summary['latency_ms_min']:.0f}ms  max={summary['latency_ms_max']:.0f}ms")
        print(f"  After quant alloc:    {after_quant_alloc:.0f} MB")
        print(f"  Generation peak VRAM: {gen_peak:.0f} MB")


if __name__ == "__main__":
    main()
