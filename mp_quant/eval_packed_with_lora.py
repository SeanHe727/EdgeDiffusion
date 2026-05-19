#!/usr/bin/env python3
"""Evaluation: generate images from packed UNet (+ optional Q-LoRA) and
compare to FP16 baseline.

Modes:
  --no-lora              : packed only (matches Stage 3 baseline)
  --lora <path>          : packed + Q-LoRA adapters loaded from path

Compares against `fp16_*.png` in --baseline-dir (the FP16 baseline images
produced by the runner alongside fake-quant images). Reports per-prompt and
aggregate MSE/MAE against FP16 truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mp_quant.build_packed_unet import build_packed_unet
from mp_quant.qlora_lora import (
    load_lora_state_dict, wrap_packed_linears_with_lora,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-safetensors", type=Path, required=True)
    ap.add_argument("--packed-manifest", type=Path, required=True)
    ap.add_argument("--lora", type=Path, default=None,
                    help="Path to lora .safetensors (lora_final.safetensors). Omit for packed-only.")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Dir containing fp16_*.png + selected_prompts.json")
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt-count", type=int, default=64)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(args.repo_root / ".tmp"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.float16

    print(f"Loading packed manifest...")
    manifest = json.loads(args.packed_manifest.read_text(encoding="utf-8"))
    print(f"Loading packed state (CPU)...")
    packed_state = load_file(str(args.packed_safetensors), device="cpu")
    print(f"Building packed UNet...")
    unet, stats = build_packed_unet(packed_state, manifest, device, dtype)
    del packed_state
    torch.cuda.empty_cache()
    print(f"  packed: {stats['n_w4']} W4 + {stats['n_w8']} W8 layers")

    mode_label = "packed_only"
    if args.lora is not None:
        print(f"Wrapping with LoRA r={args.lora_rank}, alpha={args.lora_alpha}...")
        n_wrapped = wrap_packed_linears_with_lora(unet, r=args.lora_rank, alpha=args.lora_alpha, dtype=dtype)
        print(f"  wrapped {n_wrapped} layers; loading LoRA state from {args.lora.name}")
        lora_state = load_file(str(args.lora), device=str(device))
        load_lora_state_dict(unet, lora_state)
        mode_label = f"packed_lora_r{args.lora_rank}"
    unet.eval()
    if hasattr(unet, "disable_gradient_checkpointing"):
        unet.disable_gradient_checkpointing()

    print(f"Building SDXL pipeline with this UNet...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        manifest["base_model"], unet=unet, torch_dtype=dtype,
        variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    sp = args.baseline_dir / "selected_prompts.json"
    if not sp.exists():
        sys.exit(f"Missing {sp}")
    items = json.loads(sp.read_text(encoding="utf-8"))[: args.prompt_count]
    print(f"Loaded {len(items)} prompts from {sp.name}")

    print(f"\nGenerating {len(items)} images at {args.width}x{args.height}...")
    metrics = []
    for it in items:
        idx, seed, prompt, src = it["index"], it["seed"], it["prompt"], it["source"]
        gen = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                   height=args.height, width=args.width, generator=gen).images[0]
        out_fname = args.output_dir / f"{mode_label}_{idx:02d}_seed{seed}_{Path(src).stem}.png"
        img.save(out_fname)
        # Compare to FP16 baseline
        fp16_cands = list(args.baseline_dir.glob(f"fp16_{idx:02d}_seed{seed}_*.png"))
        if fp16_cands:
            ref = np.asarray(Image.open(fp16_cands[0]).convert("RGB")).astype(np.float32) / 255.0
            cur = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
            mse = float(np.mean((ref - cur) ** 2))
            mae = float(np.mean(np.abs(ref - cur)))
            metrics.append({"index": idx, "seed": seed, "mse": mse, "mae": mae,
                            "category": it.get("category"), "prompt": prompt[:80]})
        if idx % 8 == 0 or idx == len(items):
            print(f"  [{idx:02d}/{len(items)}] seed={seed} {prompt[:60]}"
                  + (f"  mse_vs_fp16={metrics[-1]['mse']:.5f}" if metrics else ""))

    if metrics:
        mses = [m["mse"] for m in metrics]
        maes = [m["mae"] for m in metrics]
        order = np.argsort(mses)[::-1]
        summary = {
            "mode": mode_label, "n_images": len(metrics),
            "mse_mean": float(np.mean(mses)), "mae_mean": float(np.mean(maes)),
            "mse_max": float(np.max(mses)),  "mae_max": float(np.max(maes)),
            "metrics": metrics,
            "worst_5": [metrics[int(i)] for i in order[:5]],
            "best_5":  [metrics[int(i)] for i in order[-5:][::-1]],
        }
        print(f"\nVs FP16 baseline ({len(metrics)} pairs):")
        print(f"  MSE mean = {summary['mse_mean']:.6f}")
        print(f"  MAE mean = {summary['mae_mean']:.6f}")
        print(f"  MSE max  = {summary['mse_max']:.6f}")
        print(f"  MAE max  = {summary['mae_max']:.6f}")
        print(f"\nWorst 5 prompts:")
        for m in summary["worst_5"]:
            print(f"  #{m['index']:02d} mse={m['mse']:.4f} [{m.get('category','?')}] {m['prompt']}")
        (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
