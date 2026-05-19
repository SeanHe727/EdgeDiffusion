#!/usr/bin/env python3
"""Standalone tester: load packed SDXL-Lightning artifact -> dequant -> generate images.

Designed to run as its own process (no concurrent baseline pipeline) so peak
RAM stays low. Steps:
  1. Read packed safetensors + manifest from disk.
  2. Reconstruct fp16 state dict (W4 nibble unpack, W8 cast, FP16 pass-through).
  3. Build a fresh UNet shell from base SDXL config; load_state_dict.
  4. Build SDXL pipeline using this UNet; generate test images.

If --baseline-dir is provided, the corresponding `quantw*_*.png` files from
that dir are compared against the freshly generated images (per-pixel MSE/MAE).
Used to confirm packing is lossless: in-memory fake-quant generation should
match unpacked-from-disk generation bit-for-bit (modulo nondeterminism).
"""

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


def unpack_int4(packed: torch.Tensor, out_f: int, in_f: int) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(out_f, in_f, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = low.to(torch.int8)
    out[:, 1::2] = high.to(torch.int8)
    return out


def dequant_state(packed_state: dict, manifest: dict, device: str, dtype=torch.float16) -> dict:
    out: dict[str, torch.Tensor] = {}
    for key in manifest["fp16_passthrough"]:
        if key in packed_state:
            out[key] = packed_state[key].to(device, dtype)
    gs = manifest["group_size"]
    for layer_name, info in manifest["layers"].items():
        packed = packed_state[f"{layer_name}.weight_packed"].to(device)
        scale = packed_state[f"{layer_name}.scale"].to(device, dtype)
        out_f, in_f = info["shape"]
        bits = info["bits"]
        q = unpack_int4(packed, out_f, in_f) if bits == 4 else packed
        n_groups = in_f // gs
        q_g = q.view(out_f, n_groups, gs).to(dtype)
        w = (q_g * scale.unsqueeze(2)).view(out_f, in_f)
        out[f"{layer_name}.weight"] = w
    return out


def load_prompts_from_baseline(baseline_dir: Path, count: int) -> list[dict]:
    selected = baseline_dir / "selected_prompts.json"
    if not selected.exists():
        return []
    items = json.loads(selected.read_text(encoding="utf-8"))
    return items[:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-safetensors", type=Path, required=True)
    ap.add_argument("--packed-manifest", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="Optional: dir of fake-quant images. If given, uses its prompts/seeds and computes per-image MSE/MAE.")
    ap.add_argument("--prompt-count", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--default-seed-base", type=int, default=12345)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(args.repo_root / ".tmp"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {args.packed_manifest.name}")
    manifest = json.loads(args.packed_manifest.read_text(encoding="utf-8"))
    sz = manifest["size"]
    print(f"  layers={len(manifest['layers'])}  fp16-passthrough={len(manifest['fp16_passthrough'])}")
    print(f"  packed: {sz['packed_total_bytes']/1024/1024:.1f} MB, reduction {sz['reduction']*100:.1f}%")

    print(f"Loading packed safetensors (CPU): {args.packed_safetensors.name}")
    packed_state = load_file(str(args.packed_safetensors), device="cpu")
    print(f"  {len(packed_state)} tensors")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print("Dequantizing into fp16 state on GPU...")
    full_state = dequant_state(packed_state, manifest, device, dtype)
    del packed_state
    torch.cuda.empty_cache()
    print(f"  state-dict entries: {len(full_state)}")

    print("Building UNet shell + loading dequant state...")
    unet = UNet2DConditionModel.from_config(manifest["base_model"], subfolder="unet").to(device, dtype)
    missing, unexpected = unet.load_state_dict(full_state, strict=False)
    print(f"  missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    if missing:
        print(f"  first 5 missing: {missing[:5]}")
    del full_state
    torch.cuda.empty_cache()

    print("Building SDXL pipeline with dequanted UNet...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        manifest["base_model"], unet=unet, torch_dtype=dtype,
        variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=False)

    baseline_items = []
    if args.baseline_dir is not None:
        baseline_items = load_prompts_from_baseline(args.baseline_dir, args.prompt_count)
        if not baseline_items:
            print(f"WARNING: --baseline-dir given but no selected_prompts.json found; falling back")
    if not baseline_items:
        defaults = [
            "A large brown horse walking through a lush green forest.",
            "A plate full of food is ready to be eaten.",
            "A bathroom with a black and white checkered floor.",
            "cars that are stopped at a traffic light",
        ]
        for i, p in enumerate(defaults[: args.prompt_count]):
            baseline_items.append({"index": i + 1, "seed": args.default_seed_base + i * 9973,
                                   "prompt": p, "source": f"default_{i+1:02d}"})

    print(f"\nGenerating {len(baseline_items)} test images at {args.width}x{args.height}, steps={args.steps}")
    metrics = []
    for it in baseline_items:
        idx, seed, prompt = it["index"], it["seed"], it["prompt"]
        gen = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                   height=args.height, width=args.width, generator=gen).images[0]
        out_fname = args.output_dir / f"unpacked_{idx:02d}_seed{seed}.png"
        img.save(out_fname)
        line = f"  [{idx:02d}] seed={seed} -> {out_fname.name}: {prompt[:60]}"
        if args.baseline_dir is not None:
            # Match against gptq_linearw*_{idx:02d}_seed{seed}_*.png in baseline dir
            candidates = list(args.baseline_dir.glob(f"gptq_linearw*_{idx:02d}_seed{seed}_*.png"))
            if not candidates:
                candidates = list(args.baseline_dir.glob(f"linearw*_{idx:02d}_seed{seed}_*.png"))
            if candidates:
                bl = np.asarray(Image.open(candidates[0]).convert("RGB")).astype(np.float32) / 255.0
                up = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
                mse = float(np.mean((bl - up) ** 2))
                mae = float(np.mean(np.abs(bl - up)))
                metrics.append({"index": idx, "mse": mse, "mae": mae, "baseline": candidates[0].name})
                line += f"  mse={mse:.6f} mae={mae:.6f}"
        print(line)

    summary = {
        "packed_safetensors": str(args.packed_safetensors),
        "size": manifest["size"],
        "metrics": metrics,
    }
    if metrics:
        summary["mse_mean"] = float(np.mean([m["mse"] for m in metrics]))
        summary["mae_mean"] = float(np.mean([m["mae"] for m in metrics]))
        summary["mse_max"] = float(np.max([m["mse"] for m in metrics]))
        print(f"\nVs baseline images: MSE mean={summary['mse_mean']:.6f}  MAE mean={summary['mae_mean']:.6f}  MSE max={summary['mse_max']:.6f}")
        print("(MSE/MAE should be near 0 if pack/unpack is lossless and generation is deterministic)")
    (args.output_dir / "unpack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
