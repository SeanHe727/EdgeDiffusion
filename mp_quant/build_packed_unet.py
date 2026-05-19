#!/usr/bin/env python3
"""Stage 3: build a UNet whose target Linear layers are PackedInt4/Int8Linear.

Loads packed safetensors + manifest, builds a UNet shell, replaces all target
Linear layers with PackedLinear modules (weights stored as INT4/INT8 buffers on
device, dequanted only inside forward()), then runs a small generation test
and reports peak VRAM.

Run two ways and diff peak VRAM to confirm Stage 3 savings:
  --mode fp16    : full fp16 UNet (baseline)
  --mode packed  : PackedLinear UNet (this stage)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mp_quant.packed_linear import PackedInt4Linear, PackedInt8Conv2d, PackedInt8Linear


def build_packed_unet(packed_state: dict, manifest: dict, device: str,
                      dtype: torch.dtype = torch.float16):
    """Build UNet with PackedLinear replacements. Replaced Linears never get
    fp16-dense weights allocated on device — only their packed buffers are loaded."""
    base_model = manifest["base_model"]
    gs = manifest["group_size"]

    # 1. Build random-init fp16 shell on CPU (cheap; weights get replaced/loaded below).
    unet = UNet2DConditionModel.from_config(base_model, subfolder="unet").to(torch.device("cpu"), dtype)

    # 2. Replace target Linears / Convs on CPU. Original random fp16 weights
    #    for these layers are dropped here, never reach GPU.
    n_w4 = n_w8_linear = n_w8_conv = 0
    for layer_name, info in manifest["layers"].items():
        bits = info["bits"]
        if bits not in (4, 8):
            continue
        packed = packed_state[f"{layer_name}.weight_packed"]
        scale = packed_state[f"{layer_name}.scale"]
        bias_key = f"{layer_name}.bias"
        bias_tensor = packed_state[bias_key].clone() if bias_key in packed_state else None
        ltype = info.get("type", "Linear")
        if ltype == "Linear":
            layer_gs = info.get("group_size", gs)
            if bits == 4:
                new = PackedInt4Linear.from_state(packed, scale, layer_gs, bias=bias_tensor, dtype=dtype)
                n_w4 += 1
            else:
                new = PackedInt8Linear.from_state(packed, scale, layer_gs, bias=bias_tensor, dtype=dtype)
                n_w8_linear += 1
        elif ltype == "Conv2d":
            new = PackedInt8Conv2d.from_state(packed, scale, info, bias=bias_tensor, dtype=dtype)
            n_w8_conv += 1
        else:
            continue
        parent_fqn, leaf = layer_name.rsplit(".", 1)
        parent = unet.get_submodule(parent_fqn)
        setattr(parent, leaf, new)
    n_w8 = n_w8_linear + n_w8_conv

    # 3. Load fp16 passthrough tensors (conv, norm, embeddings, biases for non-quantized
    #    layers) via state_dict. After replacement, the keys we still need are the
    #    passthrough ones whose paths still exist in unet.
    current_keys = set(dict(unet.state_dict()).keys())
    target_state = {k: packed_state[k] for k in manifest["fp16_passthrough"]
                    if k in packed_state and k in current_keys}
    missing, unexpected = unet.load_state_dict(target_state, strict=False)
    # `missing` will include keys we replaced (weight_packed / scale buffers' parent
    # had its `.weight` removed). Filter for genuinely unexpected misses.
    suspect = [m for m in missing if not m.endswith(".weight_packed")
               and not m.endswith(".scale") and m not in current_keys]
    if suspect:
        print(f"WARNING: {len(suspect)} unexpected missing keys after load. First 5: {suspect[:5]}")

    # 4. Move whole UNet to GPU.
    unet = unet.to(device)
    return unet, {"n_w4": n_w4, "n_w8": n_w8,
                  "n_w8_linear": n_w8_linear, "n_w8_conv": n_w8_conv}


def load_fp16_unet(steps: int, device: str, dtype: torch.dtype, base_model: str, lightning_repo: str):
    """Load full fp16 SDXL-Lightning UNet."""
    unet = UNet2DConditionModel.from_config(base_model, subfolder="unet").to(device, dtype)
    ckpt = hf_hub_download(lightning_repo, f"sdxl_lightning_{steps}step_unet.safetensors")
    state = load_file(ckpt, device=device)
    unet.load_state_dict(state, strict=False)
    return unet


def measure_peak_vram(label: str) -> float:
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
    print(f"  [{label}] peak_alloc={peak:.0f} MB,  peak_reserved={reserved:.0f} MB")
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fp16", "packed"), required=True)
    ap.add_argument("--packed-safetensors", type=Path, default=None)
    ap.add_argument("--packed-manifest", type=Path, default=None)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt-count", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--ckpt-steps", type=int, default=None,
                    help="Which SDXL-Lightning checkpoint to load (1/2/4/8). "
                         "Defaults to --steps. Set explicitly when --steps != native distilled count.")
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="Optional: dir of fake-quant images for MSE comparison.")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(args.repo_root / ".tmp"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.float16

    if args.mode == "fp16":
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        lightning_repo = "ByteDance/SDXL-Lightning"
        torch.cuda.reset_peak_memory_stats()
        print(f"[FP16] Loading full UNet...")
        ckpt_steps = args.ckpt_steps if args.ckpt_steps is not None else args.steps
        unet = load_fp16_unet(ckpt_steps, device, dtype, base_model, lightning_repo)
        n_w4 = n_w8 = 0
    else:
        if not args.packed_safetensors or not args.packed_manifest:
            sys.exit("--mode packed requires --packed-safetensors and --packed-manifest")
        print(f"[PACKED] Loading manifest: {args.packed_manifest.name}")
        manifest = json.loads(args.packed_manifest.read_text(encoding="utf-8"))
        base_model = manifest["base_model"]
        lightning_repo = manifest["lightning_repo"]
        print(f"[PACKED] Loading packed state (CPU)...")
        packed_state = load_file(str(args.packed_safetensors), device="cpu")
        print(f"  {len(packed_state)} tensors")
        torch.cuda.reset_peak_memory_stats()
        print(f"[PACKED] Building UNet with PackedLinear replacements...")
        unet, stats = build_packed_unet(packed_state, manifest, device, dtype)
        n_w4, n_w8 = stats["n_w4"], stats["n_w8"]
        del packed_state
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  replaced: {n_w4} W4, {n_w8} W8 Linear modules")

    peak_after_unet = measure_peak_vram("after unet load")

    print(f"Building SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, unet=unet, torch_dtype=dtype,
        variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    peak_after_pipe = measure_peak_vram("after pipe build")

    # Resolve prompts: either from baseline-dir selected_prompts.json or defaults
    items = []
    if args.baseline_dir is not None:
        sp = args.baseline_dir / "selected_prompts.json"
        if sp.exists():
            items = json.loads(sp.read_text(encoding="utf-8"))[: args.prompt_count]
            # Mirror prompts list into output_dir so downstream eval can re-key.
            (args.output_dir / "selected_prompts.json").write_text(
                json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    if not items:
        for i, p in enumerate([
            "A large brown horse walking through a lush green forest.",
            "A plate full of food is ready to be eaten.",
            "A bathroom with a black and white checkered floor.",
            "cars that are stopped at a traffic light",
        ][: args.prompt_count]):
            items.append({"index": i + 1, "seed": 12345 + i * 9973, "prompt": p,
                          "source": f"default_{i+1:02d}"})

    print(f"\nGenerating {len(items)} images at {args.width}x{args.height}, steps={args.steps}")
    torch.cuda.reset_peak_memory_stats()
    metrics = []
    for it in items:
        idx, seed, prompt = it["index"], it["seed"], it["prompt"]
        gen = torch.Generator(device=device).manual_seed(seed)
        img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                   height=args.height, width=args.width, generator=gen).images[0]
        src_stem = Path(it.get("source", "")).stem
        suffix = f"_{src_stem}" if src_stem else ""
        out_fname = args.output_dir / f"{args.mode}_{idx:02d}_seed{seed}{suffix}.png"
        img.save(out_fname)
        line = f"  [{idx:02d}] seed={seed} -> {out_fname.name}: {prompt[:60]}"
        if args.baseline_dir is not None:
            cands = list(args.baseline_dir.glob(f"gptq_linearw*_{idx:02d}_seed{seed}_*.png"))
            if cands:
                bl = np.asarray(Image.open(cands[0]).convert("RGB")).astype(np.float32) / 255.0
                up = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
                mse = float(np.mean((bl - up) ** 2))
                mae = float(np.mean(np.abs(bl - up)))
                metrics.append({"index": idx, "mse": mse, "mae": mae})
                line += f"  mse={mse:.6f} mae={mae:.6f}"
        print(line)

    peak_generation = measure_peak_vram("during/after generation")

    summary = {
        "mode": args.mode,
        "base_model": base_model,
        "n_w4_replaced": n_w4,
        "n_w8_replaced": n_w8,
        "peak_vram_mb": {
            "after_unet_load": peak_after_unet,
            "after_pipe_build": peak_after_pipe,
            "during_generation": peak_generation,
        },
        "metrics": metrics,
    }
    if metrics:
        summary["mse_mean"] = float(np.mean([m["mse"] for m in metrics]))
        summary["mae_mean"] = float(np.mean([m["mae"] for m in metrics]))
        print(f"\nVs baseline: MSE mean={summary['mse_mean']:.6f}  MAE mean={summary['mae_mean']:.6f}")
    print(f"\nPeak VRAM during generation: {peak_generation:.0f} MB")

    (args.output_dir / "stage3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
