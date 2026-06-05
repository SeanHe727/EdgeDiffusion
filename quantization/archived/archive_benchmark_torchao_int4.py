#!/usr/bin/env python3
"""Quick benchmark: FP16 baseline vs torchao Int4WeightOnly on SDXL-Lightning UNet.

Two modes, run in separate processes to isolate VRAM/state:
  --mode fp16          : load fp16 UNet, time 4-step 1024x1024 generation
  --mode torchao_int4  : load fp16 UNet, apply torchao Int4WeightOnly, time same

Skip critical layers (conv_in/conv_out/time_embedding/etc.) to match our
recipe — keeps these in fp16 even under torchao quantize_().

Reports per-call latency (median over N iters) and peak VRAM.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
SKIP_PATTERNS = ("time_embedding", "time_emb_proj", "add_embedding", "conv_in", "conv_out")


def load_fp16_unet(steps: int, device: str, dtype: torch.dtype):
    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(device, dtype)
    ckpt = hf_hub_download(LIGHTNING_REPO, f"sdxl_lightning_{steps}step_unet.safetensors")
    state = load_file(ckpt, device=device)
    unet.load_state_dict(state, strict=False)
    del state
    torch.cuda.empty_cache()
    return unet


def apply_torchao_int4(unet: nn.Module) -> int:
    """Apply torchao Int4WeightOnlyConfig to all Linear in UNet except SKIP_PATTERNS.
    Returns number of layers quantized."""
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    # Build a per-module filter: True = quantize this module.
    n_target = 0
    target_names: set[str] = set()
    for name, mod in unet.named_modules():
        if isinstance(mod, nn.Linear) and not any(p in name for p in SKIP_PATTERNS):
            target_names.add(name)
            n_target += 1

    def filter_fn(mod: nn.Module, fqn: str) -> bool:
        return fqn in target_names

    # Try multiple configs — default backend on RTX 5070 needs `mslk` which
    # isn't always installed. Fall back to the legacy version=1 path.
    last_err = None
    for kwargs in (
        {"group_size": 128, "version": 1},      # legacy tinygemm/tensor_core_tiled path
        {"group_size": 128},                     # default (mslk)
    ):
        try:
            cfg = Int4WeightOnlyConfig(**kwargs)
            print(f"  applying torchao Int4WeightOnlyConfig({kwargs}) to {n_target} Linear layers...")
            quantize_(unet, cfg, filter_fn=filter_fn)
            return n_target
        except Exception as e:
            print(f"  config {kwargs} failed: {type(e).__name__}: {str(e)[:140]}")
            last_err = e
    raise RuntimeError(f"All torchao Int4 configs failed; last error: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("fp16", "torchao_int4", "torchao_int8", "bnb_nf4", "bnb_int8"), required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--prompts", type=str, nargs="+", default=[
        "A large brown horse walking through a lush green forest.",
        "A plate full of food is ready to be eaten.",
    ])
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=12345)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(args.repo_root / ".tmp"))

    device = "cuda"
    dtype = torch.float16

    torch.cuda.reset_peak_memory_stats()
    print(f"[{args.mode}] Loading FP16 UNet ({args.steps}-step)...")
    unet = load_fp16_unet(args.steps, device, dtype)
    n_quant = 0
    if args.mode == "torchao_int4":
        n_quant = apply_torchao_int4(unet)
    elif args.mode == "torchao_int8":
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        target = {n for n, m in unet.named_modules()
                  if isinstance(m, nn.Linear) and not any(p in n for p in SKIP_PATTERNS)}
        print(f"  applying torchao Int8WeightOnlyConfig to {len(target)} Linear layers...")
        quantize_(unet, Int8WeightOnlyConfig(), filter_fn=lambda m, n: n in target)
        n_quant = len(target)
    elif args.mode in ("bnb_nf4", "bnb_int8"):
        import bitsandbytes as bnb
        import bitsandbytes.nn as bnn
        quant_kind = "nf4" if args.mode == "bnb_nf4" else "int8"
        targets: list[tuple[nn.Module, str, nn.Linear]] = []
        for parent_name, parent_mod in unet.named_modules():
            for attr_name, child in list(parent_mod.named_children()):
                fqn = f"{parent_name}.{attr_name}" if parent_name else attr_name
                if isinstance(child, nn.Linear) and not any(p in fqn for p in SKIP_PATTERNS):
                    targets.append((parent_mod, attr_name, child))
        print(f"  replacing {len(targets)} Linears with bnb {quant_kind}...")
        for parent_mod, attr_name, child in targets:
            if quant_kind == "nf4":
                new = bnn.Linear4bit(child.in_features, child.out_features,
                                     bias=child.bias is not None,
                                     compute_dtype=torch.float16, quant_type="nf4",
                                     quant_storage=torch.uint8)
                # quantize on assignment by sending fp16 weight to GPU through Params4bit
                w = child.weight.data.to(device, dtype)
                new.weight = bnn.Params4bit(w, requires_grad=False, quant_type="nf4")
                new = new.to(device)
            else:  # int8
                new = bnn.Linear8bitLt(child.in_features, child.out_features,
                                       bias=child.bias is not None, has_fp16_weights=False,
                                       threshold=6.0)
                new.weight = bnb.nn.Int8Params(child.weight.data.to(device, dtype),
                                                requires_grad=False, has_fp16_weights=False)
                new = new.to(device)
            if child.bias is not None:
                new.bias = nn.Parameter(child.bias.data.to(device, dtype))
            setattr(parent_mod, attr_name, new)
        n_quant = len(targets)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_load_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    after_load_alloc = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  after unet load: peak={after_load_peak:.0f}MB, current={after_load_alloc:.0f}MB")

    print(f"Building SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, unet=unet, torch_dtype=dtype,
        variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def one_call(i: int) -> float:
        prompt = args.prompts[i % len(args.prompts)]
        gen = torch.Generator(device=device).manual_seed(args.seed_base + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                     height=args.height, width=args.width, generator=gen).images[0]
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f"Warmup {args.warmup} iters...")
    for i in range(args.warmup):
        t = one_call(i)
        print(f"  warmup [{i+1}/{args.warmup}]: {t*1000:.0f} ms")

    print(f"Timing {args.iters} iters ({args.width}x{args.height}, {args.steps} steps)...")
    times = []
    for i in range(args.iters):
        t = one_call(args.warmup + i)
        times.append(t * 1000)
        print(f"  iter [{i+1}/{args.iters}]: {t*1000:.0f} ms")
    gen_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\n=== {args.mode} RESULTS ===")
    print(f"  Quantized Linear layers:  {n_quant}")
    print(f"  After UNet load peak:     {after_load_peak:.0f} MB")
    print(f"  Generation peak VRAM:     {gen_peak:.0f} MB")
    print(f"  Latency: min={min(times):.0f}ms  median={statistics.median(times):.0f}ms  "
          f"max={max(times):.0f}ms  mean={statistics.mean(times):.0f}ms")


if __name__ == "__main__":
    main()
