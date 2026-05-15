#!/usr/bin/env python3
"""
Latency benchmark: Baseline vs Pruned vs Pruned+ToMe.

Measures per-image inference latency (ms) and throughput (img/s) across
three configurations using identical prompts and seeds.

Requirements:
    pip install tomesd

Usage:
    python prune/benchmark_latency.py
    python prune/benchmark_latency.py --pruned-model models/distill_final.safetensors
    python prune/benchmark_latency.py --warmup 5 --runs 20
"""

import os
import sys
import time
import argparse
import torch
from diffusers import StableDiffusionPipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPTS = [
    "a beautiful landscape with mountains and lake, highly detailed, 4k",
    "a cute cat sitting on a windowsill, digital art, warm lighting",
    "a cyberpunk city skyline at night, neon lights, futuristic",
    "a portrait of a woman with flowers in her hair, soft lighting",
]
SEEDS = [42, 123, 2024, 7]


# ── Benchmark runner ──────────────────────────────────────────────────────────

def benchmark_pipe(pipe, prompts, seeds, steps, guidance, warmup_runs, bench_runs, device, label):
    """Run warmup then timed inference. Returns dict with latency stats (ms)."""
    print(f"\n[{label}] warming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        gen = torch.Generator(device=device).manual_seed(seeds[i % len(seeds)])
        with torch.no_grad():
            pipe(prompt=prompts[i % len(prompts)],
                 num_inference_steps=steps, guidance_scale=guidance,
                 height=512, width=512, generator=gen)

    if device == 'cuda':
        torch.cuda.synchronize()

    print(f"[{label}] benchmarking ({bench_runs} runs)...")
    latencies = []
    for i in range(bench_runs):
        prompt = prompts[i % len(prompts)]
        seed   = seeds[i % len(seeds)]
        gen    = torch.Generator(device=device).manual_seed(seed)

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            pipe(prompt=prompt,
                 num_inference_steps=steps, guidance_scale=guidance,
                 height=512, width=512, generator=gen)

        if device == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    avg  = sum(latencies) / len(latencies)
    mn   = min(latencies)
    mx   = max(latencies)
    # simple std
    std  = (sum((x - avg) ** 2 for x in latencies) / len(latencies)) ** 0.5
    return {"label": label, "avg_ms": avg, "min_ms": mn, "max_ms": mx,
            "std_ms": std, "fps": 1000 / avg}


def print_results(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Model':<35} {'Avg(ms)':>9} {'Min(ms)':>9} {'Std(ms)':>9} {'FPS':>7}")
    print("-" * 70)
    baseline_avg = results[0]["avg_ms"]
    for r in results:
        speedup = f"  ({baseline_avg / r['avg_ms']:.2f}x)" if r != results[0] else ""
        print(f"{r['label']:<35} {r['avg_ms']:>9.1f} {r['min_ms']:>9.1f} "
              f"{r['std_ms']:>9.1f} {r['fps']:>7.2f}{speedup}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from prune import sp_core

    parser = argparse.ArgumentParser(description="Latency benchmark: Baseline vs Pruned vs Pruned+ToMe")
    parser.add_argument('--base-model',    default='models/sd-turbo')
    parser.add_argument('--sd15-model',    default='models/sd-1-5',
                        help="Path to SD-1.5 model dir (benchmarked at 30 steps, guidance=7.5)")
    parser.add_argument('--pruned-model',  default='models/distill_final.safetensors')
    parser.add_argument('--device',        default=None)
    parser.add_argument('--steps',         type=int,   default=4)
    parser.add_argument('--guidance',      type=float, default=0.0)
    parser.add_argument('--warmup',        type=int,   default=3,  help="Warmup runs before timing")
    parser.add_argument('--runs',          type=int,   default=10, help="Timed runs per config")
    parser.add_argument('--tome-ratio',    type=float, default=0.5,
                        help="ToMe merge ratio (0.0-1.0). 0.5 = merge 50%% of tokens.")
    parser.add_argument('--skip-baseline', action='store_true',
                        help="Skip baseline to save time (baseline is slow to load twice)")
    parser.add_argument('--skip-sd15',     action='store_true',
                        help="Skip SD-1.5 benchmark")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float16 if device == 'cuda' else torch.float32
    print(f"Device: {device} | dtype: {dtype} | steps: {args.steps} | runs: {args.runs}")

    base_model_source  = sp_core.resolve_base_model_source(args.base_model)
    pruned_config_path = args.pruned_model.replace('.safetensors', '.config.json')

    results = []

    # ── 0. SD-1.5 (30 steps, guidance=7.5) ───────────────────────────────────
    if not args.skip_sd15 and os.path.exists(args.sd15_model):
        print("\nLoading SD-1.5 (30 steps, guidance=7.5)...")
        pipe_sd15 = StableDiffusionPipeline.from_pretrained(
            args.sd15_model, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        ).to(device)
        pipe_sd15.set_progress_bar_config(disable=True)
        params_sd15 = sum(p.numel() for p in pipe_sd15.unet.parameters()) / 1e6
        print(f"SD-1.5 UNet: {params_sd15:.1f}M params")
        results.append(benchmark_pipe(
            pipe_sd15, PROMPTS, SEEDS, 20, 7.5,
            args.warmup, args.runs, device,
            label=f"SD-1.5 30-step ({params_sd15:.0f}M params)",
        ))
        del pipe_sd15
        if device == 'cuda': torch.cuda.empty_cache()

    # ── 1. Baseline SD-Turbo ──────────────────────────────────────────────────
    if not args.skip_baseline:
        print("\nLoading baseline SD-Turbo (4 steps)...")
        pipe_base = StableDiffusionPipeline.from_pretrained(
            base_model_source, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        ).to(device)
        pipe_base.set_progress_bar_config(disable=True)

        params_base = sum(p.numel() for p in pipe_base.unet.parameters()) / 1e6
        print(f"Baseline UNet: {params_base:.1f}M params")

        results.append(benchmark_pipe(
            pipe_base, PROMPTS, SEEDS, args.steps, args.guidance,
            args.warmup, args.runs, device,
            label=f"SD-Turbo {args.steps}-step ({params_base:.0f}M params)",
        ))
        # 2-step variant
        results.append(benchmark_pipe(
            pipe_base, PROMPTS, SEEDS, 2, args.guidance,
            args.warmup, args.runs, device,
            label=f"SD-Turbo 2-step ({params_base:.0f}M params)",
        ))
        del pipe_base
        if device == 'cuda': torch.cuda.empty_cache()

    # ── 2. Pruned model ───────────────────────────────────────────────────────
    print("\nLoading pruned+distilled UNet...")
    from prune.pruned_rebuild import create_unet_from_safetensors
    pruned_unet = create_unet_from_safetensors(args.pruned_model, pruned_config_path)
    params_pruned = sum(p.numel() for p in pruned_unet.parameters()) / 1e6
    print(f"Pruned UNet: {params_pruned:.1f}M params")

    pruned_unet = pruned_unet.to(device=device, dtype=dtype).eval()
    pipe_pruned = StableDiffusionPipeline.from_pretrained(
        base_model_source, unet=pruned_unet, torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False, local_files_only=True,
    ).to(device)
    pipe_pruned.set_progress_bar_config(disable=True)

    results.append(benchmark_pipe(
        pipe_pruned, PROMPTS, SEEDS, args.steps, args.guidance,
        args.warmup, args.runs, device,
        label=f"Pruned {args.steps}-step ({params_pruned:.0f}M params)",
    ))
    # 2-step variant
    results.append(benchmark_pipe(
        pipe_pruned, PROMPTS, SEEDS, 2, args.guidance,
        args.warmup, args.runs, device,
        label=f"Pruned 2-step ({params_pruned:.0f}M params)",
    ))

    # ── 3. Pruned + ToMe ──────────────────────────────────────────────────────
    print(f"\nApplying Token Merging (ratio={args.tome_ratio})...")
    try:
        import tomesd
        # Apply ToMe patch — merges similar tokens before each attention layer
        # ratio=0.5 reduces token count by ~50%, directly cutting attention compute
        tomesd.apply_patch(pipe_pruned, ratio=args.tome_ratio)
        print(f"ToMe applied successfully.")

        results.append(benchmark_pipe(
            pipe_pruned, PROMPTS, SEEDS, args.steps, args.guidance,
            args.warmup, args.runs, device,
            label=f"Pruned+ToMe (ratio={args.tome_ratio})",
        ))

        # Also try a lower ratio for quality/speed tradeoff comparison
        if args.tome_ratio != 0.3:
            tomesd.remove_patch(pipe_pruned)
            tomesd.apply_patch(pipe_pruned, ratio=0.3)
            results.append(benchmark_pipe(
                pipe_pruned, PROMPTS, SEEDS, args.steps, args.guidance,
                args.warmup, args.runs, device,
                label=f"Pruned+ToMe (ratio=0.3)",
            ))

    except ImportError:
        print("WARNING: tomesd not installed. Skipping ToMe benchmark.")
        print("         Install with: pip install tomesd")

    # ── Results ───────────────────────────────────────────────────────────────
    print_results(results)

    # Print memory usage if on CUDA
    if device == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU memory: {mem_mb:.0f} MB")


if __name__ == '__main__':
    main()
