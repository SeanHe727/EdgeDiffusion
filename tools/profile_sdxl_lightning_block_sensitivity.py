#!/usr/bin/env python3
"""Block-level Linear fake-quant sensitivity profiling for SDXL-Lightning."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.profile_sdxl_lightning_sensitivity import (
    BASE_MODEL,
    LIGHTNING_REPO,
    SKIP_PATTERNS,
    atomic_save_json,
    build_pipe,
    build_samples,
    choose_prompts,
    divergence,
    fake_quant_2d_,
    find_linear_layers,
    load_lightning_unet,
    run_unet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--bits", type=str, default="8,4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "float32"))
    parser.add_argument("--selection-seed", type=int, default=20260517)
    parser.add_argument(
        "--blocks",
        type=str,
        default="",
        help="Comma-separated block names. Empty means all blocks with Linear layers.",
    )
    return parser.parse_args()


def block_name(layer_name: str) -> str:
    match = re.match(r"^(down_blocks\.\d+|up_blocks\.\d+|mid_block)(?:\.|$)", layer_name)
    if match:
        return match.group(1)
    return layer_name.split(".", 1)[0]


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("HF_HOME", str(repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(repo_root / ".tmp"))

    out_path = args.output if args.output.is_absolute() else repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bits = [int(bit.strip()) for bit in args.bits.split(",") if bit.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    prompts = choose_prompts(repo_root / "dataset", args.prompt_count, args.selection_seed)
    print(
        f"SDXL-Lightning block sensitivity: Linear grouped by block, bits={bits}, "
        f"prompts={len(prompts)}, samples={len(prompts) * args.steps}, "
        f"size={args.width}x{args.height}, dtype={dtype}, device={device}"
    )
    print(f"Output: {out_path}")

    unet = load_lightning_unet(args.steps, dtype, device)
    pipe = build_pipe(unet, dtype, device)

    print("Building calibration samples...")
    samples = build_samples(pipe, prompts, args, dtype, device)
    print(f"Built {len(samples)} samples")
    print("Computing baseline UNet outputs...")
    baseline_outs = run_unet(pipe.unet, samples)

    layers = find_linear_layers(pipe.unet)
    by_block: dict[str, list[tuple[str, object]]] = {}
    for name, mod in layers:
        by_block.setdefault(block_name(name), []).append((name, mod))

    requested_blocks = split_csv(args.blocks)
    if requested_blocks:
        by_block = {name: by_block[name] for name in requested_blocks if name in by_block}
    ordered_blocks = sorted(by_block, key=lambda x: (0 if x.startswith("down") else 1 if x == "mid_block" else 2, x))
    print("Blocks:")
    for name in ordered_blocks:
        params = sum(mod.weight.numel() for _, mod in by_block[name])
        print(f"  {name}: {len(by_block[name])} Linear layers, {params/1e6:.2f}M params")

    report = {
        "metadata": {
            "base_model": BASE_MODEL,
            "lightning_repo": LIGHTNING_REPO,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "prompt_count": len(prompts),
            "sample_count": len(samples),
            "bits": bits,
            "group_size": args.group_size,
            "dtype": str(dtype),
            "device": device,
            "skip_patterns": list(SKIP_PATTERNS),
            "prompts": [p.__dict__ for p in prompts],
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "blocks": {},
    }

    for block_idx, block in enumerate(ordered_blocks, start=1):
        block_layers = by_block[block]
        entry = report["blocks"].setdefault(
            block,
            {
                "linear_layers": len(block_layers),
                "param_count": sum(mod.weight.numel() for _, mod in block_layers),
                "layers": [name for name, _ in block_layers],
            },
        )
        originals = [(mod, mod.weight.detach().clone()) for _, mod in block_layers]
        for bit in bits:
            key = f"int{bit}"
            t0 = time.time()
            try:
                weight_mses = []
                with torch.no_grad():
                    for _, mod in block_layers:
                        weight_mses.append(fake_quant_2d_(mod.weight, bit, args.group_size)["weight_mse"])
                quant_outs = run_unet(pipe.unet, samples)
                score = divergence(quant_outs, baseline_outs)
                entry[key] = score["one_minus_cosine"]
                entry[f"{key}_detail"] = {
                    **score,
                    "mean_weight_mse": float(sum(weight_mses) / max(1, len(weight_mses))),
                    "max_weight_mse": float(max(weight_mses) if weight_mses else 0.0),
                }
                print(
                    f"[{block_idx}/{len(ordered_blocks)}] {block} {key}: "
                    f"1-cos={score['one_minus_cosine']:.6e}, rel_l2={score['relative_l2']:.6e}, "
                    f"layers={len(block_layers)}, params={entry['param_count']/1e6:.2f}M, {time.time()-t0:.1f}s"
                )
            except Exception as exc:
                entry[key] = None
                entry[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{block_idx}/{len(ordered_blocks)}] {block} {key}: ERROR {entry[f'{key}_error']}")
            finally:
                with torch.no_grad():
                    for mod, weight in originals:
                        mod.weight.copy_(weight)
                if "quant_outs" in locals():
                    del quant_outs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                atomic_save_json(out_path, report)

    report["metadata"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_save_json(out_path, report)
    print(f"DONE: {out_path}")


if __name__ == "__main__":
    main()
