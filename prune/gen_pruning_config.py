#!/usr/bin/env python3
"""
Automatic sensitivity-threshold pruning config generator.

Supports two sensitivity metrics:
  - "combined" (legacy): MSE+LPIPS+Taylor+CLIP from sensitivity_report_v2.json
  - "ld" (new): Latent Divergence from sensitivity_ld_report.json

For iterative pruning, use --already-pruned to specify how much has already
been pruned, and --target for the new total target.

Usage:
  python gen_auto_pruning_config.py [--target 0.07] [--max-ratio 0.60]
  python gen_auto_pruning_config.py --metric ld --target 0.14 --already-pruned 0.07
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRUNE_DIR = os.path.dirname(os.path.abspath(__file__))

# Will be loaded based on --metric flag
results = None
METRIC_KEY = "combined"  # or "ld_score"

# Parameter counts per component (M)
PARAM_COUNTS = {
    "down_blocks.0": {"resnet": 7.0, "self_attn": 0.8, "cross_attn": 0.8, "ffn": 2.6},
    "down_blocks.1": {"resnet": 27.8, "self_attn": 3.1, "cross_attn": 3.1, "ffn": 10.5},
    "down_blocks.2": {"resnet": 55.6, "self_attn": 13.1, "cross_attn": 13.1, "ffn": 41.9},
    "down_blocks.3": {"resnet": 62.3},
    "mid_block":     {"resnet": 62.3, "self_attn": 13.1, "cross_attn": 13.1, "ffn": 41.9},
    "up_blocks.0":   {"resnet": 147.5},
    "up_blocks.1":   {"resnet": 139.3, "self_attn": 13.1, "cross_attn": 13.1, "ffn": 41.9},
    "up_blocks.2":   {"resnet": 55.6, "self_attn": 3.1, "cross_attn": 3.1, "ffn": 10.5},
    "up_blocks.3":   {"resnet": 7.0, "self_attn": 1.2, "cross_attn": 1.2, "ffn": 3.9},
}

TOTAL_PARAMS = 858.49  # M

# Protected blocks — never prune resnet/FFN/conv (attention handled separately via ATTN_HARDCODED)
# Emptied: SA scores now decide pruning for all blocks including down_blocks.0 and up_blocks.3
PROTECTED_BLOCKS = set()

# Attention: per-head pruning, discrete. Map ratio -> actual heads removed.
# 8 heads total. ratio > 1/8 (0.125) removes 1 head, > 2/8 removes 2, etc.
ATTN_HEAD_COUNT = 8
ATTN_HEAD_DIM = {
    "down_blocks.1": 80,   # 640 / 8
    "down_blocks.2": 160,  # 1280 / 8
    "mid_block": 160,
    "up_blocks.1": 160,
    "up_blocks.2": 80,
}

# Attention is hardcoded: specify max heads to remove per block
# Based on sensitivity analysis — attention is discrete, can't use threshold curve
ATTN_HARDCODED = {}  # R4+: no attention pruning — all blocks already pruned 3 rounds


def compute_param_counts_from_model(model_path, config_path=None):
    """Dynamically compute per-block per-component param counts from a model."""
    from prune.pruned_rebuild import create_unet_from_safetensors
    if not config_path:
        config_path = model_path.replace('.safetensors', '.config.json')
    unet = create_unet_from_safetensors(model_path, config_path)

    blocks = [
        "down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
        "mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3",
    ]

    def classify(name):
        parts = name.lower()
        if "ff.net" in parts or "ff." in parts:
            return "ffn"
        if "attn2" in parts:
            return "cross_attn"
        if "attn1" in parts:
            return "self_attn"
        if "resnets" in parts:
            return "resnet"
        return "other"

    def get_block(name):
        for b in blocks:
            if name.startswith(b):
                return b
        return "other"

    counts = {}
    for name, module in unet.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        block = get_block(name)
        if block == "other":
            continue
        comp = classify(name)
        if comp == "other":
            continue
        n = sum(p.numel() for p in module.parameters())
        if block not in counts:
            counts[block] = {}
        counts[block][comp] = counts[block].get(comp, 0) + n

    # Convert to millions
    for block in counts:
        for comp in counts[block]:
            counts[block][comp] = round(counts[block][comp] / 1e6, 1)

    total = sum(p.numel() for p in unet.parameters()) / 1e6

    # Detect remaining attention heads per block
    attn_heads = {}
    for name, module in unet.named_modules():
        if hasattr(module, 'heads') and hasattr(module, 'to_q'):
            block = get_block(name)
            if block != "other":
                attn_heads[block] = module.heads

    del unet
    return counts, round(total, 2), attn_heads


def load_sensitivity_data(metric, report_path=None):
    global results, METRIC_KEY
    if metric == "ld":
        path = report_path or os.path.join(PRUNE_DIR, "sensitivity_ld_report.json")
        METRIC_KEY = "ld_score"
    else:
        path = report_path or os.path.join(PRUNE_DIR, "sensitivity_report_v2.json")
        METRIC_KEY = "combined"

    with open(path) as f:
        sens_data = json.load(f)
    results = sens_data["results"]

    print(f"  Loaded sensitivity data from {path} (metric: {METRIC_KEY})")


def get_sensitivity_curve(block, comp):
    """Return (ratios, scores) arrays for a component's sensitivity curve."""
    if block not in results or comp not in results[block]:
        return None, None
    data = results[block][comp]
    ratios = []
    scores = []
    for r_str in sorted(data.keys(), key=lambda x: float(x)):
        ratios.append(float(r_str))
        scores.append(data[r_str][METRIC_KEY])
    return np.array(ratios), np.array(scores)


def find_ratio_at_threshold(ratios, scores, threshold, max_ratio=0.60):
    """Find the pruning ratio where sensitivity = threshold (linear interpolation).

    Returns the max ratio where score <= threshold, capped at max_ratio.
    Supports negative thresholds (Z-normalized scores).
    """
    # If all scores exceed threshold (even at ratio=0), can't prune at all
    if len(scores) > 1 and scores[1] > threshold:
        # scores[0] is always 0 (ratio=0.0). Check if first non-zero ratio exceeds
        return 0.0

    # Find where the curve crosses the threshold
    for i in range(len(scores) - 1):
        if scores[i] <= threshold < scores[i + 1]:
            # Linear interpolation between points i and i+1
            frac = (threshold - scores[i]) / (scores[i + 1] - scores[i])
            ratio = ratios[i] + frac * (ratios[i + 1] - ratios[i])
            return min(ratio, max_ratio)

    # If all scores are below threshold, return max measured ratio (capped)
    if scores[-1] <= threshold:
        return min(ratios[-1], max_ratio)

    # Fallback
    return 0.0


def compute_total_removal(threshold, max_ratio=0.60, snap_to=0.05):
    """Compute total parameter removal for a given sensitivity threshold."""
    total_removed = 0.0
    per_block = {}

    for block, comps in PARAM_COUNTS.items():
        block_ratios = {}
        for comp, params in comps.items():
            if comp in ("self_attn", "cross_attn"):
                # Attention: use hardcoded heads (allowed even in protected blocks)
                heads_to_remove = ATTN_HARDCODED.get(block, {}).get(comp, 0)
                if heads_to_remove > 0:
                    ratio = heads_to_remove / ATTN_HEAD_COUNT
                    # Use 3 decimal places to preserve 0.125 (1/8) exactly
                    block_ratios[comp] = round(ratio, 3)
                    total_removed += params * ratio
                continue

            # Conv/FFN: skip if block is protected
            if block in PROTECTED_BLOCKS:
                continue

            # ResNet and FFN: use threshold curve
            ratios, scores = get_sensitivity_curve(block, comp)
            if ratios is None:
                continue

            raw_ratio = find_ratio_at_threshold(ratios, scores, threshold, max_ratio)

            # Snap to nearest granularity
            snapped = round(raw_ratio / snap_to) * snap_to
            snapped = max(0.0, min(snapped, max_ratio))

            if snapped > 0:
                block_ratios[comp] = round(snapped, 2)
                total_removed += params * snapped

        if block_ratios:
            per_block[block] = block_ratios

    return total_removed, per_block


def main():
    global PARAM_COUNTS, TOTAL_PARAMS, ATTN_HEAD_COUNT, ATTN_HARDCODED

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=0.20, help="Target param removal fraction (of CURRENT model)")
    parser.add_argument("--max-ratio", type=float, default=0.60, help="Max pruning ratio per component")
    parser.add_argument("--snap", type=float, default=0.05, help="Snap ratios to this granularity")
    parser.add_argument("--metric", choices=["combined", "ld"], default="combined",
                        help="Sensitivity metric: 'combined' (legacy) or 'ld' (latent divergence)")
    parser.add_argument("--sensitivity-report", default=None,
                        help="Path to sensitivity JSON report (overrides default path)")
    parser.add_argument("--model_path", default=None,
                        help="Path to pruned model safetensors (for dynamic param counting)")
    parser.add_argument("--model_config", default=None,
                        help="Path to pruned model config JSON")
    parser.add_argument("--already-pruned", type=float, default=0.0,
                        help="Fraction already pruned (for incremental config generation)")
    args = parser.parse_args()

    # Dynamic param counts from pruned model
    if args.model_path:
        print("Computing param counts from pruned model...")
        dyn_counts, dyn_total, dyn_attn_heads = compute_param_counts_from_model(
            args.model_path, args.model_config)
        PARAM_COUNTS = dyn_counts
        TOTAL_PARAMS = dyn_total
        print(f"  Current model: {TOTAL_PARAMS:.1f}M params")
        for block in sorted(PARAM_COUNTS.keys()):
            parts = [f"{c}={v:.1f}M" for c, v in PARAM_COUNTS[block].items()]
            print(f"    {block}: {', '.join(parts)}")

        # Adjust attention heads based on remaining heads
        if dyn_attn_heads:
            print(f"  Remaining attention heads: {dyn_attn_heads}")
            # Scale down hardcoded heads-to-remove based on remaining
            for block in list(ATTN_HARDCODED.keys()):
                remaining = dyn_attn_heads.get(block, ATTN_HEAD_COUNT)
                if remaining < ATTN_HEAD_COUNT:
                    # Already pruned some heads, reduce further removal
                    for atype in list(ATTN_HARDCODED[block].keys()):
                        orig_remove = ATTN_HARDCODED[block][atype]
                        # Scale: keep same proportion of removal relative to remaining
                        new_remove = max(0, min(orig_remove, remaining - 2))  # keep at least 2 heads
                        ATTN_HARDCODED[block][atype] = new_remove
            print(f"  Adjusted attention pruning: {ATTN_HARDCODED}")

    load_sensitivity_data(args.metric, report_path=args.sensitivity_report)

    target_removal = TOTAL_PARAMS * args.target

    print(f"=== Auto Pruning Config Generator ===")
    print(f"Target: {args.target*100:.0f}% ({target_removal:.1f}M of {TOTAL_PARAMS:.1f}M)")
    print(f"Max ratio per component: {args.max_ratio*100:.0f}%")
    print(f"Snap granularity: {args.snap*100:.0f}%")
    print()

    # Binary search for the right threshold
    # Auto-detect score range for proper search bounds
    all_scores = []
    for block, comps in PARAM_COUNTS.items():
        if block in PROTECTED_BLOCKS:
            continue
        for comp in comps:
            if comp in ("self_attn", "cross_attn"):
                continue
            ratios, scores = get_sensitivity_curve(block, comp)
            if scores is not None and len(scores) > 0:
                all_scores.extend(scores.tolist())
    if all_scores:
        hi = max(all_scores) * 1.1
        lo = min(all_scores) - 0.1 * (max(all_scores) - min(all_scores))
    else:
        hi, lo = 1.0, 0.0
    best_threshold = 0.0
    best_removal = 0.0
    best_per_block = {}

    for _ in range(100):
        mid = (lo + hi) / 2
        removal, per_block = compute_total_removal(mid, args.max_ratio, args.snap)

        if abs(removal - target_removal) < 0.5:  # within 0.5M
            best_threshold = mid
            best_removal = removal
            best_per_block = per_block
            break

        if removal < target_removal:
            lo = mid
        else:
            hi = mid

        best_threshold = mid
        best_removal = removal
        best_per_block = per_block

    print(f"Optimal sensitivity threshold: {best_threshold:.4f}")
    print(f"Total removal: {best_removal:.1f}M ({best_removal/TOTAL_PARAMS*100:.1f}%)")
    print()

    # Display results
    print(f"{'Component':<35} {'Params':>8} {'Ratio':>7} {'Removed':>9} {'Method':>12}")
    print("-" * 75)

    total_check = 0
    all_blocks = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
                  "mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"]

    for block in all_blocks:
        block_config = best_per_block.get(block, {})
        block_params = PARAM_COUNTS.get(block, {})
        for comp in ["resnet", "self_attn", "cross_attn", "ffn"]:
            if comp not in block_params:
                continue
            params = block_params[comp]
            ratio = block_config.get(comp, 0.0)
            removed = params * ratio
            total_check += removed

            if block in PROTECTED_BLOCKS:
                method = "protected"
            elif comp in ("self_attn", "cross_attn"):
                method = "hardcoded"
            else:
                method = f"thresh={best_threshold:.3f}"

            if ratio > 0 or block in PROTECTED_BLOCKS:
                status = f"{ratio:>6.0%}" if ratio > 0 else "   0%"
                print(f"{block+'.'+comp:<35} {params:>7.1f}M {status} {removed:>8.1f}M {method:>12}")

    print("-" * 75)
    print(f"{'TOTAL':<35} {'':>8} {'':>7} {total_check:>8.1f}M  → {total_check/TOTAL_PARAMS*100:.1f}%")
    print(f"Remaining: {TOTAL_PARAMS - total_check:.1f}M params")

    # Show sensitivity curve intersection visualization
    print(f"\n=== Sensitivity Curve Intersections (threshold={best_threshold:.4f}) ===")
    for block in all_blocks:
        if block in PROTECTED_BLOCKS:
            continue
        block_params = PARAM_COUNTS.get(block, {})
        for comp in ["resnet", "ffn"]:
            if comp not in block_params:
                continue
            ratios, scores = get_sensitivity_curve(block, comp)
            if ratios is None:
                continue
            raw = find_ratio_at_threshold(ratios, scores, best_threshold, args.max_ratio)
            snapped = round(raw / args.snap) * args.snap
            label = block.replace('down_blocks.','D').replace('up_blocks.','U').replace('mid_block','Mid')
            print(f"  {label}.{comp:<10}: raw={raw:.3f} -> snapped={snapped:.2f}")

    # Save config
    output_config = {
        "method": "sensitivity_threshold_auto",
        "description": f"{args.target*100:.0f}% pruning. Auto-generated via sensitivity threshold={best_threshold:.4f}. "
                       f"Attention: mid_block only. All blocks SA-driven.",
        "total_params_M": TOTAL_PARAMS,
        "threshold": round(best_threshold, 4),
        "configs": {
            f"{int(args.target*100)}%": {
                "target": args.target,
                "actual": round(best_removal / TOTAL_PARAMS, 4),
                "params_removed_M": round(best_removal, 1),
                "strategy": "sensitivity_threshold_auto",
                "sensitivity_threshold": round(best_threshold, 4),
                "attention_config": "hardcoded_per_head",
                "protected_blocks": list(PROTECTED_BLOCKS),
                "per_block": best_per_block
            }
        }
    }

    pct = int(args.target * 100)
    output_path = os.path.join(PRUNE_DIR, f"pruning_config_{pct}pct_auto.json")
    with open(output_path, "w") as f:
        json.dump(output_config, f, indent=2)
    print(f"\nConfig saved to: {output_path}")


if __name__ == "__main__":
    main()
