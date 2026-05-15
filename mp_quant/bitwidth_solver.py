#!/usr/bin/env python3
"""
Stage 2: Bit-width assignment solver.

Takes the per-layer sensitivity profile produced by Stage 1 and assigns
each layer one of {fp16, int8, int4} such that:
  - Total compressed UNet size is approximately equal to a target
  - Total quality loss (sum of sensitivity scores at assigned bit-widths) is minimised

Algorithm
─────────
Two views, one solver:

  - For each layer there are 3 (cost, quality_loss) candidates:
        fp16:  cost = 1.0 * params,  loss = 0
        int8:  cost = 0.5 * params,  loss = sensitivity[int8]
        int4:  cost = 0.25 * params, loss = sensitivity[int4]

  - We want to minimise total quality_loss subject to
        total_cost <= (1 - target_size_reduction) * total_params

  - Greedy threshold-based assignment:
        Binary-search a sensitivity threshold T.
        At threshold T, each layer is assigned the LOWEST-BIT option whose
        sensitivity is below T:
            sensitivity[int4] < T          → int4
            sensitivity[int4] >= T > sens[int8] → int8
            else                            → fp16
        Larger T → more aggressive quantization → smaller model.
        Binary-search T until total size matches the target.

  - This is a Pareto-greedy strategy:
        At any threshold T, every layer with quality_loss < T gets the
        compressed option. Layers with higher loss stay fp16. Equivalent to
        sorting by (loss / size_saved) and picking from the top.

Layers with sensitivity[bit] = None (e.g. quantization failed for that
combination) are forced to fp16 for that bit-width — they can never be
assigned that quantized option.

Output: a JSON config that Stage 3 (apply_quant.py) reads layer-by-layer.

Run:
  python mp_quant/bitwidth_solver.py                 # uses defaults from config
  python mp_quant/bitwidth_solver.py --target 0.5    # target 50% size reduction
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "mp_quant_config.yaml")


def _load_yaml(path):
    if not os.path.exists(path):
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Relative size cost per parameter, by bit-width.
# fp16 = 2 bytes, int8 = 1 byte, int4 = 0.5 bytes.
# We work in fp16-relative units so 1.0 = fp16 size.
BIT_COST = {
    "fp16": 1.0,
    "int8": 0.5,
    "int4": 0.25,
}


def assign_for_threshold(layers, threshold, bits_priority):
    """For each layer, pick the smallest-cost option with sensitivity < threshold.

    bits_priority is an iterable from most aggressive to most conservative,
    e.g. ["int4", "int8", "fp16"]. fp16 is always allowed (sensitivity 0).
    """
    assignment = {}
    total_cost = 0.0
    total_loss = 0.0
    for fqn, info in layers.items():
        chosen = "fp16"
        chosen_loss = 0.0
        for bit in bits_priority:
            if bit == "fp16":
                break  # fp16 fallback already chosen
            s = info.get(bit)
            if s is None:
                continue  # this bit-width unavailable for this layer
            if s < threshold:
                chosen = bit
                chosen_loss = s
                break
        assignment[fqn] = chosen
        total_cost += info["param_count"] * BIT_COST[chosen]
        total_loss += chosen_loss
    return assignment, total_cost, total_loss


def solve(layers, target_size_reduction, bits_priority=None,
          tol=0.005, max_iter=60):
    """Binary-search the sensitivity threshold to hit the target size reduction.

    Args:
        layers: dict[fqn -> {param_count, int8, int4, type}]
        target_size_reduction: 0.4 → 40% smaller than fp16 baseline
        bits_priority: order to try bit-widths per layer. Default favours int4
                       (most aggressive) over int8 over fp16.
        tol: relative size tolerance for stopping the binary search
        max_iter: safety cap on binary search iterations
    """
    if bits_priority is None:
        bits_priority = ["int4", "int8", "fp16"]

    total_params = sum(info["param_count"] for info in layers.values())
    fp16_total_cost = total_params * BIT_COST["fp16"]
    target_cost = fp16_total_cost * (1.0 - target_size_reduction)

    # Threshold bounds: 0 (no quantization) → max sensitivity * 2 (max compression)
    all_sens = [s for info in layers.values()
                for b in ("int4", "int8")
                for s in [info.get(b)] if s is not None]
    if not all_sens:
        raise ValueError("No usable sensitivity values found in input.")

    lo, hi = 0.0, max(all_sens) * 2.0

    # Sanity: max threshold should hit highest compression possible
    _, max_compressed_cost, _ = assign_for_threshold(layers, hi, bits_priority)
    if max_compressed_cost > target_cost:
        # Target is unrealistic — can't compress enough even at max threshold.
        # Return the best we can do.
        print(f"  WARNING: target {target_size_reduction:.1%} unreachable. "
              f"Max compression possible: "
              f"{1 - max_compressed_cost/fp16_total_cost:.1%}")
        return assign_for_threshold(layers, hi, bits_priority)

    # Binary search
    best = None
    for it in range(max_iter):
        mid = (lo + hi) / 2
        assignment, cost, loss = assign_for_threshold(layers, mid, bits_priority)
        ratio = cost / fp16_total_cost
        achieved_reduction = 1.0 - ratio

        if abs(achieved_reduction - target_size_reduction) < tol:
            best = (assignment, cost, loss, mid)
            break
        if cost > target_cost:
            lo = mid     # need more compression → higher threshold
        else:
            hi = mid     # over-compressed → lower threshold

        best = (assignment, cost, loss, mid)

    return best


def summarize(assignment, layers):
    """Per-bit-width breakdown of assignment."""
    by_bit = {"fp16": 0, "int8": 0, "int4": 0}
    params_by_bit = {"fp16": 0, "int8": 0, "int4": 0}
    for fqn, bit in assignment.items():
        by_bit[bit] += 1
        params_by_bit[bit] += layers[fqn]["param_count"]
    return by_bit, params_by_bit


def main():
    parser = argparse.ArgumentParser(
        description="Solve per-layer bit-width assignment from sensitivity profile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default=DEFAULT_CONFIG)
    parser.add_argument("--sensitivity", default=None,
                        help="Path to sensitivity.json (default: results_dir/sensitivity.json)")
    parser.add_argument("--target",      type=float, default=None,
                        help="Target fractional size reduction (e.g. 0.4 = 40%% smaller). "
                             "Default: from config.")
    parser.add_argument("--output",      default=None,
                        help="Output bitwidth_config.json path")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    results_dir = cfg.get("results_dir", "mp_quant/results")
    sens_path   = args.sensitivity or os.path.join(results_dir, "sensitivity.json")
    target      = args.target if args.target is not None else cfg.get("target_size_reduction", 0.40)
    out_path    = args.output  or os.path.join(results_dir, "bitwidth_config.json")

    print(f"Sensitivity:  {sens_path}")
    print(f"Target size reduction: {target:.0%}")

    with open(sens_path, encoding="utf-8") as f:
        report = json.load(f)
    layers = report["layers"]
    print(f"Loaded {len(layers)} layers")

    # Optional: filter to only Linear or only Conv2d for analysis
    # (e.g. if you want to apply different solvers per layer type)

    result = solve(layers, target_size_reduction=target)
    if result is None:
        print("ERROR: solver failed")
        sys.exit(1)
    assignment, cost, loss, threshold = result

    total_params = sum(info["param_count"] for info in layers.values())
    achieved_reduction = 1.0 - cost / (total_params * BIT_COST["fp16"])

    by_bit, params_by_bit = summarize(assignment, layers)

    print(f"\n──── Solution ────")
    print(f"Threshold (sensitivity cutoff): {threshold:.5f}")
    print(f"Target reduction:               {target:.1%}")
    print(f"Achieved reduction:             {achieved_reduction:.1%}")
    print(f"Total sensitivity loss:         {loss:.4f}")
    print(f"\nPer-bit assignment:")
    print(f"  {'bit':<6} {'layers':>8} {'params':>10} {'% of total':>12}")
    for bit in ("int4", "int8", "fp16"):
        pct = 100 * params_by_bit[bit] / max(1, total_params)
        print(f"  {bit:<6} {by_bit[bit]:>8} {params_by_bit[bit]/1e6:>9.2f}M {pct:>11.1f}%")

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "target_size_reduction": target,
            "achieved_reduction":    achieved_reduction,
            "threshold":             threshold,
            "total_sensitivity_loss": loss,
            "by_bit": {bit: {"layers": by_bit[bit],
                             "params": params_by_bit[bit]} for bit in by_bit},
            "assignment":            assignment,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
