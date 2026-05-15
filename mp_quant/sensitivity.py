#!/usr/bin/env python3
"""
Stage 1: Per-layer quantization sensitivity profiling.

For every Linear and Conv2d module in the UNet (minus a hard-coded skip list of
known-sensitive layers like conv_in / conv_out), we quantize THAT LAYER ALONE
to each candidate bit-width (int8, int4) and measure how much the model's output
drifts from the fp16 baseline.

The drift metric is LD-score style: mean + std distance of the UNet output
(epsilon prediction) on calibration samples. Lower score = lower sensitivity =
safer to quantize.

Output:
  mp_quant/results/sensitivity.json
    {
      "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q": {
        "int8": 0.0123,
        "int4": 0.0891,
        "type": "Linear",
        "param_count": 102400,
        ...
      },
      ...
    }

Why this matters for the pipeline:
  - The bit-width solver (Stage 2) reads these per-layer sensitivities and
    decides which layers go to int8, int4, or stay fp16.
  - Without this, you'd be quantizing blindly. With it, you make a principled
    Pareto trade-off (compression vs quality).

Implementation notes:
  - For each test layer we deepcopy the original module, apply torchao with a
    filter_fn that matches ONLY this layer, run forward on calib data, then
    restore the original module via setattr.
  - This avoids rebuilding the UNet between tests (slow) while keeping the
    measurement isolated (we change exactly one layer at a time).
  - INT4 requires bf16; INT8 works in fp16. We run the two passes separately
    so the baseline outputs match the quantization dtype.

Run:
  python mp_quant/sensitivity.py
  python mp_quant/sensitivity.py --max-prompts 8  # quick test
"""
import os
import sys
import gc
import copy
import json
import time
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "mp_quant_config.yaml")


# —— Helpers ──────────────────────────────────────────────────────────────────

def _load_config(path):
    if not os.path.exists(path):
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_unet_source(args, cfg):
    """Get (weights_path, config_path) from CLI/config or HF download."""
    local_w = args.unet_weights or cfg.get("unet_weights")
    local_c = args.unet_config  or cfg.get("unet_config")
    if local_w and local_c:
        return local_w, local_c
    repo     = cfg.get("unet_repo")
    filename = cfg.get("unet_filename", "distill_final.safetensors")
    if not repo:
        raise ValueError("Need unet_weights+unet_config or unet_repo in config")
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download(repo, filename)
    config  = hf_hub_download(repo, filename.replace(".safetensors", ".config.json"))
    return weights, config


def _get_parent_attr(model, fqn):
    """Walk a dotted module path and return (parent_module, leaf_attr_name)."""
    parts = fqn.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def find_target_layers(unet, skip_patterns, include_conv2d=False):
    """Find target modules for sensitivity profiling.

    Args:
        skip_patterns: list of substring patterns to exclude (e.g. "conv_in",
                       "time_embedding"). Layers whose FQN contains any of
                       these substrings are skipped.
        include_conv2d: whether to include Conv2d layers. Default False
                       because torchao's Int8/Int4 weight-only configs
                       only quantize nn.Linear out of the box, and our
                       Stage 3 GPTQ also targets Linear (Conv2d → fp16 in v1).

    Returns list of (fqn, module_type_name, param_count) tuples, sorted by fqn.
    """
    types = (nn.Linear, nn.Conv2d) if include_conv2d else (nn.Linear,)
    targets = []
    for name, mod in unet.named_modules():
        if not isinstance(mod, types):
            continue
        if any(s in name for s in skip_patterns):
            continue
        n_params = sum(p.numel() for p in mod.parameters())
        targets.append((name, type(mod).__name__, n_params))
    return sorted(targets, key=lambda x: x[0])


# —— Calibration data ─────────────────────────────────────────────────────────

def build_calib_samples(pipe, prompts, n_steps, resolution, seed, device, dtype):
    """Generate a fixed batch of (noisy_latent, timestep, enc_hs) tuples.

    We run the teacher's denoising trajectory once to get realistic latents at
    each timestep. Every sensitivity test will use the SAME inputs, so any
    differences in UNet output come purely from the layer-level quantization
    we're isolating.
    """
    pipe.scheduler.set_timesteps(n_steps, device=device)
    ts = pipe.scheduler.timesteps

    samples = []
    with torch.no_grad():
        for prompt in prompts:
            text_input = pipe.tokenizer(prompt, padding="max_length", max_length=77,
                                        truncation=True, return_tensors="pt").to(device)
            enc_hs = pipe.text_encoder(text_input.input_ids)[0].to(dtype=dtype)

            torch.manual_seed(seed)
            latent = torch.randn(
                (1, 4, resolution // 8, resolution // 8),
                device=device, dtype=dtype,
            ) * pipe.scheduler.init_noise_sigma

            for step_i, t in enumerate(ts):
                pipe.scheduler._step_index = step_i
                latent_in = pipe.scheduler.scale_model_input(latent, t)
                samples.append({
                    "latent_in": latent_in.detach().clone(),
                    "timestep":  t,
                    "enc_hs":    enc_hs.detach().clone(),
                })
                # Advance teacher trajectory so the NEXT step's input is realistic
                noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=enc_hs).sample
                pipe.scheduler._step_index = step_i
                latent = pipe.scheduler.step(noise_pred, t, latent).prev_sample
    return samples


def run_unet_on_samples(unet, samples):
    """Forward unet on every sample and return the stack of outputs."""
    outs = []
    with torch.no_grad():
        for s in samples:
            out = unet(s["latent_in"], s["timestep"],
                       encoder_hidden_states=s["enc_hs"]).sample
            outs.append(out.detach().cpu().float())
    return outs


# —— LD-style divergence metric ───────────────────────────────────────────────

def compute_divergence(quant_outs, baseline_outs):
    """Sum of (mean_distance + std_distance) across calib samples.

    Mirrors the LD-Pruner metric: captures BOTH the mean drift (output is
    biased) and the std drift (output distribution is reshaped). A layer with
    high score = the model output as a whole moves a lot if we quantize this
    layer.
    """
    total = 0.0
    for q, b in zip(quant_outs, baseline_outs):
        # Per-sample mean+std distance (averaged over channels & spatial dims)
        mean_dist = (q.mean() - b.mean()).abs().item()
        std_dist  = (q.std()  - b.std() ).abs().item()
        total += mean_dist + std_dist
    return total / max(1, len(quant_outs))


# —— Quantization recipes ─────────────────────────────────────────────────────

def get_quant_config(bit):
    """Return a torchao config object for the requested bit-width."""
    if bit == "int8":
        from torchao.quantization import Int8WeightOnlyConfig
        return Int8WeightOnlyConfig()
    if bit == "int4":
        from torchao.quantization import Int4WeightOnlyConfig
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        return Int4WeightOnlyConfig(int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
    raise ValueError(f"Unknown bit-width: {bit}")


def quantize_single_layer(unet, layer_fqn, bit):
    """Apply torchao quantization to EXACTLY one layer (by fully-qualified name)."""
    from torchao.quantization import quantize_
    config = get_quant_config(bit)
    quantize_(unet, config, filter_fn=lambda m, fqn: fqn == layer_fqn)


# —— Sensitivity sweep ────────────────────────────────────────────────────────

def profile_sensitivity(unet, samples, target_layers, bits, baseline_outs, device):
    """For each (layer, bit) pair, quantize that layer alone and measure drift.

    Strategy: snapshot the original module via deepcopy → apply torchao quant
    on the whole UNet with a filter_fn that matches only this layer → run
    inference on calib samples → restore original module via setattr.

    The UNet state stays consistent throughout (one quant test at a time);
    no full rebuild needed between tests.
    """
    results = {}
    n_targets = len(target_layers)

    for i, (layer_fqn, mod_type, n_params) in enumerate(target_layers):
        results[layer_fqn] = {
            "type":        mod_type,
            "param_count": n_params,
        }
        parent, attr = _get_parent_attr(unet, layer_fqn)
        original = copy.deepcopy(getattr(parent, attr))

        for bit in bits:
            t0 = time.time()
            try:
                quantize_single_layer(unet, layer_fqn, bit)
                quant_outs = run_unet_on_samples(unet, samples)
                score = compute_divergence(quant_outs, baseline_outs)
                results[layer_fqn][bit] = round(score, 6)
            except Exception as e:
                # Layer can't be quantized at this bit (e.g. INT4 shape constraint)
                results[layer_fqn][bit] = None
                results[layer_fqn][f"{bit}_error"] = f"{type(e).__name__}: {e}"
            finally:
                # Restore original module no matter what
                setattr(parent, attr, copy.deepcopy(original))

            dt = time.time() - t0
            score_str = (f"{results[layer_fqn][bit]:.5f}"
                         if results[layer_fqn][bit] is not None else "FAIL")
            print(f"  [{i+1:>3}/{n_targets}] {layer_fqn} ({mod_type}, {n_params/1e6:.2f}M)"
                  f"  {bit}={score_str}  ({dt:.1f}s)")

        del original
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# —— UNet loader (reuses prune.pruned_rebuild) ───────────────────────────────

def load_unet(weights_path, config_path, device, dtype):
    from prune.pruned_rebuild import create_unet_from_safetensors
    unet = create_unet_from_safetensors(weights_path, config_path)
    unet = unet.to(device=device, dtype=dtype)
    unet.eval()
    return unet


# —— Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-layer quantization sensitivity profiling for SD-Turbo UNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--unet-weights", default=None)
    parser.add_argument("--unet-config",  default=None)
    parser.add_argument("--max-prompts",  type=int, default=None,
                        help="Override calib_max_prompts from config (for quick tests)")
    parser.add_argument("--bits", default=None,
                        help="Comma-separated bit-widths to test (default: from config)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results_dir/sensitivity.json)")
    parser.add_argument("--include-conv2d", action="store_true",
                        help="Also profile Conv2d layers (requires Conv2d-capable "
                             "quantization recipes; default off because torchao + "
                             "our GPTQ implementation currently target Linear only).")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: CPU mode is very slow. Quantization kernels also require CUDA.")

    base_model   = cfg.get("base_model", "stabilityai/sd-turbo")
    skip_layers  = cfg.get("skip_layers", [])
    n_steps      = cfg.get("inference_steps", 4)
    resolution   = cfg.get("resolution",     512)
    seed         = cfg.get("seed",           42)
    results_dir  = cfg.get("results_dir",    "mp_quant/results")
    n_prompts    = args.max_prompts or cfg.get("calib_max_prompts", 32)
    bits         = ([b.strip() for b in args.bits.split(",")]
                    if args.bits else cfg.get("quant_candidates", ["int8", "int4"]))

    os.makedirs(results_dir, exist_ok=True)
    out_path = args.output or os.path.join(results_dir, "sensitivity.json")

    # Load prompts
    import glob as _glob
    prompt_paths = sorted(_glob.glob(os.path.join(cfg.get("calib_dataset", "dataset"),
                                                  "*.txt")))[:n_prompts]
    if not prompt_paths:
        print(f"ERROR: no prompts found in {cfg.get('calib_dataset')}")
        sys.exit(1)
    prompts = []
    for p in prompt_paths:
        with open(p, encoding="utf-8") as f:
            prompts.append(f.read().strip())
    print(f"Loaded {len(prompts)} prompts")

    # Resolve UNet source once (same weights for INT8 and INT4 passes)
    unet_weights, unet_config = _resolve_unet_source(args, cfg)
    print(f"UNet source: {unet_weights}")

    # Load base pipeline (for tokenizer, text_encoder, scheduler)
    from diffusers import StableDiffusionPipeline
    print("Loading base pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)

    # Run separate passes per bit-width because they need different dtypes
    # (INT4 TILE_PACKED_TO_4D only accepts bf16; INT8 works with fp16).
    all_results = {}
    for bit in bits:
        dtype = torch.bfloat16 if bit == "int4" else torch.float16
        print(f"\n{'='*70}\nPass: bit={bit}  dtype={dtype}\n{'='*70}")

        unet = load_unet(unet_weights, unet_config, device, dtype)

        # Need pipe to match dtype for text encoder + scheduler
        pipe.text_encoder = pipe.text_encoder.to(dtype=dtype)
        pipe.unet = unet  # use our UNet for calib data generation

        # 1. Build calib samples
        print("Building calibration samples...")
        samples = build_calib_samples(pipe, prompts, n_steps, resolution, seed, device, dtype)
        print(f"  {len(samples)} samples ({len(prompts)} prompts × {n_steps} steps)")

        # 2. Baseline outputs at fp16/bf16 (no quantization)
        print("Computing baseline outputs...")
        baseline_outs = run_unet_on_samples(unet, samples)

        # 3. Find target layers (Linear only by default; --include-conv2d to expand)
        targets = find_target_layers(unet, skip_layers,
                                     include_conv2d=args.include_conv2d)
        type_counts = {}
        for _, t, _ in targets:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Profiling {len(targets)} target layers: {type_counts}")

        # 4. Profile sensitivity for this single bit
        bit_results = profile_sensitivity(unet, samples, targets, [bit], baseline_outs, device)

        # Merge into all_results (each layer gets entries for each bit)
        for layer_fqn, info in bit_results.items():
            if layer_fqn not in all_results:
                all_results[layer_fqn] = {
                    "type":        info["type"],
                    "param_count": info["param_count"],
                }
            all_results[layer_fqn][bit] = info.get(bit)
            if f"{bit}_error" in info:
                all_results[layer_fqn][f"{bit}_error"] = info[f"{bit}_error"]

        # Clean up before next pass
        del unet, samples, baseline_outs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # 5. Save
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "base_model":  base_model,
                "unet_source": unet_weights,
                "n_prompts":   len(prompts),
                "n_steps":     n_steps,
                "resolution":  resolution,
                "seed":        seed,
                "skip_layers": skip_layers,
                "bits":        bits,
            },
            "layers": all_results,
        }, f, indent=2)
    print(f"\nSensitivity report saved: {out_path}")

    # Brief summary
    print("\n──── Top 10 most-sensitive layers (avg of all bits) ────")
    avg_scores = []
    for fqn, info in all_results.items():
        scores = [info[b] for b in bits if info.get(b) is not None]
        if scores:
            avg_scores.append((fqn, sum(scores) / len(scores), info["type"], info["param_count"]))
    avg_scores.sort(key=lambda x: -x[1])
    for fqn, score, mtype, params in avg_scores[:10]:
        print(f"  {score:.5f}  {fqn:<70} ({mtype}, {params/1e6:.2f}M)")

    print("\n──── Top 10 least-sensitive layers (best to quantize) ────")
    for fqn, score, mtype, params in avg_scores[-10:][::-1]:
        print(f"  {score:.5f}  {fqn:<70} ({mtype}, {params/1e6:.2f}M)")


if __name__ == "__main__":
    main()
