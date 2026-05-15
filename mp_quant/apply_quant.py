#!/usr/bin/env python3
"""
Stage 3: Apply mixed-precision quantization.

Reads the per-layer bit-width assignment produced by Stage 2 and applies the
corresponding quantization to each layer using our self-implemented GPTQ
(see mp_quant/gptq.py).

Flow:
  1. Load fp16 UNet from HF (or local override).
  2. Load bit-width assignment from Stage 2 (results/bitwidth_config.json).
  3. Build calibration samples (16-32 prompts × 4 timesteps).
  4. Group layers by bit-width.
  5. For each non-fp16 layer:
       - Collect input Hessian via forward hook on calib data.
       - Run GPTQ at the assigned bit-width.
       - Replace the layer's weight tensor with the quantized values.
  6. Save the resulting UNet (still in safetensors / .pt format with fake-quant
     weights — actual packed storage is a deployment concern, decoupled here).

Notes:
  - We do "fake quantization" — the weight tensor is rounded to the quant grid
    but stored as fp16/bf16. This is enough for evaluating quality (Stage 5)
    and for Stage 4 (QLoRA recovery). Real packed storage (INT4/INT8 bits in
    actual smaller dtype) can be added at the very end for deployment.
  - Conv2d layers in v1 are left fp16 (the solver assigns them fp16 because
    Stage 1 only profiled Linear). Extending GPTQ to Conv2d is future work.

Run:
  python mp_quant/apply_quant.py
  python mp_quant/apply_quant.py --bitwidth-config <path>
"""
import os
import sys
import gc
import json
import time
import argparse
import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mp_quant.sensitivity import (
    _load_config,
    _resolve_unet_source,
    load_unet,
    build_calib_samples,
)
from mp_quant.gptq import HessianAccumulator, gptq_quantize_linear


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "mp_quant_config.yaml")


# INT4 requires bf16 (TILE_PACKED_TO_4D constraint). If ANY layer is assigned
# int4, we run the whole UNet in bf16 — INT8 layers tolerate bf16 with
# negligible loss vs fp16, but INT4 cannot work in fp16.
def pick_dtype(assignment):
    return torch.bfloat16 if any(b == "int4" for b in assignment.values()) else torch.float16


def collect_hessians_in_one_pass(unet, target_fqns, calib_samples, device):
    """Run calib data ONCE; accumulate Hessians for every target layer in parallel."""
    accumulators = {}
    handles = []
    name_to_module = dict(unet.named_modules())
    for fqn in target_fqns:
        mod = name_to_module[fqn]
        if not isinstance(mod, nn.Linear):
            continue                          # safety guard; we only quantize Linear here
        acc = HessianAccumulator(mod.in_features, device=device)
        accumulators[fqn] = acc

        def make_hook(acc):
            def hook(_mod, args, _kwargs):
                acc.add_batch(args[0])
            return hook
        handles.append(mod.register_forward_pre_hook(make_hook(acc), with_kwargs=True))

    with torch.no_grad():
        for s in calib_samples:
            _ = unet(s["latent_in"], s["timestep"],
                     encoder_hidden_states=s["enc_hs"]).sample
    for h in handles:
        h.remove()

    return {fqn: acc.finalize() for fqn, acc in accumulators.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Apply mixed-precision quantization via GPTQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",           default=DEFAULT_CONFIG)
    parser.add_argument("--bitwidth-config",  default=None,
                        help="Path to bitwidth_config.json from Stage 2")
    parser.add_argument("--output",           default=None,
                        help="Output safetensors path")
    parser.add_argument("--unet-weights",     default=None)
    parser.add_argument("--unet-config",      default=None)
    parser.add_argument("--max-prompts",      type=int, default=None,
                        help="Override calib_max_prompts from config")
    parser.add_argument("--group-size",       type=int, default=128,
                        help="GPTQ group size for scale granularity")
    parser.add_argument("--damping",          type=float, default=0.01,
                        help="Hessian diagonal damping (% of mean diag)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: GPU required for GPTQ (Hessian inverse + group-wise quant)")
        sys.exit(1)

    results_dir = cfg.get("results_dir", "mp_quant/results")
    output_dir  = cfg.get("output_dir",  "mp_quant/output")
    bw_path     = args.bitwidth_config or os.path.join(results_dir, "bitwidth_config.json")
    n_prompts   = args.max_prompts or cfg.get("calib_max_prompts", 32)

    # —— Load bit-width assignment ────────────────────────────────────────────
    with open(bw_path, encoding="utf-8") as f:
        bw_report = json.load(f)
    assignment = bw_report["assignment"]
    n_total    = len(assignment)
    n_int4 = sum(1 for v in assignment.values() if v == "int4")
    n_int8 = sum(1 for v in assignment.values() if v == "int8")
    n_fp16 = sum(1 for v in assignment.values() if v == "fp16")
    print(f"Bit-width plan ({n_total} layers): "
          f"int4={n_int4}, int8={n_int8}, fp16={n_fp16}")
    print(f"Achieved size reduction: {bw_report.get('achieved_reduction', 0):.1%}")

    dtype = pick_dtype(assignment)
    print(f"Model dtype: {dtype} ({'bf16 forced by INT4 layers' if dtype==torch.bfloat16 else 'fp16'})")

    # —— Load UNet + pipeline (for tokenizer / text_encoder / scheduler) ─────
    print("\nLoading UNet + base pipeline...")
    unet_w, unet_c = _resolve_unet_source(args, cfg)
    unet = load_unet(unet_w, unet_c, device, dtype)

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.get("base_model", "stabilityai/sd-turbo"),
        torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.unet = unet

    # —— Build calibration samples ────────────────────────────────────────────
    import glob as _glob
    prompts = []
    for p in sorted(_glob.glob(os.path.join(cfg.get("calib_dataset", "dataset"),
                                            "*.txt")))[:n_prompts]:
        with open(p, encoding="utf-8") as f:
            prompts.append(f.read().strip())
    print(f"\nBuilding calibration samples ({len(prompts)} prompts × {cfg.get('inference_steps', 4)} steps)...")
    samples = build_calib_samples(
        pipe, prompts,
        n_steps=cfg.get("inference_steps", 4),
        resolution=cfg.get("resolution", 512),
        seed=cfg.get("seed", 42),
        device=device, dtype=dtype,
    )
    print(f"  {len(samples)} samples")

    # —— Collect Hessians for ALL layers to be quantized (one calib pass) ────
    target_fqns = [fqn for fqn, bit in assignment.items() if bit != "fp16"]
    print(f"\nCollecting Hessians for {len(target_fqns)} layers (one forward pass)...")
    t0 = time.time()
    hessians = collect_hessians_in_one_pass(unet, target_fqns, samples, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    # —— Apply GPTQ layer by layer ────────────────────────────────────────────
    print(f"\nRunning GPTQ ...")
    t0 = time.time()
    name_to_module = dict(unet.named_modules())
    total_err = 0.0
    n_processed = 0
    by_bit_err = {"int8": 0.0, "int4": 0.0}
    by_bit_count = {"int8": 0, "int4": 0}

    for fqn, bit in assignment.items():
        if bit == "fp16":
            continue
        mod = name_to_module[fqn]
        if not isinstance(mod, nn.Linear):
            # Conv2d in assignment for some reason; skip (left as fp16)
            print(f"  [skip Conv2d] {fqn}")
            continue
        bits = 4 if bit == "int4" else 8
        H = hessians[fqn]
        stats = gptq_quantize_linear(mod, H, bits=bits, group_size=args.group_size,
                                     damping=args.damping)
        total_err += stats["layer_error"]
        by_bit_err[bit] += stats["layer_error"]
        by_bit_count[bit] += 1
        n_processed += 1
        if n_processed % 20 == 0 or n_processed <= 3:
            print(f"  [{n_processed:>3}/{len(target_fqns)}] {fqn} → {bit} "
                  f"(err={stats['layer_error']:.5f})")
    print(f"\nGPTQ complete: {n_processed} layers in {time.time()-t0:.1f}s")
    print(f"  avg error int8: "
          f"{by_bit_err['int8']/max(1, by_bit_count['int8']):.5f} ({by_bit_count['int8']} layers)")
    print(f"  avg error int4: "
          f"{by_bit_err['int4']/max(1, by_bit_count['int4']):.5f} ({by_bit_count['int4']} layers)")

    # —— Sanity forward ───────────────────────────────────────────────────────
    print(f"\nSanity forward on {min(8, len(samples))} samples...")
    bad = 0
    with torch.no_grad():
        for s in samples[:8]:
            out = unet(s["latent_in"], s["timestep"],
                       encoder_hidden_states=s["enc_hs"]).sample
            if not torch.isfinite(out).all():
                bad += 1
    if bad > 0:
        print(f"  WARNING: {bad}/8 outputs had NaN/Inf — investigate")
    else:
        print(f"  OK")

    # —— Save ────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(unet_w))[0]   # e.g. "distill_final"
    suffix = f"_mp_r{int(100*bw_report.get('achieved_reduction', 0))}"
    out_st  = args.output or os.path.join(output_dir, f"{base}{suffix}.safetensors")
    out_cfg = out_st.replace(".safetensors", ".config.json")

    state = {k: v.detach().cpu().contiguous() for k, v in unet.state_dict().items()}
    save_file(state, out_st)

    # Reuse source model_config + add quantization metadata
    with open(unet_c, encoding="utf-8") as f:
        src_meta = json.load(f)
    sidecar = {
        **{k: v for k, v in src_meta.items() if k != "model_config"},
        "quantization": {
            "method":        "mixed_precision_gptq",
            "group_size":    args.group_size,
            "damping":       args.damping,
            "assignment":    assignment,
            "size_reduction": bw_report.get("achieved_reduction"),
            "model_dtype":   str(dtype).replace("torch.", ""),
        },
        "model_config": src_meta.get("model_config", {}),
    }
    with open(out_cfg, "w") as f:
        json.dump(sidecar, f, indent=2)

    sz_mb = os.path.getsize(out_st) / 1024 ** 2
    print(f"\nSaved:")
    print(f"  {out_st}   ({sz_mb:.1f} MB)")
    print(f"  {out_cfg}")


if __name__ == "__main__":
    main()
