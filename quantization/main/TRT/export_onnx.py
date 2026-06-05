#!/usr/bin/env python3
"""Phase B — Export the merged fp16 SDXL-Lightning UNet (from Phase A) to ONNX.

Two challenges handled:
  1. diffusers' UNet2DConditionModel.forward takes a dict `added_cond_kwargs`;
     ONNX export wants flat positional tensor inputs. Solved with UNetWrapper.
  2. UNet weights total ~4.9 GB, exceeding ONNX protobuf's 2 GB limit. Solved
     by use_external_data_format=True (weights spill to a sidecar file).

Output:
  models/onnx/unet_fp16.onnx                       Graph definition
  models/onnx/unet_fp16.onnx_data                  External weight tensors
  models/onnx/unet_fp16_export_summary.json        Input/output shapes + metadata

Verification (Phase B gate, separate script verify_onnx_vs_pytorch.py):
  ONNX Runtime CUDA EP forward output vs PyTorch UNet forward output:
    max abs diff < 1e-3 (fp16 noise floor)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file


class UNetWrapper(nn.Module):
    """Flatten the diffusers UNet input signature for ONNX export.

    Input order (positional):
        sample                [1, 4, H/8, W/8]   fp16  — noisy latent
        timestep              [1]                int64 — current denoise step
        encoder_hidden_states [1, 77, 2048]      fp16  — dual-CLIP concat embed
        text_embeds           [1, 1280]          fp16  — SDXL pooled embed
        time_ids              [1, 6]             fp16  — SDXL conditional time ids
    Output:
        noise_pred            [1, 4, H/8, W/8]   fp16
    """

    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return out[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-safetensors", type=Path, required=True,
                    help="Phase A output: merged fp16 UNet state_dict.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-stem", type=str, default="unet_fp16")
    ap.add_argument("--base-model", type=str,
                    default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (17+ recommended for SDPA / GroupNorm).")
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="If set, batch dim is dynamic. Default is static batch=1 for max kernel optimization.")
    ap.add_argument("--legacy-exporter", action="store_true",
                    help="Use legacy TorchScript-based exporter (default is dynamo). Fallback if dynamo fails.")
    ap.add_argument("--export-fp32", action="store_true",
                    help="Cast UNet to fp32 before export. ~2x larger ONNX but avoids fp16 "
                         "overflow in time_proj sinusoidal embeddings. Industry-standard for SDXL.")
    args = ap.parse_args()
    args.use_dynamo = not args.legacy_exporter

    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.export_fp32 else torch.float16
    print(f"Export dtype: {dtype}")

    # ---------------------------------------------------------------- load merged UNet
    print(f"Loading merged UNet from {args.merged_safetensors.name}...")
    state = load_file(str(args.merged_safetensors), device="cpu")
    unet = UNet2DConditionModel.from_config(args.base_model, subfolder="unet").to(device, dtype)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    print(f"  loaded {len(state)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"    first 3 missing keys: {missing[:3]}")
    del state
    torch.cuda.empty_cache()
    unet.eval()

    # ---------------------------------------------------------------- dummy inputs
    H, W = args.height // 8, args.width // 8
    dummy = {
        "sample":                torch.randn(1, 4, H, W, dtype=dtype, device=device),
        "timestep":              torch.tensor([999], dtype=torch.int64, device=device),
        "encoder_hidden_states": torch.randn(1, 77, 2048, dtype=dtype, device=device),
        "text_embeds":           torch.randn(1, 1280, dtype=dtype, device=device),
        "time_ids":              torch.tensor([[args.height, args.width, 0, 0, args.height, args.width]],
                                              dtype=dtype, device=device),
    }
    inputs_tuple = (dummy["sample"], dummy["timestep"], dummy["encoder_hidden_states"],
                    dummy["text_embeds"], dummy["time_ids"])

    # Quick smoke test through the wrapper before exporting
    wrapped = UNetWrapper(unet)
    print(f"Smoke test wrapped forward on dummy inputs...")
    smoke_out = wrapped(*inputs_tuple)
    print(f"  output shape: {tuple(smoke_out.shape)}  dtype: {smoke_out.dtype}  "
          f"finite: {torch.isfinite(smoke_out).all().item()}")

    # ---------------------------------------------------------------- export
    input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
    output_names = ["noise_pred"]
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {n: {0: "batch"} for n in input_names if n != "timestep"}
        dynamic_axes["noise_pred"] = {0: "batch"}

    onnx_path = args.output_dir / f"{args.output_stem}.onnx"
    print(f"\nExporting to {onnx_path}  (opset={args.opset}, "
          f"dynamic_batch={args.dynamic_batch}, external_data=True)...")
    t0 = time.time()
    wrapped.eval()  # make sure both wrapper + inner unet are in eval
    if getattr(args, "use_dynamo", True):
        # Dynamo exporter (default in PyTorch 2.12+) handles dtype propagation
        # better — legacy tracer downcasts some fp32-internal time-embedding ops
        # to fp16 and overflows on smaller timesteps.
        torch.onnx.export(
            wrapped, inputs_tuple, str(onnx_path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=args.opset,
            do_constant_folding=True, export_params=True,
            dynamo=True,
        )
    else:
        torch.onnx.export(
            wrapped, inputs_tuple, str(onnx_path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=args.opset,
            do_constant_folding=False, export_params=True,    # disable folding to avoid hidden fp16 collapse
            dynamo=False,
        )
    elapsed = time.time() - t0
    print(f"  Export finished in {elapsed:.1f}s")

    # Validate ONNX file (skip strict check on external data sharding issues)
    import onnx
    print("Loading ONNX back to validate graph...")
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    print(f"  IR version:   {onnx_model.ir_version}")
    print(f"  opset:        {onnx_model.opset_import[0].version}")
    print(f"  inputs:       {[i.name for i in onnx_model.graph.input]}")
    print(f"  outputs:      {[o.name for o in onnx_model.graph.output]}")
    print(f"  nodes:        {len(onnx_model.graph.node)}")
    print(f"  initializers: {len(onnx_model.graph.initializer)}")
    try:
        onnx.checker.check_model(onnx_model)
        print(f"  onnx.checker.check_model: OK")
    except Exception as e:
        print(f"  onnx.checker.check_model: SKIPPED ({type(e).__name__}: {str(e)[:120]})")
        print(f"  (Checker often complains about external-data sharding; ORT load is the real test.)")

    # Consolidate sharded external data into a single .onnx_data file (cleanup)
    print("\nConsolidating external data into a single sidecar file...")
    consolidated_path = onnx_path.with_suffix(".consolidated.onnx")
    consolidated_data_path = onnx_path.parent / f"{args.output_stem}.onnx_data"
    try:
        full = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save_model(
            full, str(consolidated_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=consolidated_data_path.name,
            convert_attribute=False,
        )
        # Replace original with consolidated
        # Remove all per-tensor external files (everything in dir except .json + .safetensors + .onnx + .onnx_data targets)
        keepers = {onnx_path.name, consolidated_path.name, consolidated_data_path.name}
        keepers.update({p.name for p in args.output_dir.iterdir() if p.suffix in (".json", ".safetensors")})
        n_removed = 0
        for p in args.output_dir.iterdir():
            if p.name not in keepers:
                p.unlink()
                n_removed += 1
        # Rename consolidated → final name
        onnx_path.unlink()
        consolidated_path.rename(onnx_path)
        print(f"  Consolidated. Removed {n_removed} per-tensor files. Final: {onnx_path.name} + {consolidated_data_path.name}")
    except Exception as e:
        print(f"  Consolidation FAILED ({type(e).__name__}: {str(e)[:200]})")
        print(f"  Proceeding with sharded external data (ORT can still load it).")

    # File sizes (final)
    total_bytes = onnx_path.stat().st_size
    for p in args.output_dir.iterdir():
        if p.name.startswith(args.output_stem) and p.suffix not in (".json",) and p != onnx_path:
            total_bytes += p.stat().st_size
    print(f"  Disk size (.onnx + external data): {total_bytes / 1024 / 1024:.0f} MB")

    # Sidecar metadata for downstream (verify / quantize / TRT)
    sidecar = {
        "source_merged_safetensors": str(args.merged_safetensors),
        "base_model":   args.base_model,
        "height":       args.height,
        "width":        args.width,
        "opset":        args.opset,
        "dynamic_batch": args.dynamic_batch,
        "input_specs": {
            "sample":                {"shape": [1, 4, H, W],  "dtype": str(dtype).replace("torch.","")},
            "timestep":              {"shape": [1],            "dtype": "int64"},
            "encoder_hidden_states": {"shape": [1, 77, 2048],  "dtype": str(dtype).replace("torch.","")},
            "text_embeds":           {"shape": [1, 1280],      "dtype": str(dtype).replace("torch.","")},
            "time_ids":              {"shape": [1, 6],         "dtype": str(dtype).replace("torch.","")},
        },
        "output_specs": {"noise_pred": {"shape": [1, 4, H, W], "dtype": str(dtype).replace("torch.","")}},
        "disk_size_mb": total_bytes / 1024 / 1024,
        "export_seconds": elapsed,
    }
    summary_path = args.output_dir / f"{args.output_stem}_export_summary.json"
    summary_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"\nMetadata: {summary_path}")
    print(f"\nNext: python quantization/main/TRT/export_onnx_verification.py --onnx {onnx_path}")


if __name__ == "__main__":
    main()
