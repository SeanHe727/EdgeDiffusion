#!/usr/bin/env python3
"""Phase C-ORT — Quantize the fp32 SDXL-Lightning UNet ONNX to INT8 weights
using ONNX Runtime's quantize_dynamic.

What this does mathematically:
  For each MatMul / Gemm node with a weight initializer:
    - Compute per-channel symmetric scale = max(|W|) / 127
    - Replace W (fp32 [out, in]) with W_int8 + scale (fp16/fp32)
    - Insert DequantizeLinear before the GEMM at runtime (CUDA EP fuses this
      into INT8 GEMM when possible; CPU EP uses INT8 MatMul kernel directly)

Result: smaller ONNX (~3 GB vs 10 GB fp32) + faster inference where INT8
kernels exist. Activations stay fp32/fp16 — dynamic ranges computed
per-call.

Cross-platform: the output unet_int8.onnx loads on CUDA / CPU / OpenVINO /
CoreML / RKNN. INT8 weight-only deployment is the most widely-supported
quantization scheme.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-onnx", type=Path, required=True,
                    help="Source fp32 ONNX (Phase B output).")
    ap.add_argument("--output-onnx", type=Path, required=True,
                    help="Target INT8 ONNX path.")
    ap.add_argument("--weight-type", choices=("int8", "uint8"), default="int8",
                    help="QuantType for weights. int8 (symmetric) is standard.")
    ap.add_argument("--per-channel", action="store_true", default=True,
                    help="Per-output-channel quantization (better quality than per-tensor).")
    ap.add_argument("--reduce-range", action="store_true",
                    help="Restrict to [-64, 64] for legacy hardware. Default off on modern GPUs.")
    ap.add_argument("--op-types", type=str, default="MatMul,Gemm",
                    help="Comma-separated op types to quantize. Conv is excluded by default "
                         "because ORT's ConvInteger has no CUDA/CPU kernel for SDXL shapes.")
    args = ap.parse_args()

    if not args.input_onnx.exists():
        sys.exit(f"Input ONNX not found: {args.input_onnx}")
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)

    weight_type = QuantType.QInt8 if args.weight_type == "int8" else QuantType.QUInt8

    print(f"Source: {args.input_onnx}")
    print(f"  size: {args.input_onnx.stat().st_size / 1024 / 1024:.0f} MB graph "
          f"(+ external data files)")
    print(f"Target: {args.output_onnx}")
    print(f"Config: weight_type={weight_type}, per_channel={args.per_channel}, "
          f"reduce_range={args.reduce_range}")

    t0 = time.time()
    print(f"\nRunning quantize_dynamic... (this loads the full fp32 model into RAM)")
    op_types = [s.strip() for s in args.op_types.split(",") if s.strip()]
    print(f"  op_types_to_quantize: {op_types}")
    quantize_dynamic(
        model_input=str(args.input_onnx),
        model_output=str(args.output_onnx),
        weight_type=weight_type,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        op_types_to_quantize=op_types,
        # Use external data for the resulting INT8 weights (still likely >2GB)
        use_external_data_format=True,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")

    # Report file sizes
    out_bytes = args.output_onnx.stat().st_size
    out_data_path = args.output_onnx.with_suffix(args.output_onnx.suffix + "_data")
    if not out_data_path.exists():
        # ORT might name it differently — find any sibling extra files
        siblings = [p for p in args.output_onnx.parent.iterdir()
                    if p.name.startswith(args.output_onnx.stem) and p != args.output_onnx
                    and p.suffix not in (".json",)]
        for p in siblings:
            out_bytes += p.stat().st_size
    else:
        out_bytes += out_data_path.stat().st_size

    in_bytes = args.input_onnx.stat().st_size
    in_data_path = args.input_onnx.parent / f"{args.input_onnx.stem}.onnx_data"
    if in_data_path.exists():
        in_bytes += in_data_path.stat().st_size

    print(f"\nSize comparison:")
    print(f"  fp32 source: {in_bytes  / 1024 / 1024:7.0f} MB")
    print(f"  int8 target: {out_bytes / 1024 / 1024:7.0f} MB")
    print(f"  reduction:   {(1 - out_bytes / in_bytes) * 100:.1f}%")
    print(f"\nNext: run mp_quant/eval_onnx_e2e.py --onnx {args.output_onnx}")


if __name__ == "__main__":
    main()
