#!/usr/bin/env python3
"""Phase C-ORT (real version) — Static INT8 QDQ quantization of the fp32
SDXL-Lightning UNet ONNX.

Differences vs the earlier dynamic attempt:
  - Static (not dynamic): we run real calibration data through the fp32 ONNX
    to record activation min/max, computing precise per-tensor scales.
  - QDQ format (not QOperator): inserts Quantize/DequantizeLinear nodes around
    each quantized op. The original MatMul / Conv / Gemm node stays in the
    graph; runtimes (ORT, TRT) recognize QDQ and dispatch to INT8 kernels
    where available, otherwise fall back to fp32 with the wrapped op.
  - Conv is included: SDXL UNet has most of its parameters in Conv. Including
    Conv in QDQ gives meaningful (~70%) compression that dynamic quant could
    not achieve (its ConvInteger op has no kernel implementation).

Calibration data: the 256 teacher trajectory samples already cached for Q-LoRA
training in qlora_teacher_cache_128p_1024 are reused as calibration inputs.

Critical nodes excluded by name pattern (analog of PyTorch SKIP_PATTERNS):
  conv_in, conv_out, time_embedding, time_emb_proj, add_embedding.
These are latent-adjacent or time-conditioning paths where INT8 noise tends
to amplify. Matching what our PyTorch pipeline already does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from safetensors.torch import load_file


CRITICAL_NAME_FRAGMENTS = (
    "conv_in", "conv_out",
    "time_embedding", "time_emb_proj", "add_embedding",
    # Light extra safety — small embedding-side projections
    "add_time_proj",
)


class TeacherCacheCalibrationReader(CalibrationDataReader):
    """Yields cached (sample, timestep, embeds, time_ids) tuples for static
    quantization. Each get_next() call returns a dict matching the ONNX input
    names; returns None when exhausted."""

    def __init__(self, teacher_cache_dir: Path, n_samples: int, onnx_input_dtypes: dict):
        self.files = sorted(teacher_cache_dir.glob("prompt*_step*.safetensors"))[:n_samples]
        if not self.files:
            raise FileNotFoundError(f"No teacher-cache samples found in {teacher_cache_dir}")
        self.idx = 0
        self.dtypes = onnx_input_dtypes
        print(f"  CalibrationDataReader: {len(self.files)} samples queued")

    def get_next(self) -> dict | None:
        if self.idx >= len(self.files):
            return None
        f = self.files[self.idx]
        self.idx += 1
        sample = load_file(str(f), device="cpu")
        # Map teacher-cache keys → ONNX input names, cast to declared dtypes
        feeds = {
            "sample":                sample["latent_in"].numpy(),
            "timestep":              sample["timestep"].numpy(),
            "encoder_hidden_states": sample["prompt_embeds"].numpy(),
            "text_embeds":           sample["pooled_embeds"].numpy(),
            "time_ids":              sample["time_ids"].numpy(),
        }
        out = {}
        for k, v in feeds.items():
            target_dtype = self.dtypes[k]
            out[k] = v.astype(target_dtype)
            if k == "timestep" and out[k].ndim == 0:
                out[k] = out[k][None]
        if self.idx % 32 == 0 or self.idx == len(self.files):
            print(f"  calib feed {self.idx}/{len(self.files)}: {f.name}")
        return out

    def rewind(self) -> None:
        self.idx = 0


_NP_DTYPE = {
    "tensor(float)":  np.float32, "tensor(float16)": np.float16,
    "tensor(int64)":  np.int64,   "tensor(int32)":   np.int32,
}


def collect_nodes_to_exclude(onnx_path: Path) -> list[str]:
    """Return node names matching any CRITICAL_NAME_FRAGMENTS."""
    print(f"Inspecting ONNX graph to find critical nodes to exclude...")
    model = onnx.load(str(onnx_path), load_external_data=False)
    exclude = []
    for node in model.graph.node:
        nm = node.name or ""
        if any(frag in nm for frag in CRITICAL_NAME_FRAGMENTS):
            exclude.append(nm)
    print(f"  Found {len(exclude)} critical nodes (will stay fp32). Samples:")
    for n in exclude[:5]:
        print(f"    {n}")
    if len(exclude) > 5:
        print(f"    ... +{len(exclude)-5} more")
    return exclude


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-onnx", type=Path, required=True,
                    help="Source fp32 ONNX (Phase B output).")
    ap.add_argument("--output-onnx", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, required=True,
                    help="Dir with prompt*_step*.safetensors for calibration.")
    ap.add_argument("--n-calib", type=int, default=64,
                    help="Number of calibration samples (256 available, 64 is usually enough).")
    ap.add_argument("--op-types", type=str, default="MatMul,Conv,Gemm",
                    help="Comma-separated op types to quantize.")
    ap.add_argument("--no-conv", action="store_true",
                    help="Skip Conv ops (only quantize MatMul/Gemm). Smaller compression but safer.")
    ap.add_argument("--calib-method", choices=("minmax", "entropy", "percentile", "distribution"),
                    default="percentile",
                    help="Percentile (99.99) is recommended for SDXL (outlier-tolerant).")
    ap.add_argument("--activation-type", choices=("int8", "uint8"), default="uint8",
                    help="QUInt8 (asymmetric) is recommended for SDXL — better fits "
                         "GELU/SiLU/LayerNorm output distributions than symmetric QInt8.")
    ap.add_argument("--skip-preprocess", action="store_true",
                    help="Skip the quant_pre_process step (shape inference + symbolic shape).")
    args = ap.parse_args()

    if not args.input_onnx.exists():
        sys.exit(f"Input ONNX not found: {args.input_onnx}")
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------- 1. Pre-process the fp32 ONNX (recommended)
    # quant_pre_process adds shape inference + minor optimizations that help
    # the static quantizer assign correct scales. Output is a "preprocessed"
    # fp32 ONNX we feed to quantize_static below.
    preprocessed_path = args.input_onnx.with_name(args.input_onnx.stem + "_preprocessed.onnx")
    if not args.skip_preprocess and not preprocessed_path.exists():
        print(f"Pre-processing ONNX (shape inference + small fusion)...")
        t0 = time.time()
        quant_pre_process(
            input_model_path=str(args.input_onnx),
            output_model_path=str(preprocessed_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=False,
            external_data_location=preprocessed_path.name + "_data",
            external_data_size_threshold=1024,
        )
        print(f"  Pre-process done in {time.time() - t0:.0f}s → {preprocessed_path.name}")
    elif preprocessed_path.exists():
        print(f"Reusing existing preprocessed ONNX: {preprocessed_path.name}")
    else:
        print(f"Skipping pre-processing (--skip-preprocess given)")
        preprocessed_path = args.input_onnx
    quant_input = preprocessed_path

    # ----------------------------- 2. Probe ONNX input dtypes for calibration reader
    print(f"Probing ONNX input dtypes from {quant_input.name}...")
    import onnxruntime as ort
    probe_sess = ort.InferenceSession(
        str(quant_input), providers=["CPUExecutionProvider"])
    input_dtypes = {i.name: _NP_DTYPE[i.type] for i in probe_sess.get_inputs()}
    print(f"  ONNX inputs: {input_dtypes}")
    del probe_sess

    # ----------------------------- 3. Build calibration reader
    reader = TeacherCacheCalibrationReader(
        args.teacher_cache, n_samples=args.n_calib, onnx_input_dtypes=input_dtypes,
    )

    # ----------------------------- 4. Collect critical-node exclusions
    nodes_to_exclude = collect_nodes_to_exclude(quant_input)

    # ----------------------------- 5. Run static QDQ quantization
    op_types = [s.strip() for s in args.op_types.split(",") if s.strip()]
    if args.no_conv:
        op_types = [o for o in op_types if o != "Conv"]
    method_map = {
        "minmax":       CalibrationMethod.MinMax,
        "entropy":      CalibrationMethod.Entropy,
        "percentile":   CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    act_type = QuantType.QUInt8 if args.activation_type == "uint8" else QuantType.QInt8
    print(f"\nQuantizing static QDQ INT8:")
    print(f"  op_types_to_quantize = {op_types}")
    print(f"  calib_method          = {args.calib_method}")
    print(f"  per_channel           = True")
    print(f"  weight     type       = QInt8 (symmetric)")
    print(f"  activation type       = {'QUInt8 (asymmetric)' if act_type == QuantType.QUInt8 else 'QInt8 (symmetric)'}")
    print(f"  nodes_to_exclude      = {len(nodes_to_exclude)} critical nodes")
    print(f"  output                = {args.output_onnx}")

    t0 = time.time()
    quantize_static(
        model_input=str(quant_input),
        model_output=str(args.output_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=act_type,
        op_types_to_quantize=op_types,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=method_map[args.calib_method],
        use_external_data_format=True,
        # Run calibration forward on CUDA for speed + lower RAM pressure.
        extra_options={
            "CalibProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
    )
    print(f"\nquantize_static done in {time.time() - t0:.0f}s")

    # ----------------------------- 6. Report sizes
    def dir_size_for_stem(stem: Path) -> int:
        n = 0
        for p in stem.parent.iterdir():
            if p.name.startswith(stem.name):
                n += p.stat().st_size
        return n

    src_bytes = dir_size_for_stem(args.input_onnx.with_suffix(""))
    out_bytes = dir_size_for_stem(args.output_onnx.with_suffix(""))
    print(f"\nSize comparison:")
    print(f"  fp32 source: {src_bytes / 1024 / 1024:7.0f} MB")
    print(f"  int8 QDQ:    {out_bytes / 1024 / 1024:7.0f} MB")
    print(f"  reduction:   {(1 - out_bytes / src_bytes) * 100:.1f}%")
    print(f"\nNext: python mp_quant/eval_onnx_e2e.py --onnx {args.output_onnx} --ep cuda")


if __name__ == "__main__":
    main()
