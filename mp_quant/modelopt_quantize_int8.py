#!/usr/bin/env python3
"""Phase C-ORT (production path) — INT8 QDQ quantization of the fp32
SDXL-Lightning UNet via NVIDIA modelopt (formerly AMMO).

WHY MODELOPT INSTEAD OF onnxruntime.quantization.quantize_static:
  - quantize_static on a 10 GB SDXL ONNX hits BFCArena fragmentation in
    onnxruntime 1.23: even with 50+ GB free host RAM, the arena can't find a
    contiguous ~80 MB buffer for an intermediate MatMul tensor after thousands
    of allocate/free cycles. Verified twice (Windows 64 GB, EC2 g5.2xlarge 32 GB).
  - quantize_static's design envelope is <1 GB models (BERT, MobileNet, etc.).
  - NVIDIA modelopt is NVIDIA's official PTQ toolkit, designed for LLMs and
    diffusion models. Calibration is incremental, memory-efficient, and the
    output QDQ ONNX is what TensorRT consumes most cleanly.

This script:
  1. Loads the fp32 ONNX
  2. Walks the teacher cache for calibration data (reuses the same 256 samples
     we have for ORT-attempted-and-failed and for the eventual TRT calibrator).
  3. Calls modelopt's quantize() with INT8 QDQ format.
  4. Writes a self-contained INT8 ONNX with QDQ nodes.

INSTALL (on EC2):
  pip install "nvidia-modelopt[onnx]"
  # or for all backends: pip install "nvidia-modelopt[all]"

VERIFY API (modelopt's surface may shift across versions):
  python -c "from modelopt.onnx.quantization import quantize; help(quantize)"
  python -c "import modelopt.onnx.quantization as q; print(q.__version__ if hasattr(q,'__version__') else 'unknown')"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


# modelopt's CalibrationDataReader interface matches onnxruntime's
# (it deliberately mirrors that contract). Import at runtime so this file is
# loadable even before modelopt install.
def _make_calib_reader(teacher_cache_dir: Path, n_samples: int, input_dtypes: dict):
    try:
        # modelopt re-exports ORT's interface under this path in recent versions
        from modelopt.onnx.quantization.calib_utils import CalibrationDataReader  # type: ignore
    except ImportError:
        # Fallback: modelopt accepts ORT's CalibrationDataReader directly
        from onnxruntime.quantization import CalibrationDataReader  # type: ignore

    class TeacherCacheReader(CalibrationDataReader):
        def __init__(self):
            self.files = sorted(teacher_cache_dir.glob("prompt*_step*.safetensors"))[:n_samples]
            if not self.files:
                raise FileNotFoundError(f"No calibration samples in {teacher_cache_dir}")
            self.idx = 0
            print(f"  CalibrationDataReader: {len(self.files)} samples queued")

        def get_next(self):
            if self.idx >= len(self.files):
                return None
            f = self.files[self.idx]
            self.idx += 1
            s = load_file(str(f), device="cpu")
            feeds = {
                "sample":                s["latent_in"].numpy(),
                "timestep":              s["timestep"].numpy(),
                "encoder_hidden_states": s["prompt_embeds"].numpy(),
                "text_embeds":           s["pooled_embeds"].numpy(),
                "time_ids":              s["time_ids"].numpy(),
            }
            out = {}
            for k, v in feeds.items():
                out[k] = v.astype(input_dtypes[k])
                if k == "timestep" and out[k].ndim == 0:
                    out[k] = out[k][None]
            if self.idx % 32 == 0 or self.idx == len(self.files):
                print(f"  calib feed {self.idx}/{len(self.files)}: {f.name}")
            return out

        def rewind(self):
            self.idx = 0

    return TeacherCacheReader()


_NP_DTYPE = {
    "tensor(float)":  np.float32, "tensor(float16)": np.float16,
    "tensor(int64)":  np.int64,   "tensor(int32)":   np.int32,
}


def probe_input_dtypes(onnx_path: Path) -> dict:
    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return {i.name: _NP_DTYPE[i.type] for i in s.get_inputs()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-onnx", type=Path, required=True)
    ap.add_argument("--output-onnx", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, required=True)
    ap.add_argument("--n-calib", type=int, default=64,
                    help="64 is enough for entropy/percentile on SDXL.")
    ap.add_argument("--calib-method", choices=("entropy", "percentile", "max"),
                    default="entropy",
                    help="modelopt default is 'max'; 'entropy' is recommended for diffusion.")
    ap.add_argument("--op-types", type=str, default="MatMul,Conv,Gemm")
    args = ap.parse_args()

    if not args.input_onnx.exists():
        sys.exit(f"Input ONNX not found: {args.input_onnx}")
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)

    print(f"Probing input dtypes from {args.input_onnx.name}...")
    input_dtypes = probe_input_dtypes(args.input_onnx)
    print(f"  {input_dtypes}")

    print(f"\nBuilding calibration reader ({args.n_calib} samples)...")
    reader = _make_calib_reader(args.teacher_cache, args.n_calib, input_dtypes)

    op_types = [s.strip() for s in args.op_types.split(",") if s.strip()]

    # ------------------------------------------------------------------ modelopt
    # The exact import path / function signature may shift slightly across
    # modelopt versions; this code targets the stable `modelopt.onnx.quantization`
    # surface. If your installed version has moved things, run
    #   python -c "from modelopt.onnx.quantization import quantize; help(quantize)"
    # and tweak the call below to match.
    from modelopt.onnx.quantization import quantize  # type: ignore

    print(f"\nRunning modelopt.onnx.quantize:")
    print(f"  quantize_mode  = int8")
    print(f"  calib method   = {args.calib_method}")
    print(f"  op_types       = {op_types}")
    print(f"  output         = {args.output_onnx}")

    t0 = time.time()
    # Common modelopt API surface (works on recent versions):
    quantize(
        onnx_path=str(args.input_onnx),
        calibration_data_reader=reader,
        calibration_method=args.calib_method,
        quantize_mode="int8",
        op_types_to_quantize=op_types,
        output_path=str(args.output_onnx),
        use_external_data_format=True,
    )
    print(f"\nDone in {time.time() - t0:.0f}s")

    # Report sizes
    def dir_bytes(stem: Path) -> int:
        total = 0
        for p in stem.parent.iterdir():
            if p.name.startswith(stem.name):
                total += p.stat().st_size
        return total

    src = dir_bytes(args.input_onnx.with_suffix(""))
    out = dir_bytes(args.output_onnx.with_suffix(""))
    print(f"\nSize comparison:")
    print(f"  fp32 source: {src/1024/1024:7.0f} MB")
    print(f"  int8 QDQ:    {out/1024/1024:7.0f} MB")
    print(f"  reduction:   {(1 - out/src)*100:.1f}%")
    print(f"\nNext: python mp_quant/eval_onnx_e2e.py --onnx {args.output_onnx} --ep cuda")


if __name__ == "__main__":
    main()
