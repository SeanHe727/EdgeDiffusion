"""Build TRT INT8+FP16 engine for SDXL UNet from fp32 ONNX.

Pipeline:
  1. Parse onnx via OnnxParser
  2. Calibrate using teacher cache (IInt8EntropyCalibrator2)
  3. Build serialized engine with FP16+INT8 flags, save to disk

Run:
  PYTHONPATH=/opt/ebs/py_extra LD_LIBRARY_PATH=/opt/ebs/py_extra/tensorrt_libs:$LD_LIBRARY_PATH \\
    /opt/pytorch/bin/python -u quantization/main/TRT/build_trt_engine.py
"""
import argparse
import os
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from safetensors.torch import load_file


# Maps ONNX input name -> (teacher cache key, numpy dtype, expected shape)
INPUT_SPEC = {
    "sample":                ("latent_in",     np.float32, (1, 4, 128, 128)),
    "timestep":              ("timestep",      np.int64,   (1,)),
    "encoder_hidden_states": ("prompt_embeds", np.float32, (1, 77, 2048)),
    "text_embeds":           ("pooled_embeds", np.float32, (1, 1280)),
    "time_ids":              ("time_ids",      np.float32, (1, 6)),
}


def _np_dtype_to_torch(dt):
    return {np.float32: torch.float32, np.int64: torch.int64}[dt]


class TeacherCacheCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_dir: Path, n: int, cache_file: Path):
        super().__init__()
        self._files = sorted(cache_dir.glob("prompt*_step*.safetensors"))[:n]
        if not self._files:
            raise RuntimeError(f"No calibration samples in {cache_dir}")
        self._idx = 0
        self._cache_file = cache_file
        # Pre-allocate device buffers (one per ONNX input)
        self._dev = {}
        for name, (_, dtype, shape) in INPUT_SPEC.items():
            self._dev[name] = torch.empty(
                shape, dtype=_np_dtype_to_torch(dtype), device="cuda"
            )
        # Hold a python ref to the int list so it lives across get_batch returns
        self._last_ptrs = None
        print(f"[calib] {len(self._files)} samples, cache={cache_file}")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self._idx >= len(self._files):
            return None
        sample = load_file(str(self._files[self._idx]), device="cpu")
        self._idx += 1
        if self._idx % 8 == 0 or self._idx == 1:
            print(f"[calib] {self._idx}/{len(self._files)}")
        for onnx_name, (src_key, np_dt, shape) in INPUT_SPEC.items():
            t = sample[src_key]
            if onnx_name == "timestep":
                # scalar fp32 in cache -> int64 [1] for ONNX
                arr = np.array([t.item()], dtype=np.int64)
                self._dev[onnx_name].copy_(torch.from_numpy(arr).cuda())
            else:
                # cache is fp16, ONNX expects fp32
                self._dev[onnx_name].copy_(t.to(torch.float32).cuda())
        # Return pointers in the order TRT requested
        self._last_ptrs = [int(self._dev[n].data_ptr()) for n in names]
        return self._last_ptrs

    def read_calibration_cache(self):
        if self._cache_file.exists():
            print(f"[calib] reusing cache {self._cache_file}")
            return self._cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_bytes(cache)
        print(f"[calib] wrote cache {self._cache_file} ({len(cache)} B)")


def build(args):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    print(f"[trt] parsing {args.input_onnx}")
    if not parser.parse_from_file(args.input_onnx):
        for i in range(parser.num_errors):
            print(f"  parse err[{i}]: {parser.get_error(i)}")
        raise SystemExit(1)
    print(f"[trt] parsed: {network.num_layers} layers")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30
    )
    # DETAILED profiling verbosity lets EngineInspector return per-layer
    # precision after build. Without it, inspector returns only layer names.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.FP16)

    if args.qdq:
        # Explicit quantization path: QDQ nodes in the ONNX carry the scales.
        # INT8 flag enables INT8 kernels; no calibrator is attached.
        # PREFER_PRECISION_CONSTRAINTS tells TRT to honour the Q/DQ precision
        # annotations rather than silently promoting layers to FP16.
        config.set_flag(trt.BuilderFlag.INT8)
        print("[trt] mode: EXPLICIT_QUANTIZATION (QDQ ONNX, no calibrator)")
    elif args.int8:
        config.set_flag(trt.BuilderFlag.INT8)
        cache_file = Path(args.engine_path).with_suffix(".calib.cache")
        config.int8_calibrator = TeacherCacheCalibrator(
            Path(args.teacher_cache), args.n_calib, cache_file
        )
        print("[trt] mode: IMPLICIT_QUANTIZATION (calibrator)")

    # Static shapes -> single optimization profile with min=opt=max
    profile = builder.create_optimization_profile()
    for name, (_, _, shape) in INPUT_SPEC.items():
        profile.set_shape(name, shape, shape, shape)
    config.add_optimization_profile(profile)
    if args.int8 and not args.qdq:
        config.set_calibration_profile(profile)

    print("[trt] building engine (this can take 30-90 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    out = Path(args.engine_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(serialized)
    print(f"[trt] engine saved {out} ({out.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-onnx", default="models/onnx/unet_fp32.onnx")
    ap.add_argument("--engine-path", default="models/trt/unet_int8_fp16.engine")
    ap.add_argument("--teacher-cache", default="qlora_teacher_cache_128p_1024")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--workspace-gb", type=int, default=12)
    ap.add_argument("--int8", action="store_true", default=True)
    ap.add_argument("--no-int8", dest="int8", action="store_false")
    ap.add_argument("--qdq", action="store_true", default=False,
                    help="Explicit-quantization mode: read scales from QDQ nodes, no calibrator")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
