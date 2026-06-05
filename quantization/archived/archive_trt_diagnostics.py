"""TRT diagnostics — engine inspector + alt build modes for Phase 3 root-cause work.

Modes:
  inspect      Dump engine layer-precision summary (counts INT8 / FP16 / FP32).
  build-fp32   Build a FP32-only TRT engine from QDQ ONNX (no FP16, no INT8 flag).
               If MSE is still bad here, the QDQ insertion broke the graph math
               (independent of any quantization).
  build-strong Build with STRONGLY_TYPED network flag, no FP16/INT8 builder flags,
               no PREFER_PRECISION_CONSTRAINTS. Forces TRT to honour the QDQ
               precision annotations strictly.

All builds reuse the input-shape spec from build_trt_engine.py.
"""
from __future__ import annotations
import argparse
import collections
from pathlib import Path

import tensorrt as trt

from build_trt_engine import INPUT_SPEC


def inspect(engine_path: str) -> None:
    import json
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    # Inspector needs an execution context bound to read per-layer info.
    context = engine.create_execution_context()
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context

    info_str = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    data = json.loads(info_str)
    layer_names = data.get("Layers", [])
    print(f"[inspect] engine: {engine_path}")
    print(f"[inspect] total plan layers: {len(layer_names)}")
    try:
        print(f"[inspect] device memory: {engine.device_memory_size/1e6:.1f} MB")
    except Exception:
        pass

    counts = collections.Counter()
    type_prec = collections.Counter()
    matmul_conv_total = collections.Counter()
    parse_fail = 0
    for i in range(len(layer_names)):
        raw = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
        if i < 3:
            print(f"[inspect] sample[{i}] raw={repr(raw)[:400]}")
        try:
            L = json.loads(raw)
        except Exception:
            parse_fail += 1
            continue
        if not isinstance(L, dict):
            parse_fail += 1
            continue
        prec = L.get("Precision") or L.get("PrecisionType") or "unknown"
        lt = L.get("LayerType", "unknown")
        counts[prec] += 1
        type_prec[(lt, prec)] += 1
        if any(tok in lt for tok in ("Conv", "Gemm", "Matrix")):
            matmul_conv_total[prec] += 1
    print(f"[inspect] per-layer parse failures: {parse_fail}")
    print("\n[inspect] precision histogram (all layers):")
    for k, v in counts.most_common():
        print(f"  {k:20s} {v}")
    print("\n[inspect] MatMul/Conv/Gemm-like layers by precision:")
    for k, v in matmul_conv_total.most_common():
        print(f"  {k:20s} {v}")
    print("\n[inspect] top (LayerType, Precision) pairs:")
    for (lt, prec), v in type_prec.most_common(15):
        print(f"  {lt:35s} {prec:10s} {v}")


def _add_static_profile(builder, config):
    profile = builder.create_optimization_profile()
    for name, (_, _, shape) in INPUT_SPEC.items():
        profile.set_shape(name, shape, shape, shape)
    config.add_optimization_profile(profile)


def build(mode: str, input_onnx: str, engine_path: str, workspace_gb: int) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)

    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if mode == "build-strong":
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        print("[trt] network flag: STRONGLY_TYPED")
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, logger)
    print(f"[trt] parsing {input_onnx}")
    if not parser.parse_from_file(input_onnx):
        for i in range(parser.num_errors):
            print(f"  parse err[{i}]: {parser.get_error(i)}")
        raise SystemExit(1)
    print(f"[trt] parsed: {network.num_layers} layers")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if mode == "build-fp32":
        # No FP16, no INT8 — pure fp32 execution of the QDQ graph.
        # Q/DQ nodes will execute (introducing int8 rounding via scale) but
        # all kernels run in fp32. Tests whether the QDQ insertion itself
        # corrupted the graph math.
        print("[trt] mode: FP32-only (no FP16, no INT8 flag)")
    elif mode == "build-strong":
        # Strongly-typed: TRT respects QDQ annotations strictly. Do NOT set
        # FP16/INT8 builder flags — strongly-typed mode rejects them.
        print("[trt] mode: STRONGLY_TYPED (no builder precision flags)")
    else:
        raise ValueError(f"unknown build mode: {mode}")

    _add_static_profile(builder, config)

    print("[trt] building engine (this may take 30-90 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")
    out = Path(engine_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(serialized)
    print(f"[trt] engine saved {out} ({out.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["inspect", "build-fp32", "build-strong"])
    ap.add_argument("--input-onnx")
    ap.add_argument("--engine-path")
    ap.add_argument("--workspace-gb", type=int, default=12)
    args = ap.parse_args()
    if args.mode == "inspect":
        if not args.engine_path:
            raise SystemExit("--engine-path required for inspect")
        inspect(args.engine_path)
    else:
        if not (args.input_onnx and args.engine_path):
            raise SystemExit("--input-onnx and --engine-path required for build modes")
        build(args.mode, args.input_onnx, args.engine_path, args.workspace_gb)


if __name__ == "__main__":
    main()
