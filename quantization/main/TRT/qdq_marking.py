"""
Insert explicit QDQ (QuantizeLinear / DequantizeLinear) nodes into fp32 ONNX
using per-tensor scales from a TRT calibration cache.

Quantization scheme
-------------------
- Symmetric INT8, zero_point = 0 for all tensors
- Activations  : scale from TRT calib cache (absolute-max / 127 already encoded)
- Weights      : scale from calib cache 'weight_output' entry if present,
                 else computed on-the-fly as max(abs(weight)) / 127
- Target ops   : MatMul, Gemm, Conv  (same set TRT uses for INT8)

QDQ placement
-------------
For each target node the pattern becomes:

  prev_output ─► Q ─► DQ ─► target_node_input   (input activation)
  weight_init ─► Q ─► DQ ─► target_node_weight   (weight, if initializer)
  target_node_output ─► Q ─► DQ ─► next_nodes     (output activation)

TRT 10 consumes explicit QDQ ONNX directly via the
EXPLICIT_QUANTIZATION path, giving much better INT8 layer coverage
than the implicit calibration path used in the first engine build.

ORT also runs the resulting model via CUDAExecutionProvider with
INT8 kernels (QuantizeLinear / DequantizeLinear are fused away).

Usage
-----
  PYTHONPATH=. python quantization/main/TRT/qdq_marking.py \\
    --input-onnx  models/onnx/unet_fp32.onnx \\
    --output-onnx models/onnx/unet_int8_qdq_explicit.onnx \\
    --calib-cache models/trt/unet_int8_fp16.calib.cache \\
    --op-types MatMul,Gemm,Conv
"""

import argparse
import struct
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ---------------------------------------------------------------------------
# Calib-cache parsing
# ---------------------------------------------------------------------------

def parse_calib_cache(cache_path: Path) -> dict[str, float]:
    """Return {tensor_name: scale_float32}.

    TRT calib cache format:
        TRT-<version>-EntropyCalibration2
        tensor_name: XXXXXXXX   (8 hex chars = IEEE 754 big-endian float32 of the scale)
    """
    scales: dict[str, float] = {}
    with open(cache_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("TRT-"):
                continue
            if ":" not in line:
                continue
            name, hex_val = line.split(":", 1)
            name = name.strip()
            hex_val = hex_val.strip()
            try:
                scale = struct.unpack(">f", bytes.fromhex(hex_val))[0]
                scales[name] = abs(float(scale))
            except Exception:
                pass
    return scales


# ---------------------------------------------------------------------------
# Weight reading from external data
# ---------------------------------------------------------------------------

def _external_info(init: TensorProto) -> dict[str, str]:
    return {e.key: e.value for e in init.external_data}


def read_weight_scale(init: TensorProto, data_dir: Path) -> float:
    """Compute per-tensor symmetric INT8 scale: max(|W|) / 127."""
    if init.data_location != TensorProto.EXTERNAL:
        arr = numpy_helper.to_array(init)
        return float(np.max(np.abs(arr))) / 127.0

    info = _external_info(init)
    data_file = data_dir / info["location"]
    offset = int(info.get("offset", 0))
    length = int(info.get("length", -1))

    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.INT8: np.int8,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
    }
    dtype = dtype_map.get(init.data_type, np.float32)
    itemsize = np.dtype(dtype).itemsize
    n_elem = length // itemsize if length > 0 else -1

    abs_max = 0.0
    chunk = 1 << 22  # 4 M elements per read
    with open(data_file, "rb") as fh:
        fh.seek(offset)
        remaining = n_elem
        while remaining != 0:
            to_read = min(chunk, remaining) if remaining > 0 else chunk
            raw = fh.read(to_read * itemsize)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=dtype)
            abs_max = max(abs_max, float(np.max(np.abs(arr.astype(np.float32)))))
            if remaining > 0:
                remaining -= len(arr)
                if remaining <= 0:
                    break

    return max(abs_max / 127.0, 1e-8)


# ---------------------------------------------------------------------------
# QDQ node construction helpers
# ---------------------------------------------------------------------------

def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def make_qdq_pair(
    input_name: str,
    scale: float,
    name_prefix: str,
) -> tuple[list, list, str]:
    """Return (new_nodes, new_initializers, dq_output_name).

    Inserts:
        input_name ─► QuantizeLinear ─► DequantizeLinear ─► dq_output_name
    Both use symmetric INT8 (zero_point = 0).
    """
    scale_name = _unique(f"{name_prefix}_scale")
    zp_name = _unique(f"{name_prefix}_zp")
    q_out = _unique(f"{name_prefix}_quant")
    dq_out = _unique(f"{name_prefix}_dequant")

    scale_init = numpy_helper.from_array(
        np.array(scale, dtype=np.float32), name=scale_name
    )
    zp_init = numpy_helper.from_array(
        np.array(0, dtype=np.int8), name=zp_name
    )

    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[input_name, scale_name, zp_name],
        outputs=[q_out],
        name=_unique(f"{name_prefix}_Q"),
    )
    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=[q_out, scale_name, zp_name],
        outputs=[dq_out],
        name=_unique(f"{name_prefix}_DQ"),
    )

    return [q_node, dq_node], [scale_init, zp_init], dq_out


# ---------------------------------------------------------------------------
# Main QDQ insertion
# ---------------------------------------------------------------------------

TARGET_OPS = {"MatMul", "Gemm", "Conv"}


_PASSTHROUGH_OPS = {"Reshape", "Transpose", "Cast", "Identity", "Squeeze", "Unsqueeze", "Constant"}


def _is_initializer_derived(name: str, init_map: dict, producer: dict, depth: int = 6) -> bool:
    """True if `name` is an initializer, a Constant output, or traces back to one
    through passthrough ops (Reshape/Transpose/Cast/Identity/Squeeze/Unsqueeze)."""
    if name in init_map:
        return True
    if depth <= 0 or name not in producer:
        return False
    node = producer[name]
    if node.op_type == "Constant":
        return True
    if node.op_type not in _PASSTHROUGH_OPS:
        return False
    return any(_is_initializer_derived(inp, init_map, producer, depth - 1)
               for inp in node.input if inp)


def insert_qdq(
    model: onnx.ModelProto,
    scales: dict[str, float],
    data_dir: Path,
    op_types: set[str],
    insert_output_qdq: bool = True,
    matmul_mode: str = "all",  # "all" | "linear-only" | "bmm-only"
    quantize_weights: bool = False,
    insert_activation_qdq: bool = True,
    include_substrings: list = None,
    exclude_substrings: list = None,
    activation_scale_multiplier: float = 1.0,
) -> onnx.ModelProto:
    graph = model.graph

    # Graph-level output tensor names — must never be renamed or quantized
    graph_outputs: set[str] = {o.name for o in graph.output}

    # Index initializers by name
    init_map: dict[str, TensorProto] = {i.name: i for i in graph.initializer}

    # Map tensor_name -> producing node
    producer: dict = {}
    for n in graph.node:
        for o in n.output:
            if o:
                producer[o] = n
    matmul_skipped_linear = matmul_skipped_bmm = 0

    # Build a map: old_output_name -> new_output_name (after QDQ insertion on node outputs)
    # We need this so downstream consumers get the DQ output, not the original.
    output_remap: dict[str, str] = {}

    new_nodes: list = []
    extra_inits: list = []

    total = sum(1 for n in graph.node if n.op_type in op_types)
    done = 0

    skipped_by_filter = 0
    for node in graph.node:
        if node.op_type in op_types:
            nm = node.name or ""
            if include_substrings and not any(s in nm for s in include_substrings):
                skipped_by_filter += 1
                for i, inp in enumerate(node.input):
                    if inp in output_remap:
                        node.input[i] = output_remap[inp]
                new_nodes.append(node)
                continue
            if exclude_substrings and any(s in nm for s in exclude_substrings):
                skipped_by_filter += 1
                for i, inp in enumerate(node.input):
                    if inp in output_remap:
                        node.input[i] = output_remap[inp]
                new_nodes.append(node)
                continue
        if node.op_type not in op_types:
            # Remap inputs in case an upstream node's output was replaced
            for i, inp in enumerate(node.input):
                if inp in output_remap:
                    node.input[i] = output_remap[inp]
            new_nodes.append(node)
            continue

        # Filter MatMul by mode: distinguish linear-style (input[1] initializer-derived)
        # from BMM-style (both inputs dynamic activations).
        if node.op_type == "MatMul" and matmul_mode != "all" and len(node.input) >= 2:
            is_linear = _is_initializer_derived(node.input[1], init_map, producer)
            if matmul_mode == "linear-only" and not is_linear:
                matmul_skipped_bmm += 1
                for i, inp in enumerate(node.input):
                    if inp in output_remap:
                        node.input[i] = output_remap[inp]
                new_nodes.append(node)
                continue
            if matmul_mode == "bmm-only" and is_linear:
                matmul_skipped_linear += 1
                for i, inp in enumerate(node.input):
                    if inp in output_remap:
                        node.input[i] = output_remap[inp]
                new_nodes.append(node)
                continue

        done += 1
        if done % 100 == 0 or done == 1:
            print(f"  [{done}/{total}] processing {node.op_type} nodes...")

        # --- remap inputs from prior output insertions ---
        for i, inp in enumerate(node.input):
            if inp in output_remap:
                node.input[i] = output_remap[inp]

        # --- quantize each input ---
        for inp_idx, inp_name in enumerate(list(node.input)):
            if not inp_name:
                continue  # optional Gemm bias can be empty

            prefix = f"qdq_{node.name or _unique('node')}_inp{inp_idx}"

            if inp_name in init_map:
                if not quantize_weights:
                    continue
                w_scale = read_weight_scale(init_map[inp_name], data_dir)
                qdq_nodes, qdq_inits, dq_out = make_qdq_pair(inp_name, w_scale, prefix)
                new_nodes.extend(qdq_nodes)
                extra_inits.extend(qdq_inits)
                node.input[inp_idx] = dq_out
                continue

            else:
                if not insert_activation_qdq:
                    continue
                # Activation path — use calib cache scale
                if inp_name not in scales:
                    continue  # no scale available; leave fp32
                act_scale = scales[inp_name] * activation_scale_multiplier
                qdq_nodes, qdq_inits, dq_out = make_qdq_pair(inp_name, act_scale, prefix)
                new_nodes.extend(qdq_nodes)
                extra_inits.extend(qdq_inits)
                node.input[inp_idx] = dq_out

        new_nodes.append(node)

        # --- quantize output (skip graph-level outputs — scheduler needs fp32) ---
        if insert_output_qdq:
            for out_idx, out_name in enumerate(list(node.output)):
                if not out_name or out_name not in scales:
                    continue
                if out_name in graph_outputs:
                    continue  # never quantize the final network output tensor
                out_scale = scales[out_name]
                prefix = f"qdq_{node.name or _unique('node')}_out{out_idx}"
                qdq_nodes, qdq_inits, dq_out = make_qdq_pair(out_name, out_scale, prefix)
                new_nodes.extend(qdq_nodes)
                extra_inits.extend(qdq_inits)
                output_remap[out_name] = dq_out

    # Remap graph outputs (if any were quantized)
    for out in graph.output:
        if out.name in output_remap:
            out.name = output_remap[out.name]

    # Rebuild graph node list
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(extra_inits)

    if matmul_mode != "all":
        print(f"[qdq]   MatMul filter ({matmul_mode}): "
              f"skipped {matmul_skipped_bmm} BMM-style, {matmul_skipped_linear} linear-style")
    if include_substrings or exclude_substrings:
        print(f"[qdq]   block filter: skipped {skipped_by_filter} nodes by name substring")
    return model


# ---------------------------------------------------------------------------
# Save with external data
# ---------------------------------------------------------------------------

def save_model(model: onnx.ModelProto, output_onnx: Path) -> None:
    """Save the modified graph, preserving external-data references to the
    original weight file.  New tiny initializers (scale / zero_point) are
    inlined in the .onnx protobuf.  The caller does NOT need to copy the
    9.6 GB unet_fp32.onnx_data — the QDQ model references it by filename."""
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] writing {output_onnx}")
    print(f"[save] original weights referenced via external_data (no copy needed)")
    onnx.save(model, str(output_onnx))
    onnx_sz = output_onnx.stat().st_size / 1e6
    print(f"[save] graph file: {onnx_sz:.1f} MB")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-onnx",  default="models/onnx/unet_fp32.onnx")
    ap.add_argument("--output-onnx", default="models/onnx/unet_int8_qdq_explicit.onnx")
    ap.add_argument("--calib-cache", default="models/trt/unet_int8_fp16.calib.cache")
    ap.add_argument("--op-types",    default="MatMul,Gemm,Conv")
    ap.add_argument("--no-weight-cache", action="store_true",
                    help="Always compute weight scales from tensor data (ignore calib cache weight_output entries)")
    ap.add_argument("--no-output-qdq", action="store_true",
                    help="Skip Q/DQ on op outputs — only insert input-activation Q/DQ (standard QDQ pattern)")
    ap.add_argument("--matmul-mode", choices=["all", "linear-only", "bmm-only"], default="all",
                    help="For MatMul: 'linear-only' = only if input[1] is initializer-derived (skip attention BMM); 'bmm-only' = only attention BMM")
    ap.add_argument("--quantize-weights", action="store_true",
                    help="Insert Q/DQ on weight initializers (per-tensor max/127). Off by default.")
    ap.add_argument("--no-activation-qdq", action="store_true",
                    help="Skip Q/DQ on activation inputs. Useful for ablation (weight-only QDQ).")
    ap.add_argument("--include-substrings", default="",
                    help="Comma-separated substrings; only quantize nodes whose name contains one of them.")
    ap.add_argument("--exclude-substrings", default="",
                    help="Comma-separated substrings; skip nodes whose name contains one of them.")
    ap.add_argument("--activation-scale-multiplier", type=float, default=1.0,
                    help="Multiply every activation scale by this factor (widens int8 range to avoid outlier clipping).")
    args = ap.parse_args()

    calib_cache = Path(args.calib_cache)
    input_onnx  = Path(args.input_onnx)
    output_onnx = Path(args.output_onnx)
    op_types    = set(args.op_types.split(","))
    data_dir    = input_onnx.parent  # external data lives next to the .onnx

    print(f"[calib] parsing {calib_cache}")
    scales = parse_calib_cache(calib_cache)
    print(f"[calib] {len(scales)} tensor scales loaded")

    print(f"[onnx]  loading graph structure from {input_onnx} (no external data)")
    model = onnx.load(str(input_onnx), load_external_data=False)
    print(f"[onnx]  {len(model.graph.node)} nodes, opset {model.opset_import[0].version}")

    from collections import Counter
    op_count = Counter(n.op_type for n in model.graph.node if n.op_type in op_types)
    print(f"[onnx]  target nodes: {dict(op_count)}")

    print(f"[qdq]   inserting QDQ nodes for {op_types} ...")
    inc = [s for s in args.include_substrings.split(",") if s]
    exc = [s for s in args.exclude_substrings.split(",") if s]
    model = insert_qdq(model, scales, data_dir, op_types,
                       insert_output_qdq=not args.no_output_qdq,
                       matmul_mode=args.matmul_mode,
                       quantize_weights=args.quantize_weights,
                       insert_activation_qdq=not args.no_activation_qdq,
                       include_substrings=inc or None,
                       exclude_substrings=exc or None,
                       activation_scale_multiplier=args.activation_scale_multiplier)

    new_op_count = Counter(n.op_type for n in model.graph.node)
    print(f"[qdq]   QuantizeLinear: {new_op_count['QuantizeLinear']}")
    print(f"[qdq]   DequantizeLinear: {new_op_count['DequantizeLinear']}")

    save_model(model, output_onnx)
    print("[done]")


if __name__ == "__main__":
    main()
