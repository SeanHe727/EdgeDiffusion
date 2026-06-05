#!/usr/bin/env python3
"""Remove dead (unreferenced) initializers from an ONNX file.

After onnxruntime.quantization.quantize_static produces a QDQ-format model,
the original fp32 weight initializers of quantized ops are typically left in
the file even though the graph no longer references them (they're shadowed by
the new int8 weight + DequantizeLinear chain). This bloats the on-disk size
without contributing to inference.

This script:
  1. Walks every node, collects the set of tensor names actually consumed
     as inputs (also recurses into subgraphs / control-flow ops).
  2. Compares against `graph.initializer` list.
  3. Drops any initializer NOT in the consumed set.
  4. Re-saves the cleaned ONNX with consolidated external data.

Expected reduction on QDQ-quantized SDXL UNet: ~75-85% file size shrink.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import onnx
from onnx import GraphProto


def collect_used_tensor_names(graph: GraphProto) -> set[str]:
    """Walk all nodes in the graph (including subgraphs of If/Loop/etc.) and
    return the set of every tensor name consumed as a node input."""
    used = set()
    for node in graph.node:
        for inp in node.input:
            if inp:
                used.add(inp)
        # Recurse into nested subgraphs (If/Loop control-flow nodes carry them as attributes)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                used |= collect_used_tensor_names(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    used |= collect_used_tensor_names(sub)
    # Outputs of subgraph are also "used" downstream — include them too
    for output in graph.output:
        used.add(output.name)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-onnx", type=Path, required=True)
    ap.add_argument("--output-onnx", type=Path, required=True)
    args = ap.parse_args()

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input_onnx} (with external data)...")
    t0 = time.time()
    model = onnx.load(str(args.input_onnx), load_external_data=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"\nCollecting used tensor names from {len(model.graph.node)} nodes...")
    used = collect_used_tensor_names(model.graph)
    print(f"  {len(used)} unique tensor names referenced by nodes")

    print(f"\nFiltering {len(model.graph.initializer)} initializers...")
    keep, drop = [], []
    kept_bytes = dropped_bytes = 0
    type_names = {1:"fp32", 2:"uint8", 3:"int8", 6:"int32", 7:"int64", 10:"fp16"}
    drop_by_dtype: dict[int, int] = {}
    keep_by_dtype: dict[int, int] = {}

    for init in model.graph.initializer:
        # An initializer is alive if its name is consumed by some node
        if init.name in used:
            keep.append(init)
            kept_bytes += len(init.raw_data) if init.raw_data else 0
            keep_by_dtype[init.data_type] = keep_by_dtype.get(init.data_type, 0) + 1
        else:
            drop.append(init.name)
            dropped_bytes += len(init.raw_data) if init.raw_data else 0
            drop_by_dtype[init.data_type] = drop_by_dtype.get(init.data_type, 0) + 1

    print(f"  KEEP: {len(keep):>5} initializers, {kept_bytes/1024/1024:7.1f} MB")
    for dt, cnt in sorted(keep_by_dtype.items(), key=lambda x: -x[1]):
        print(f"    {type_names.get(dt, str(dt)):<8} {cnt}")
    print(f"  DROP: {len(drop):>5} initializers, {dropped_bytes/1024/1024:7.1f} MB")
    for dt, cnt in sorted(drop_by_dtype.items(), key=lambda x: -x[1]):
        print(f"    {type_names.get(dt, str(dt)):<8} {cnt}")

    if drop:
        print(f"  Sample dropped names: {drop[:5]}")

    # Rebuild initializer list in-place
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

    # Save with consolidated external data
    print(f"\nSaving cleaned ONNX → {args.output_onnx}")
    data_path = args.output_onnx.parent / (args.output_onnx.stem + ".onnx_data")
    onnx.save_model(
        model, str(args.output_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        convert_attribute=False,
    )

    # Report final size
    final_bytes = args.output_onnx.stat().st_size + data_path.stat().st_size
    in_data = args.input_onnx.parent / (args.input_onnx.stem + ".onnx_data")
    src_bytes = args.input_onnx.stat().st_size + (in_data.stat().st_size if in_data.exists() else 0)
    print(f"\nSize comparison:")
    print(f"  Input  (with dead init):  {src_bytes  /1024/1024:7.1f} MB")
    print(f"  Output (cleaned):         {final_bytes/1024/1024:7.1f} MB")
    print(f"  Reduction:                {(1 - final_bytes/src_bytes)*100:.1f}%")


if __name__ == "__main__":
    main()
