"""Percentile-based activation calibration for SDXL UNet ONNX.

For each tensor that the existing TRT entropy-calib cache covers, recompute
the scale using p99.9 of the absolute-value distribution observed on a small
set of calibration samples. Writes a new TRT-format calib cache.

This is the "one last try" replacement for TRT entropy calibration, which we
found over-clips outliers and degrades hard prompts.
"""
from __future__ import annotations
import argparse
import gc
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from safetensors.torch import load_file


INPUT_SPEC = {
    "sample":                ("latent_in",     np.float32),
    "timestep":              ("timestep",      np.int64),
    "encoder_hidden_states": ("prompt_embeds", np.float32),
    "text_embeds":           ("pooled_embeds", np.float32),
    "time_ids":              ("time_ids",      np.float32),
}


def build_feed(sample):
    feed = {}
    for onnx_name, (src, dt) in INPUT_SPEC.items():
        t = sample[src]
        if onnx_name == "timestep":
            feed[onnx_name] = np.array([t.item()], dtype=np.int64)
        else:
            feed[onnx_name] = t.to_dense().to(dtype=__import__("torch").float32).cpu().numpy().astype(dt, copy=False)
    return feed


def parse_calib_cache_names(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("TRT-"):
            continue
        if ":" in line:
            out.append(line.split(":", 1)[0].strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-onnx", default="models/onnx/unet_fp32.onnx")
    ap.add_argument("--calib-cache-in", default="models/trt/unet_int8_fp16.calib.cache",
                    help="Existing TRT calib cache; we re-scale the same tensors")
    ap.add_argument("--calib-cache-out", default="models/trt/unet_p999.calib.cache")
    ap.add_argument("--teacher-cache", default="qlora_teacher_cache_128p_1024")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=120)
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--target-ops", default="MatMul,Gemm,Conv")
    ap.add_argument("--tmp-dir", default=None,
                    help="Where to write the augmented ONNX graph. Defaults to the same dir as input-onnx so external_data resolves natively.")
    args = ap.parse_args()

    target_ops = set(args.target_ops.split(","))
    in_onnx = Path(args.input_onnx)
    cache_names = set(parse_calib_cache_names(Path(args.calib_cache_in)))
    print(f"[calib] cache has {len(cache_names)} tensor scales")

    # Load model (graph only) to find which tensor names are produced by target ops as inputs
    m = onnx.load(str(in_onnx), load_external_data=False)
    producer_outputs = set()
    for n in m.graph.node:
        for o in n.output:
            if o:
                producer_outputs.add(o)

    # Index initializers + map tensor -> producing node (for linear-vs-BMM detection)
    init_names = {i.name for i in m.graph.initializer}
    producer = {o: n for n in m.graph.node for o in n.output if o}
    passthrough = {"Reshape", "Transpose", "Cast", "Identity", "Squeeze", "Unsqueeze", "Constant"}

    def is_init_derived(name, depth=6):
        if name in init_names:
            return True
        if depth <= 0 or name not in producer:
            return False
        nd = producer[name]
        if nd.op_type == "Constant":
            return True
        if nd.op_type not in passthrough:
            return False
        return any(is_init_derived(i, depth - 1) for i in nd.input if i)

    # Collect input activation tensors of Conv + linear-style MatMul + Gemm (matches B1-c scope).
    needed = set()
    for n in m.graph.node:
        if n.op_type not in target_ops:
            continue
        if n.op_type == "MatMul" and len(n.input) >= 2 and not is_init_derived(n.input[1]):
            continue  # BMM-style; B1-c skips these
        for inp in n.input:
            if inp and inp not in init_names:
                needed.add(inp)
    targets = sorted(needed & cache_names & set(producer_outputs))
    print(f"[calib] {len(needed)} candidate activation tensors, {len(targets)} also in cache+intermediate")

    # Load calib samples
    files = sorted(Path(args.teacher_cache).glob("prompt*_step*.safetensors"))[:args.n_samples]
    print(f"[calib] loading {len(files)} teacher cache samples")
    samples = [load_file(str(f), device="cpu") for f in files]
    feeds = [build_feed(s) for s in samples]

    # Write aug ONNX into the same dir as the input ONNX so external_data resolves natively.
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else in_onnx.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p_max_per_tensor: dict[str, float] = {}
    chunks = [targets[i:i + args.chunk_size] for i in range(0, len(targets), args.chunk_size)]
    print(f"[calib] {len(chunks)} chunks of {args.chunk_size} tensors")

    for ci, chunk_names in enumerate(chunks):
        # Reload graph (cheap, no external data) and add chunk_names as outputs
        m = onnx.load(str(in_onnx), load_external_data=False)
        # Drop existing graph outputs except the original (keep noise_pred)
        # Actually we keep originals AND append chunk outputs
        for name in chunk_names:
            vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            m.graph.output.append(vi)
        aug_path = tmp_dir / f"aug_chunk_{ci}.onnx"
        onnx.save(m, str(aug_path), save_as_external_data=False)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(
            str(aug_path), sess_options=so,
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )

        out_names = [o.name for o in sess.get_outputs()]
        wanted = set(chunk_names)
        per_sample_p: dict[str, list] = {n: [] for n in chunk_names}
        for si, feed in enumerate(feeds):
            outs = sess.run(out_names, feed)
            for n, v in zip(out_names, outs):
                if n in wanted and v.size > 0:
                    a = np.abs(v).ravel()
                    p = float(np.percentile(a, args.percentile))
                    per_sample_p[n].append(p)
            print(f"  chunk {ci+1}/{len(chunks)} sample {si+1}/{len(feeds)}")

        for n, lst in per_sample_p.items():
            if lst:
                # max of per-sample p99.9 is conservative
                p_max_per_tensor[n] = max(p_max_per_tensor.get(n, 0.0), max(lst))

        del sess; gc.collect()
        aug_path.unlink(missing_ok=True)

    # Merge with original cache: tensors we couldn't recalibrate keep their original scales
    original = {}
    for line in Path(args.calib_cache_in).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("TRT-"):
            continue
        if ":" in line:
            name, hex_val = line.split(":", 1)
            try:
                scale = struct.unpack(">f", bytes.fromhex(hex_val.strip()))[0]
                original[name.strip()] = abs(float(scale))
            except Exception:
                pass

    out_lines = ["TRT-Python-Percentile-99.9"]
    new_count = 0
    for name in original:
        if name in p_max_per_tensor:
            scale = p_max_per_tensor[name] / 127.0
            new_count += 1
        else:
            scale = original[name]
        if scale <= 0:
            scale = 1e-8
        hex_val = struct.pack(">f", float(scale)).hex()
        out_lines.append(f"{name}: {hex_val}")

    Path(args.calib_cache_out).write_text("\n".join(out_lines) + "\n")
    print(f"[calib] wrote {args.calib_cache_out}: {len(original)} tensors total, {new_count} re-scaled with p{args.percentile}")


if __name__ == "__main__":
    main()
