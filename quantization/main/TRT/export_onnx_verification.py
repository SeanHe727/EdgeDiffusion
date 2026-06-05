#!/usr/bin/env python3
"""Phase B verification — load the exported ONNX with onnxruntime CUDA EP and
compare its forward output against PyTorch UNet on the SAME inputs. Pass
condition: max abs diff < 1e-3 (fp16 noise floor).

Inputs are drawn from the cached Q-LoRA teacher trajectories (already
representative SDXL-Lightning denoising trajectories) so this also confirms the
exported model handles realistic inference inputs, not just random tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file


@torch.no_grad()
def pt_forward(unet, sample, dtype=torch.float16):
    """Run PyTorch UNet on a teacher-cache sample at the requested compute dtype."""
    out = unet(
        sample=sample["latent_in"].cuda().to(dtype),
        timestep=sample["timestep"].cuda(),
        encoder_hidden_states=sample["prompt_embeds"].cuda().to(dtype),
        added_cond_kwargs={
            "text_embeds": sample["pooled_embeds"].cuda().to(dtype),
            "time_ids":    sample["time_ids"].cuda().to(dtype),
        },
        return_dict=False,
    )[0]
    return out


_ORT_DTYPE_MAP = {
    "tensor(float16)": np.float16, "tensor(float)": np.float32,
    "tensor(int64)":   np.int64,   "tensor(int32)": np.int32,
}


def ort_forward(session, sample):
    """Run ONNX Runtime, casting each feed to the dtype declared by the ONNX input."""
    name_to_dtype = {i.name: _ORT_DTYPE_MAP[i.type] for i in session.get_inputs()}
    raw = {
        "sample":                sample["latent_in"].numpy(),
        "timestep":              sample["timestep"].numpy(),
        "encoder_hidden_states": sample["prompt_embeds"].numpy(),
        "text_embeds":           sample["pooled_embeds"].numpy(),
        "time_ids":              sample["time_ids"].numpy(),
    }
    feeds = {k: v.astype(name_to_dtype[k]) for k, v in raw.items() if k in name_to_dtype}
    if feeds["timestep"].ndim == 0:
        feeds["timestep"] = feeds["timestep"][None]
    outs = session.run(["noise_pred"], feeds)
    return outs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--merged-safetensors", type=Path, required=True,
                    help="Same Phase A safetensors used for export — needed to reload PyTorch reference.")
    ap.add_argument("--teacher-cache", type=Path, required=True,
                    help="Dir with prompt*_step*.safetensors from qlora_cache_teacher.py")
    ap.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="How many cached samples to verify against.")
    ap.add_argument("--fp32", action="store_true",
                    help="Use fp32 PyTorch compute (for verifying an fp32-exported ONNX).")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

    # ---------------------------------------------------------------- PyTorch reference
    pt_dtype = torch.float32 if args.fp32 else torch.float16
    print(f"Loading PyTorch reference UNet at {pt_dtype} ({args.merged_safetensors.name})...")
    state = load_file(str(args.merged_safetensors), device="cuda")
    unet = UNet2DConditionModel.from_config(args.base_model, subfolder="unet").to("cuda", pt_dtype)
    unet.load_state_dict(state, strict=False)
    unet.eval()
    del state
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- ONNX Runtime
    print(f"Loading ONNX session ({args.onnx.name}) with CUDAExecutionProvider...")
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(str(args.onnx), sess_options=so, providers=providers)
    print(f"  EPs in session: {session.get_providers()}")
    print(f"  Inputs:  {[(i.name, i.shape, i.type) for i in session.get_inputs()]}")
    print(f"  Outputs: {[(o.name, o.shape, o.type) for o in session.get_outputs()]}")

    # ---------------------------------------------------------------- compare
    cache_files = sorted(args.teacher_cache.glob("prompt*_step*.safetensors"))[: args.n_samples]
    if not cache_files:
        sys.exit(f"No prompt*_step*.safetensors in {args.teacher_cache}")
    print(f"\nComparing on {len(cache_files)} teacher-cache samples...")
    print(f"{'sample':<32} {'max_abs_diff':>14} {'mean_abs_diff':>14} {'finite ratio':>14}")
    print("-" * 75)

    diffs_max, diffs_mean = [], []
    for f in cache_files:
        sample = load_file(str(f), device="cpu")
        pt = pt_forward(unet, sample, dtype=pt_dtype).cpu().float().numpy()
        ort_out = ort_forward(session, sample).astype(np.float32)
        finite = float(np.isfinite(ort_out).mean())
        max_d = float(np.max(np.abs(pt - ort_out)))
        mean_d = float(np.mean(np.abs(pt - ort_out)))
        diffs_max.append(max_d); diffs_mean.append(mean_d)
        print(f"{f.name:<32} {max_d:>14.6f} {mean_d:>14.6f} {finite:>14.4f}")

    print("-" * 75)
    print(f"{'OVERALL':<32} {max(diffs_max):>14.6f} {np.mean(diffs_mean):>14.6f}")
    overall = max(diffs_max)
    gate = 1e-3
    print(f"\nGate (fp16 noise floor): max_abs_diff < {gate}")
    if overall < gate:
        print(f"  PASS — overall max abs diff {overall:.5f} < {gate}. ONNX export matches PyTorch.")
    else:
        print(f"  WARN — overall max abs diff {overall:.5f} >= {gate}. Inspect for op miscompilation.")


if __name__ == "__main__":
    main()
