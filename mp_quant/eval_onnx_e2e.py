#!/usr/bin/env python3
"""Phase B end-to-end gate — replace diffusers' UNet with an ORT-backed wrapper
that calls our exported ONNX, then generate 4 images at 1024x1024 / 4 steps and
compare to the existing FP16 baseline images by per-pixel MSE.

The single-step verify (verify_onnx_vs_pytorch.py) showed max abs diff ≈ 0.014
on noise_pred. The real "does it work?" test is full image-space MSE:
  Pass condition: MSE close to packed+LoRA baseline (≈ 0.00794)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from PIL import Image


_ORT_DTYPE_MAP = {
    "tensor(float16)": np.float16, "tensor(float)": np.float32,
    "tensor(int64)":   np.int64,   "tensor(int32)": np.int32,
}


class ORTUNetWrapper(nn.Module):
    """Drop-in replacement for UNet2DConditionModel that runs forward via an ONNX
    Runtime session. Wraps tensor↔numpy + dtype conversion at the boundary."""

    def __init__(self, onnx_path: Path, reference_unet_config_dict: dict,
                 base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = "cuda",
                 ep: str = "cuda",
                 disable_optimizer: bool = False):
        super().__init__()
        import onnxruntime as ort
        so = ort.SessionOptions()
        if disable_optimizer:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        else:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if ep == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        elif ep == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = [ep, "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        self.input_dtypes = {i.name: _ORT_DTYPE_MAP[i.type] for i in self.session.get_inputs()}
        # diffusers SDXL pipeline pokes at .config, .device, .dtype, .add_embedding.linear_1.in_features
        from diffusers.configuration_utils import FrozenDict
        self.config = FrozenDict(reference_unet_config_dict)
        self._device = torch.device(device)
        self._dtype = torch.float16
        # Stub add_embedding.linear_1.in_features so SDXL pipeline's
        # _get_add_time_ids dim-check passes.
        # For SDXL: addition_time_embed_dim (256) * 6 + projection_dim (1280) = 2816
        addition_time_embed_dim = reference_unet_config_dict.get("addition_time_embed_dim", 256)
        # 6 = len([orig_h, orig_w, crop_top, crop_left, target_h, target_w])
        # 1280 = text_encoder_2 projection_dim
        in_features_expected = addition_time_embed_dim * 6 + 1280
        from types import SimpleNamespace
        self.add_embedding = SimpleNamespace(
            linear_1=SimpleNamespace(in_features=in_features_expected)
        )

    @property
    def device(self): return self._device
    @property
    def dtype(self): return self._dtype

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, return_dict=True, **kwargs):
        if added_cond_kwargs is None:
            added_cond_kwargs = {}
        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids    = added_cond_kwargs["time_ids"]
        # Normalize timestep to a 1-element int64 tensor
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([int(timestep)], dtype=torch.int64)
        ts_cpu = timestep.detach().cpu().numpy()
        if ts_cpu.ndim == 0:
            ts_cpu = ts_cpu[None]
        feeds = {
            "sample":                sample.detach().cpu().numpy().astype(self.input_dtypes["sample"]),
            "timestep":              ts_cpu.astype(self.input_dtypes["timestep"]),
            "encoder_hidden_states": encoder_hidden_states.detach().cpu().numpy().astype(self.input_dtypes["encoder_hidden_states"]),
            "text_embeds":           text_embeds.detach().cpu().numpy().astype(self.input_dtypes["text_embeds"]),
            "time_ids":              time_ids.detach().cpu().numpy().astype(self.input_dtypes["time_ids"]),
        }
        out = self.session.run(["noise_pred"], feeds)[0]
        out_t = torch.from_numpy(out).to(self._device, self._dtype)
        if return_dict:
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
            return UNet2DConditionOutput(sample=out_t)
        return (out_t,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Dir with fp16_*.png + selected_prompts.json (FP16 baseline images).")
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--prompt-count", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--ep", choices=("cuda", "cpu"), default="cuda",
                    help="ONNX Runtime execution provider.")
    ap.add_argument("--disable-optimizer", action="store_true",
                    help="Disable ORT graph optimizations (workaround for INT8 QDQ CUDA EP issues).")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")    # force legacy HTTP -> goes to HF_HOME (D:)
    os.environ.setdefault("TMPDIR", str(args.repo_root / ".tmp"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.repo_root / ".tmp").mkdir(parents=True, exist_ok=True)
    device = "cuda"; dtype = torch.float16

    # Get a reference UNet config dict to seed our wrapper's .config
    print("Reading SDXL UNet config to seed ORTUNetWrapper.config...")
    ref_unet_cfg = UNet2DConditionModel.load_config(args.base_model, subfolder="unet")

    print(f"Building ORTUNetWrapper from {args.onnx.name}  (EP={args.ep}, disable_opt={args.disable_optimizer})...")
    ort_unet = ORTUNetWrapper(args.onnx, ref_unet_cfg, args.base_model, device=device,
                              ep=args.ep, disable_optimizer=args.disable_optimizer)
    print(f"  ORT input dtypes: {ort_unet.input_dtypes}")
    print(f"  ORT actual EPs:   {ort_unet.session.get_providers()}")

    print("Building SDXL pipeline with ORT-backed UNet...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.unet = ort_unet
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    sp = args.baseline_dir / "selected_prompts.json"
    if not sp.exists():
        sys.exit(f"Missing {sp}")
    items = json.loads(sp.read_text(encoding="utf-8"))[: args.prompt_count]

    print(f"\nGenerating {len(items)} images via ONNX RT (fp32 graph, fp16 pipeline)...")
    metrics = []
    for it in items:
        idx, seed, prompt, src = it["index"], it["seed"], it["prompt"], it["source"]
        gen = torch.Generator(device=device).manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                       height=args.height, width=args.width, generator=gen).images[0]
        elapsed = time.time() - t0
        out_fname = args.output_dir / f"ort_e2e_{idx:02d}_seed{seed}_{Path(src).stem}.png"
        img.save(out_fname)
        fp16_cands = list(args.baseline_dir.glob(f"fp16_{idx:02d}_seed{seed}_*.png"))
        line = f"  [{idx:02d}/{len(items)}] {elapsed:.1f}s  {prompt[:55]}"
        if fp16_cands:
            ref = np.asarray(Image.open(fp16_cands[0]).convert("RGB"), dtype=np.float32) / 255.0
            cur = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            mse = float(np.mean((ref - cur) ** 2))
            mae = float(np.mean(np.abs(ref - cur)))
            metrics.append({"index": idx, "mse": mse, "mae": mae})
            line += f"  mse_vs_fp16={mse:.5f}"
        print(line)

    if metrics:
        mses = [m["mse"] for m in metrics]
        print(f"\n=== ORT end-to-end vs FP16 baseline ({len(metrics)} images) ===")
        print(f"  MSE mean: {np.mean(mses):.5f}  (target ≈ 0.00794 = packed+LoRA on PyTorch)")
        print(f"  MSE max : {np.max(mses):.5f}")
        delta = float(np.mean(mses)) - 0.00794
        if abs(delta) < 0.001:
            print(f"  PASS — within ±0.001 of expected. ONNX pipeline is deployment-grade.")
        elif abs(delta) < 0.003:
            print(f"  ~OK — delta {delta:+.5f}. Small drift, acceptable; proceed to Phase C.")
        else:
            print(f"  WARN — delta {delta:+.5f}. Larger than expected.")
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
