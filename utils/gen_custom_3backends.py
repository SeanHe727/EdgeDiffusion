"""Generate fixed custom prompts across 3 backends on EC2:
   1) FP16 baseline (PyTorch SDXL-Lightning UNet untouched)
   2) Sprint 2 implicit TRT engine
   3) Phase 3 B1-c TRT INT8 engine

Writes one PNG per (prompt, backend) into per-backend output dirs.
The corresponding Phase 2 (PyTorch packed + Q-LoRA) row is produced locally
by `utils/gen_custom_phase2.py`.

Usage on EC2:
  python -u utils/gen_custom_3backends.py \
    --output-dir gen_test_output/custom_showcase \
    --implicit-engine models/trt/unet_int8_fp16.engine \
    --b1c-engine     models/trt/unet_int8_b1c.engine
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from quantization.main.TRT.trt_unet_wrapper import TRTUNetWrapper


LIGHTNING_REPO = "ByteDance/SDXL-Lightning"


def load_lightning_unet(base_model: str, steps: int, dtype, device: str) -> UNet2DConditionModel:
    """Load the SDXL-Lightning N-step distilled UNet weights into a fresh UNet shell."""
    unet = UNet2DConditionModel.from_config(base_model, subfolder="unet").to(device, dtype)
    ckpt_name = f"sdxl_lightning_{steps}step_unet.safetensors"
    ckpt_path = hf_hub_download(LIGHTNING_REPO, ckpt_name)
    state = load_file(ckpt_path, device=device)
    unet.load_state_dict(state)
    return unet


CUSTOM_PROMPTS = [
    {"index": 65, "seed": 650000, "prompt": "Portrait of a blonde woman"},
    {"index": 66, "seed": 660000, "prompt": "A cyberpunk city skyline at night, neon lights"},
    {"index": 67, "seed": 670000, "prompt": "A fluffy cat sitting on a windowsill"},
]


def gen_with_backend(pipe, prompt: str, seed: int, steps: int, h: int, w: int) -> "Image.Image":
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        return pipe(prompt, num_inference_steps=steps, guidance_scale=0.0,
                    height=h, width=w, generator=gen).images[0]


def save(img, out_dir: Path, prefix: str, idx: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"{prefix}_{idx:02d}_seed{seed}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("gen_test_output/custom_showcase"))
    ap.add_argument("--base-model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--implicit-engine", type=Path, required=True)
    ap.add_argument("--b1c-engine", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", "/opt/dlami/nvme/.tmp")
    Path("/opt/dlami/nvme/.tmp").mkdir(parents=True, exist_ok=True)

    print(f"Loading SDXL-Lightning {args.steps}-step UNet (FP16)...")
    lightning_unet = load_lightning_unet(args.base_model, args.steps, torch.float16, "cuda")

    print("Loading SDXL-Lightning pipeline (FP16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, unet=lightning_unet, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    fp16_unet = pipe.unet  # keep ref (= Lightning UNet)

    # 1) FP16 baseline
    print("\n[1/3] FP16 baseline")
    out_fp16 = args.output_dir / "fp16"
    for p in CUSTOM_PROMPTS:
        t0 = time.time()
        img = gen_with_backend(pipe, p["prompt"], p["seed"], args.steps, args.height, args.width)
        save(img, out_fp16, "fp16", p["index"], p["seed"])
        print(f"  idx {p['index']} ({time.time()-t0:.2f}s)  {p['prompt']}")

    ref_unet_cfg = UNet2DConditionModel.load_config(args.base_model, subfolder="unet")

    # 2) Sprint 2 implicit TRT
    print("\n[2/3] Sprint 2 implicit TRT")
    trt_imp = TRTUNetWrapper(args.implicit_engine, ref_unet_cfg, device="cuda")
    pipe.unet = trt_imp
    out_imp = args.output_dir / "trt_implicit"
    for p in CUSTOM_PROMPTS:
        t0 = time.time()
        img = gen_with_backend(pipe, p["prompt"], p["seed"], args.steps, args.height, args.width)
        save(img, out_imp, "trt_implicit", p["index"], p["seed"])
        print(f"  idx {p['index']} ({time.time()-t0:.2f}s)  {p['prompt']}")
    del trt_imp; torch.cuda.empty_cache()

    # 3) Phase 3 B1-c TRT
    print("\n[3/3] Phase 3 B1-c TRT INT8")
    trt_b1c = TRTUNetWrapper(args.b1c_engine, ref_unet_cfg, device="cuda")
    pipe.unet = trt_b1c
    out_b1c = args.output_dir / "trt_b1c"
    for p in CUSTOM_PROMPTS:
        t0 = time.time()
        img = gen_with_backend(pipe, p["prompt"], p["seed"], args.steps, args.height, args.width)
        save(img, out_b1c, "trt_b1c", p["index"], p["seed"])
        print(f"  idx {p['index']} ({time.time()-t0:.2f}s)  {p['prompt']}")

    pipe.unet = fp16_unet
    print(f"\n[done] outputs in {args.output_dir}/{{fp16,trt_implicit,trt_b1c}}")


if __name__ == "__main__":
    main()
