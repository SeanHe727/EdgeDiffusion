"""Generate the Phase 2 (PyTorch Packed + Q-LoRA) row for the Phase 3 showcase.

Runs locally on the RTX 5070 with the packed safetensors + LoRA. Produces the
two custom-prompt images that match the FP16 / Sprint 2 / Phase 3 rows already
generated on EC2.

Usage (from repo root):
  python utils/gen_custom_phase2.py \
    --packed-safetensors models/real_quant/sdxl_lightning_weight/sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.safetensors \
    --packed-manifest    models/real_quant/sdxl_lightning_weight/sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.json \
    --lora               qlora_rank16_e3_lr1e4/lora_final.safetensors

Output: evals/eval_output/custom_showcase/packed_lora/packed_lora_{65,66,67}_seed*.png
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from safetensors.torch import load_file
import json

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantization.main.packed.build_packed_unet import build_packed_unet
from quantization.main.packed.qlora_wrapper import load_lora_state_dict, wrap_packed_linears_with_lora


CUSTOM_PROMPTS = [
    {"index": 65, "seed": 650000, "prompt": "Portrait of a blonde woman"},
    {"index": 66, "seed": 660000, "prompt": "A cyberpunk city skyline at night, neon lights"},
    {"index": 67, "seed": 670000, "prompt": "A fluffy cat sitting on a windowsill"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-safetensors", type=Path, required=True)
    ap.add_argument("--packed-manifest", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=4.0)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "evals" / "eval_output" / "custom_showcase" / "packed_lora")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"; dtype = torch.float16

    print("Loading packed manifest + state...")
    manifest = json.loads(args.packed_manifest.read_text(encoding="utf-8"))
    packed_state = load_file(str(args.packed_safetensors), device="cpu")
    print("Building packed UNet...")
    unet, stats = build_packed_unet(packed_state, manifest, device, dtype)
    del packed_state
    torch.cuda.empty_cache()
    print(f"  packed: {stats['n_w4']} W4 + {stats['n_w8']} W8 layers")

    print(f"Wrapping with LoRA r={args.lora_rank}, alpha={args.lora_alpha}...")
    n_wrapped = wrap_packed_linears_with_lora(unet, r=args.lora_rank, alpha=args.lora_alpha, dtype=dtype)
    print(f"  wrapped {n_wrapped} layers; loading LoRA state...")
    lora_state = load_file(str(args.lora), device=device)
    load_lora_state_dict(unet, lora_state)
    unet.eval()
    if hasattr(unet, "disable_gradient_checkpointing"):
        unet.disable_gradient_checkpointing()

    print("Building SDXL pipeline with packed UNet...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        manifest["base_model"], unet=unet, torch_dtype=dtype,
        variant="fp16", use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)

    for p in CUSTOM_PROMPTS:
        gen = torch.Generator(device=device).manual_seed(p["seed"])
        with torch.no_grad():
            img = pipe(p["prompt"], num_inference_steps=args.steps, guidance_scale=0.0,
                       height=args.height, width=args.width, generator=gen).images[0]
        out = args.output_dir / f"packed_lora_{p['index']:02d}_seed{p['seed']}.png"
        img.save(out)
        print(f"  saved {out.name}  ({p['prompt']})")

    print(f"\n[done] outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
