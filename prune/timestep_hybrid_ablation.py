#!/usr/bin/env python3
"""Hybrid timestep ablation for SD-Turbo pruning.

For each prompt/seed, this script generates:
  1. a full teacher 4-step image
  2. one hybrid image per timestep, where exactly that timestep uses the
     student/pruned UNet and all other timesteps use the teacher UNet.

The main metric is final latent MSE versus the full-teacher rollout.  This
answers which inference timestep is most sensitive to student/pruned errors.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prune import sp_core
from prune.pruned_rebuild import create_unet_from_safetensors


PROMPTS = [
    "a beautiful landscape with mountains and lake, highly detailed, 4k",
    "a cute cat sitting on a windowsill, digital art, warm lighting",
    "a cyberpunk city skyline at night, neon lights, futuristic",
    "a portrait of a woman with flowers in her hair, soft lighting, artstation",
    "a red sports car parked on a rainy street, reflections, cinematic",
    "an astronaut riding a horse on the moon, surreal, high quality",
    "a cozy coffee shop interior, morning light through windows, watercolor style",
    "a golden retriever playing in autumn leaves, photography, bokeh",
]

SEEDS = [42, 123, 2024, 7, 256, 999, 55, 314]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="models/sd-turbo")
    parser.add_argument("--student-st", required=True, help="Student/pruned .safetensors")
    parser.add_argument("--student-cfg", default=None, help="Student config JSON")
    parser.add_argument("--output-dir", default="gen_test_output/timestep_hybrid_ablation")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def get_prompt_embeds(pipe: StableDiffusionPipeline, prompt: str, device: torch.device):
    if hasattr(pipe, "encode_prompt"):
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return prompt_embeds

    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    return pipe.text_encoder(text_input_ids)[0]


def prepare_latents(
    pipe: StableDiffusionPipeline,
    seed: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    channels = pipe.unet.config.in_channels
    shape = (1, channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return latents * pipe.scheduler.init_noise_sigma


def unet_step(pipe, unet, latents, timestep, prompt_embeds, step_index):
    pipe.scheduler._step_index = step_index
    latent_in = pipe.scheduler.scale_model_input(latents, timestep)
    pred = unet(latent_in, timestep, encoder_hidden_states=prompt_embeds).sample
    pipe.scheduler._step_index = step_index
    return pipe.scheduler.step(pred.float(), timestep, latents).prev_sample.to(latents.dtype)


def rollout(pipe, teacher_unet, student_unet, prompt_embeds, init_latents, student_step_idx=None):
    latents = init_latents.clone()
    for i, timestep in enumerate(pipe.scheduler.timesteps):
        unet = student_unet if i == student_step_idx else teacher_unet
        latents = unet_step(pipe, unet, latents, timestep, prompt_embeds, i)
    return latents


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents.to(dtype=pipe.vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return pipe.numpy_to_pil(image)[0]


def image_mse(a: Image.Image, b: Image.Image) -> float:
    ta = torch.from_numpy(__import__("numpy").array(a).astype("float32") / 255.0)
    tb = torch.from_numpy(__import__("numpy").array(b).astype("float32") / 255.0)
    return F.mse_loss(ta, tb).item()


def make_contact_sheet(image_paths, labels, out_path: Path, thumb=192):
    label_h = 42
    cols = len(image_paths)
    canvas = Image.new("RGB", (cols * thumb, thumb + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for col, (path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(path).convert("RGB").resize((thumb, thumb), Image.LANCZOS)
        x = col * thumb
        canvas.paste(img, (x, label_h))
        draw.text((x + 6, 7), label, fill="black", font=font)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base_source = sp_core.resolve_base_model_source(args.base_model)
    print(f"Loading teacher pipeline: {base_source}", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_source,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler.set_timesteps(args.steps, device=device)

    print(f"Loading student UNet: {args.student_st}", flush=True)
    student_cfg = args.student_cfg or args.student_st.replace(".safetensors", ".config.json")
    student_unet = create_unet_from_safetensors(args.student_st, student_cfg).to(dtype=dtype)

    if args.cpu_offload and device.type == "cuda":
        pipe.enable_sequential_cpu_offload()
        student_unet.to(device)
    else:
        pipe.to(device)
        student_unet.to(device)

    pipe.unet.eval()
    student_unet.eval()
    teacher_unet = pipe.unet

    rows = []
    prompts = PROMPTS[: args.max_prompts]
    seeds = SEEDS[: args.max_prompts]
    timestep_labels = [int(t.item()) for t in pipe.scheduler.timesteps]
    print(f"Timesteps: {timestep_labels}", flush=True)

    for prompt_idx, (prompt, seed) in enumerate(zip(prompts, seeds), 1):
        print(f"[{prompt_idx}/{len(prompts)}] seed={seed}: {prompt[:70]}...", flush=True)
        prompt_embeds = get_prompt_embeds(pipe, prompt, device).to(dtype=dtype)
        init_latents = prepare_latents(pipe, seed, args.height, args.width, dtype, device)

        with torch.no_grad():
            teacher_latents = rollout(
                pipe, teacher_unet, student_unet, prompt_embeds, init_latents, student_step_idx=None
            )
            teacher_img = decode_latents(pipe, teacher_latents)

            prompt_dir = out_dir / f"prompt_{prompt_idx:02d}"
            prompt_dir.mkdir(exist_ok=True)
            teacher_path = prompt_dir / "teacher.png"
            teacher_img.save(teacher_path)

            sheet_paths = [teacher_path]
            sheet_labels = ["teacher"]

            for step_idx, t_label in enumerate(timestep_labels):
                hybrid_latents = rollout(
                    pipe, teacher_unet, student_unet, prompt_embeds, init_latents,
                    student_step_idx=step_idx,
                )
                hybrid_img = decode_latents(pipe, hybrid_latents)
                hybrid_path = prompt_dir / f"student_at_step{step_idx}_t{t_label}.png"
                hybrid_img.save(hybrid_path)

                latent_mse = F.mse_loss(
                    hybrid_latents.float(), teacher_latents.float()
                ).item()
                pix_mse = image_mse(hybrid_img, teacher_img)
                rows.append({
                    "prompt_idx": prompt_idx,
                    "seed": seed,
                    "student_step_idx": step_idx,
                    "timestep": t_label,
                    "final_latent_mse": latent_mse,
                    "image_mse": pix_mse,
                    "image": str(hybrid_path),
                    "prompt": prompt,
                })
                print(
                    f"  student at step {step_idx} t={t_label}: "
                    f"latent_mse={latent_mse:.6g} image_mse={pix_mse:.6g}",
                    flush=True,
                )
                sheet_paths.append(hybrid_path)
                sheet_labels.append(f"s@{step_idx}\nt={t_label}")

            make_contact_sheet(sheet_paths, sheet_labels, prompt_dir / "sheet.png")

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for row in rows:
        key = (row["student_step_idx"], row["timestep"])
        summary.setdefault(key, {"latent": [], "image": []})
        summary[key]["latent"].append(row["final_latent_mse"])
        summary[key]["image"].append(row["image_mse"])

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["student_step_idx", "timestep", "mean_final_latent_mse", "mean_image_mse"])
        print("\nSummary:", flush=True)
        for (step_idx, timestep), vals in sorted(summary.items()):
            mean_latent = sum(vals["latent"]) / len(vals["latent"])
            mean_image = sum(vals["image"]) / len(vals["image"])
            writer.writerow([step_idx, timestep, mean_latent, mean_image])
            print(
                f"  step {step_idx} t={timestep}: "
                f"mean_latent_mse={mean_latent:.6g} mean_image_mse={mean_image:.6g}",
                flush=True,
            )

    print(f"\nSaved metrics: {csv_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved images under: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
