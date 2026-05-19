#!/usr/bin/env python3
"""Full per-Linear fake-quant sensitivity profiling for SDXL-Lightning."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
SKIP_PATTERNS = ("time_embedding", "time_emb_proj", "add_embedding", "conv_in", "conv_out")

CATEGORIES = {
    "people": ("person", "people", "man", "woman", "boy", "girl", "child", "couple", "family"),
    "animals": ("dog", "cat", "horse", "cow", "sheep", "zebra", "giraffe", "bear", "elephant", "bird"),
    "food": ("pizza", "sandwich", "cake", "donut", "banana", "apple", "orange", "broccoli", "food", "meal", "kitchen"),
    "vehicle": ("train", "bus", "truck", "car", "motorcycle", "airplane", "boat", "bicycle"),
    "sports": ("tennis", "baseball", "skateboard", "snowboard", "ski", "surf", "frisbee", "kite"),
    "indoor": ("room", "bed", "bathroom", "toilet", "sink", "table", "chair", "couch", "laptop", "desk"),
    "street": ("street", "traffic", "sign", "building", "city", "sidewalk", "clock", "parking"),
    "nature": ("beach", "river", "ocean", "lake", "mountain", "snow", "forest", "grass", "sky"),
    "objects": ("phone", "book", "umbrella", "suitcase", "teddy", "vase", "clock", "scissors", "keyboard"),
}
BAD_SUBSTRINGS = ("nude", "porn", "sex", "shirtless", "looking down", "between her legs")


@dataclass
class PromptItem:
    index: int
    source: str
    category: str
    seed: int
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--bits", type=str, default="8,4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "float32"))
    parser.add_argument("--selection-seed", type=int, default=20260517)
    parser.add_argument("--limit-layers", type=int, default=0, help="Debug only; 0 means all target Linear layers.")
    return parser.parse_args()


def normalize_prompt(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def choose_prompts(dataset_dir: Path, count: int, selection_seed: int) -> list[PromptItem]:
    rng = random.Random(selection_seed)
    rows: list[tuple[str, str]] = []
    for path in sorted(dataset_dir.glob("*.txt")):
        text = normalize_prompt(path.read_text(encoding="utf-8", errors="ignore"))
        low = text.lower()
        if len(text) < 15 or any(bad in low for bad in BAD_SUBSTRINGS):
            continue
        rows.append((path.name, text))

    buckets: dict[str, list[tuple[str, str]]] = {name: [] for name in CATEGORIES}
    for source, text in rows:
        low = text.lower()
        for category, keys in CATEGORIES.items():
            if any(k in low for k in keys):
                buckets[category].append((source, text))
                break
    for items in buckets.values():
        rng.shuffle(items)

    chosen: list[PromptItem] = []
    seen: set[str] = set()
    while len(chosen) < count and any(buckets.values()):
        for category in CATEGORIES:
            while buckets[category]:
                source, text = buckets[category].pop()
                if source in seen:
                    continue
                seen.add(source)
                chosen.append(PromptItem(len(chosen) + 1, source, category, 12345 + len(chosen) * 9973, text))
                break
            if len(chosen) >= count:
                break
    if len(chosen) != count:
        raise RuntimeError(f"Only selected {len(chosen)} prompts")
    return chosen


def is_skipped(name: str) -> bool:
    return any(pattern in name for pattern in SKIP_PATTERNS)


def find_linear_layers(unet: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, mod) for name, mod in unet.named_modules() if isinstance(mod, nn.Linear) and not is_skipped(name)]


@torch.no_grad()
def fake_quant_2d_(weight_2d: torch.Tensor, bits: int, group_size: int) -> dict:
    orig_dtype = weight_2d.dtype
    w = weight_2d.detach().float()
    out_f, in_f = w.shape
    if group_size <= 0:
        group_size = in_f
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    out = torch.empty_like(w)
    total_mse = 0.0
    for start in range(0, in_f, group_size):
        end = min(start + group_size, in_f)
        block = w[:, start:end]
        scale = block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.round(block / scale).clamp(qmin, qmax)
        deq = q * scale
        out[:, start:end] = deq
        total_mse += torch.mean((block - deq) ** 2).item() * (end - start)
    weight_2d.copy_(out.to(orig_dtype))
    return {"weight_mse": total_mse / in_f}


def load_lightning_unet(step: int, dtype: torch.dtype, device: str):
    ckpt = f"sdxl_lightning_{step}step_unet.safetensors"
    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(device, dtype)
    ckpt_path = hf_hub_download(LIGHTNING_REPO, ckpt)
    state = load_file(ckpt_path, device=device)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    print(f"Loaded Lightning UNet: {ckpt}, missing={len(missing)}, unexpected={len(unexpected)}")
    return unet


def build_pipe(unet, dtype: torch.dtype, device: str):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        unet=unet,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def build_samples(pipe, prompts: list[PromptItem], args: argparse.Namespace, dtype: torch.dtype, device: str) -> list[dict]:
    pipe.scheduler.set_timesteps(args.steps, device=device)
    samples = []
    for item in prompts:
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=item.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        add_time_ids = pipe._get_add_time_ids(
            (args.height, args.width),
            (0, 0),
            (args.height, args.width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
        ).to(device)
        gen = torch.Generator(device=device).manual_seed(item.seed)
        latent = torch.randn(
            (1, pipe.unet.config.in_channels, args.height // 8, args.width // 8),
            generator=gen,
            device=device,
            dtype=dtype,
        ) * pipe.scheduler.init_noise_sigma
        for step_i, timestep in enumerate(pipe.scheduler.timesteps):
            pipe.scheduler._step_index = step_i
            latent_in = pipe.scheduler.scale_model_input(latent, timestep)
            samples.append(
                {
                    "latent_in": latent_in.detach().clone(),
                    "timestep": timestep,
                    "prompt_embeds": prompt_embeds.detach().clone(),
                    "add": {
                        "text_embeds": pooled_prompt_embeds.detach().clone(),
                        "time_ids": add_time_ids.detach().clone(),
                    },
                    "prompt_index": item.index,
                    "step_index": step_i,
                }
            )
            noise_pred = pipe.unet(
                latent_in,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            ).sample
            pipe.scheduler._step_index = step_i
            latent = pipe.scheduler.step(noise_pred, timestep, latent).prev_sample
    return samples


@torch.no_grad()
def run_unet(unet, samples: list[dict]) -> list[torch.Tensor]:
    outs = []
    for sample in samples:
        out = unet(
            sample["latent_in"],
            sample["timestep"],
            encoder_hidden_states=sample["prompt_embeds"],
            added_cond_kwargs=sample["add"],
        ).sample
        outs.append(out.detach().cpu().float())
    return outs


def divergence(quant_outs: list[torch.Tensor], base_outs: list[torch.Tensor]) -> dict:
    mse_values = []
    cos_values = []
    rel_values = []
    for q, b in zip(quant_outs, base_outs):
        qf = q.flatten()
        bf = b.flatten()
        diff = qf - bf
        mse = torch.mean(diff * diff).item()
        cos = torch.nn.functional.cosine_similarity(qf, bf, dim=0).item()
        rel = torch.linalg.vector_norm(diff).item() / max(1e-12, torch.linalg.vector_norm(bf).item())
        mse_values.append(mse)
        cos_values.append(cos)
        rel_values.append(rel)
    return {
        "mse": float(sum(mse_values) / len(mse_values)),
        "cosine": float(sum(cos_values) / len(cos_values)),
        "one_minus_cosine": float(1.0 - sum(cos_values) / len(cos_values)),
        "relative_l2": float(sum(rel_values) / len(rel_values)),
    }


def atomic_save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("HF_HOME", str(repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(repo_root / ".tmp"))

    out_path = args.output if args.output.is_absolute() else repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bits = [int(bit.strip()) for bit in args.bits.split(",") if bit.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32

    prompts = choose_prompts(repo_root / "dataset", args.prompt_count, args.selection_seed)
    print(
        f"SDXL-Lightning sensitivity: all Linear layers, bits={bits}, prompts={len(prompts)}, "
        f"samples={len(prompts) * args.steps}, size={args.width}x{args.height}, dtype={dtype}, device={device}"
    )
    print(f"Output: {out_path}")

    unet = load_lightning_unet(args.steps, dtype, device)
    pipe = build_pipe(unet, dtype, device)
    print("Building calibration samples...")
    samples = build_samples(pipe, prompts, args, dtype, device)
    print(f"Built {len(samples)} samples")
    print("Computing baseline UNet outputs...")
    baseline_outs = run_unet(pipe.unet, samples)

    targets = find_linear_layers(pipe.unet)
    if args.limit_layers:
        targets = targets[: args.limit_layers]
    print(f"Profiling {len(targets)} Linear layers x {len(bits)} bits")

    report = {
        "metadata": {
            "base_model": BASE_MODEL,
            "lightning_repo": LIGHTNING_REPO,
            "steps": args.steps,
            "height": args.height,
            "width": args.width,
            "prompt_count": len(prompts),
            "sample_count": len(samples),
            "bits": bits,
            "group_size": args.group_size,
            "dtype": str(dtype),
            "device": device,
            "skip_patterns": list(SKIP_PATTERNS),
            "prompts": [asdict(item) for item in prompts],
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "layers": {},
    }
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("metadata", {}).get("steps") == args.steps:
                report["layers"].update(existing.get("layers", {}))
                print(f"Resuming from existing file: {len(report['layers'])} layer entries")
        except Exception:
            pass

    for layer_idx, (name, mod) in enumerate(targets, start=1):
        entry = report["layers"].setdefault(
            name,
            {"type": "Linear", "param_count": mod.weight.numel(), "shape": list(mod.weight.shape)},
        )
        original_weight = mod.weight.detach().clone()
        for bit in bits:
            key = f"int{bit}"
            if key in entry and entry[key] is not None:
                continue
            t0 = time.time()
            try:
                fq_stats = fake_quant_2d_(mod.weight, bit, args.group_size)
                quant_outs = run_unet(pipe.unet, samples)
                score = divergence(quant_outs, baseline_outs)
                entry[key] = score["one_minus_cosine"]
                entry[f"{key}_detail"] = {**score, **fq_stats}
                elapsed = time.time() - t0
                print(
                    f"[{layer_idx}/{len(targets)}] {name} {key}: "
                    f"1-cos={score['one_minus_cosine']:.6e}, rel_l2={score['relative_l2']:.6e}, {elapsed:.1f}s"
                )
            except Exception as exc:
                entry[key] = None
                entry[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{layer_idx}/{len(targets)}] {name} {key}: ERROR {entry[f'{key}_error']}")
            finally:
                with torch.no_grad():
                    mod.weight.copy_(original_weight)
                if "quant_outs" in locals():
                    del quant_outs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            atomic_save_json(out_path, report)
    report["metadata"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_save_json(out_path, report)
    print(f"DONE: {out_path}")


if __name__ == "__main__":
    main()
