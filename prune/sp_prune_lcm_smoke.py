#!/usr/bin/env python3
"""LCM-native one-round pruning smoke test.

This script is intentionally separate from the SD-Turbo pruning path.  It uses
LCM_Dreamshaper_v7 at its normal operating point: 768px, guidance=8, 8 steps,
original_inference_steps=50, and always passes timestep_cond to the UNet.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from diffusers import DiffusionPipeline, UNet2DConditionModel
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prune import sp_core


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="models/lcm-dreamshaper-v7")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=12)
    p.add_argument("--max-eval-samples", type=int, default=32)
    p.add_argument("--prune-ratio", type=float, default=0.02)
    p.add_argument("--bottom-pool-ratio", type=float, default=0.10)
    p.add_argument("--round-to-val", type=int, default=32)
    p.add_argument("--candidate-blocks", default="down_blocks.2,down_blocks.3,mid_block,up_blocks.0,up_blocks.1")
    p.add_argument("--protect-blocks", default="down_blocks.0,down_blocks.1,up_blocks.2,up_blocks.3")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance-scale", type=float, default=8.0)
    p.add_argument("--original-inference-steps", type=int, default=50)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def comma_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def set_lcm_timesteps(scheduler, steps, device, original_steps):
    try:
        return scheduler.set_timesteps(steps, device=device, original_inference_steps=original_steps)
    except TypeError:
        return scheduler.set_timesteps(steps, device=device)


def guidance_embedding(guidance_scale: float, batch_size: int, dim: int, dtype, device):
    w = torch.full((batch_size,), guidance_scale, dtype=dtype, device=device) * 1000.0
    half = dim // 2
    scale = torch.log(torch.tensor(10000.0, dtype=dtype, device=device)) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, dtype=dtype, device=device) * -scale)
    emb = torch.cat([torch.sin(w[:, None] * freqs[None, :]), torch.cos(w[:, None] * freqs[None, :])], dim=1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def unet_call(unet, sample, timestep, encoder_hidden_states, timestep_cond):
    return unet(
        sample,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        timestep_cond=timestep_cond,
    ).sample


def load_pipe_and_student(base_model: str, device: str):
    base_model = sp_core.resolve_base_model_source(base_model)
    local = Path(base_model).exists()
    print(f"Loading LCM pipeline: {base_model}")
    pipe = DiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=local,
    ).to(device)
    pipe.unet.eval().requires_grad_(False)

    student = UNet2DConditionModel.from_pretrained(
        base_model,
        subfolder="unet",
        torch_dtype=torch.float32,
        local_files_only=local,
    ).to(device)
    student.train()
    student.enable_gradient_checkpointing()
    print(f"Student params: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")
    return pipe, student


def generate_calib(pipe, args, device):
    paths = sorted(Path(args.dataset).glob("*.txt"))
    if not paths:
        raise ValueError(f"No .txt prompts found in {args.dataset}")
    if len(paths) > args.max_samples:
        random.seed(args.seed)
        random.shuffle(paths)
        paths = paths[: args.max_samples]

    teacher = pipe.unet
    scheduler = pipe.scheduler
    dtype = next(teacher.parameters()).dtype
    cond_dim = int(getattr(teacher.config, "time_cond_proj_dim", 256))
    latent_h = args.height // getattr(pipe, "vae_scale_factor", 8)
    latent_w = args.width // getattr(pipe, "vae_scale_factor", 8)

    data = []
    print(
        f"Generating LCM calib: {len(paths)} prompts x {args.steps} steps "
        f"({args.width}x{args.height}, guidance={args.guidance_scale})"
    )
    for path in tqdm(paths, desc="calib"):
        prompt = path.read_text(encoding="utf-8").strip()
        with torch.no_grad():
            text = pipe.tokenizer(
                [prompt], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)
            enc = pipe.text_encoder(text.input_ids)[0].to(dtype=dtype)
            timestep_cond = guidance_embedding(args.guidance_scale, 1, cond_dim, dtype, device)
            set_lcm_timesteps(scheduler, args.steps, device, args.original_inference_steps)
            latents = torch.randn(1, 4, latent_h, latent_w, device=device, dtype=dtype) * scheduler.init_noise_sigma
            for step_idx, t in enumerate(scheduler.timesteps):
                latents_in = latents.clone()
                model_in = scheduler.scale_model_input(latents, t).to(dtype=dtype)
                pred = unet_call(teacher, model_in, t, enc, timestep_cond)
                if hasattr(scheduler, "_step_index"):
                    scheduler._step_index = step_idx
                out = scheduler.step(pred, t, latents_in)
                latents = getattr(out, "prev_sample", out[0])
                data.append({
                    "noisy_latents": model_in.detach(),
                    "timesteps": t.unsqueeze(0),
                    "encoder_hidden_states": enc.detach(),
                    "timestep_cond": timestep_cond.detach(),
                    "teacher_pred": pred.detach(),
                    "step_index": step_idx,
                })
    print(f"Calib samples: {len(data)}")
    return data


def prunable_modules(model: nn.Module, candidate_blocks: Iterable[str], protect_blocks: Iterable[str]):
    out = []
    for name, module in model.named_modules():
        if not any(name.startswith(b) for b in candidate_blocks):
            continue
        if any(name.startswith(b) for b in protect_blocks):
            continue
        if sp_core._is_attention_name(name) or sp_core._is_transformer_ffn_name(name):
            continue
        if sp_core._is_sampler_module(name) or sp_core._is_interface_conv(name):
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        if not name.endswith(".conv1"):
            continue
        out.append((name, module))
    return out


def channel_score(weight, grad):
    return (weight * grad).abs().mean(dim=tuple(range(1, weight.dim()))).detach().float()


def accumulate_scores(student, calib, modules, device):
    scores: Dict[nn.Module, torch.Tensor] = {}
    max_scores: Dict[nn.Module, torch.Tensor] = {}
    counts: Dict[nn.Module, int] = {}
    dtype = next(student.parameters()).dtype
    student.train()
    for i, sample in enumerate(calib, 1):
        student.zero_grad(set_to_none=True)
        noisy = sample["noisy_latents"].to(device=device, dtype=dtype)
        t = sample["timesteps"].to(device)
        enc = sample["encoder_hidden_states"].to(device=device, dtype=dtype)
        cond = sample["timestep_cond"].to(device=device, dtype=dtype)
        target = sample["teacher_pred"].to(device=device, dtype=dtype)
        pred = unet_call(student, noisy, t, enc, cond)
        loss = F.mse_loss(pred, target)
        loss.backward()
        for _, module in modules:
            if module.weight.grad is None:
                continue
            sc = channel_score(module.weight, module.weight.grad)
            scores[module] = scores.get(module, torch.zeros_like(sc.cpu())) + sc.cpu()
            max_scores[module] = sc.cpu() if module not in max_scores else torch.maximum(max_scores[module], sc.cpu())
            counts[module] = counts.get(module, 0) + 1
        if i % 50 == 0:
            print(f"  scored {i}/{len(calib)}")
    for m in list(scores):
        scores[m] /= max(counts[m], 1)
    return scores, max_scores


def rank_percentile(x):
    flat = x.float().flatten()
    order = torch.argsort(flat)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.linspace(0, 1, flat.numel()) if flat.numel() > 1 else torch.ones_like(flat)
    return ranks.reshape(x.shape)


def split_by_prompt(calib, steps):
    groups = [calib[i:i + steps] for i in range(0, len(calib), steps)]
    a, b = [], []
    for idx, group in enumerate(groups):
        (a if idx % 2 == 0 else b).extend(group)
    return a, b


def build_importance(modules, avg, mx, a_avg, b_avg, bottom_pool_ratio):
    imp = {}
    for name, module in modules:
        if module not in avg:
            continue
        final = torch.maximum(rank_percentile(avg[module]), rank_percentile(mx[module]))
        if module in a_avg and module in b_avg:
            stable = (rank_percentile(a_avg[module]) <= bottom_pool_ratio) & (rank_percentile(b_avg[module]) <= bottom_pool_ratio)
            final = torch.where(stable, final, torch.full_like(final, 1e6))
            print(f"  stable {name}: {int(stable.sum())}/{stable.numel()}")
        else:
            final = torch.full_like(final, 1e6)
        imp[module] = final
    return imp


@torch.no_grad()
def eval_loss(student, calib, device, max_samples):
    dtype = next(student.parameters()).dtype
    student.eval()
    vals = []
    for sample in calib[:max_samples]:
        noisy = sample["noisy_latents"].to(device=device, dtype=dtype)
        t = sample["timesteps"].to(device)
        enc = sample["encoder_hidden_states"].to(device=device, dtype=dtype)
        cond = sample["timestep_cond"].to(device=device, dtype=dtype)
        target = sample["teacher_pred"].to(device=device, dtype=dtype)
        pred = unet_call(student, noisy, t, enc, cond)
        vals.append(F.mse_loss(pred.float(), target.float()).item())
    student.train()
    return sum(vals) / max(len(vals), 1)


def prune_with_dg(student, calib_sample, modules, importance, ratio, round_to, device):
    dtype = next(student.parameters()).dtype
    example = {
        "sample": calib_sample["noisy_latents"].to(device=device, dtype=dtype),
        "timestep": calib_sample["timesteps"].to(device),
        "encoder_hidden_states": calib_sample["encoder_hidden_states"].to(device=device, dtype=dtype),
        "timestep_cond": calib_sample["timestep_cond"].to(device=device, dtype=dtype),
    }
    student.eval()
    with torch.no_grad():
        _ = student(**example)
    dg = tp.DependencyGraph().build_dependency(student, example_inputs=example)

    name_to_module = dict(student.named_modules())
    changed = []
    for name, module in modules:
        module = name_to_module.get(name)
        if module is None or module not in importance:
            continue
        out_ch = module.out_channels
        n = int(out_ch * ratio)
        if round_to > 1:
            n = (n // round_to) * round_to if n >= round_to else round_to
        n = min(max(n, 1), out_ch - 1)
        score = importance[module]
        finite = torch.isfinite(score) & (score < 1e5)
        if int(finite.sum()) < n:
            print(f"  skip {name}: stable candidates {int(finite.sum())} < requested {n}")
            continue
        idxs = torch.argsort(score)[:n].tolist()
        group = dg.get_pruning_group(module, tp.prune_conv_out_channels, idxs=idxs)
        if dg.check_pruning_group(group):
            group.prune()
            changed.append((name, out_ch, out_ch - n))
            print(f"  pruned {name}: {out_ch}->{out_ch - n}")
        else:
            print(f"  skip {name}: invalid pruning group")
    return changed


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=== LCM pruning smoke ===")
    print(f"device={device}, steps={args.steps}, guidance={args.guidance_scale}, size={args.width}x{args.height}")
    pipe, student = load_pipe_and_student(args.base_model, device)
    calib = generate_calib(pipe, args, device)
    a_data, b_data = split_by_prompt(calib, args.steps)

    candidates = comma_list(args.candidate_blocks)
    protected = comma_list(args.protect_blocks)
    modules = prunable_modules(student, candidates, protected)
    print(f"candidate conv1 modules: {len(modules)}")
    for name, module in modules:
        print(f"  {name}: {module.out_channels}")

    pre = eval_loss(student, calib, device, args.max_eval_samples)
    print(f"pre_pred_loss={pre:.8f}")

    avg, mx = accumulate_scores(student, calib, modules, device)
    a_avg, _ = accumulate_scores(student, a_data, modules, device)
    b_avg, _ = accumulate_scores(student, b_data, modules, device)
    imp = build_importance(modules, avg, mx, a_avg, b_avg, args.bottom_pool_ratio)

    changed = prune_with_dg(student, calib[0], modules, imp, args.prune_ratio, args.round_to_val, device)
    post = eval_loss(student, calib, device, args.max_eval_samples)
    rel = (post / pre - 1) * 100 if pre > 0 else 0.0
    print(f"post_pred_loss={post:.8f} ({rel:+.2f}% vs pre)")
    print(f"changed_modules={len(changed)}")

    print(f"Saving: {args.output}")
    sp_core.save_checkpoint(student, args.output)
    print("DONE")


if __name__ == "__main__":
    main()
