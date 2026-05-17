#!/usr/bin/env python3
"""Progressive small-step pruning for SD-Turbo.

This is a clean pruning path that intentionally avoids the old softmask phase.

One run performs one conservative physical-pruning round:
  1. generate teacher-trajectory calibration data
  2. accumulate long-data Taylor channel scores
  3. combine weighted avg-rank and weighted max-rank
  4. apply A/B stability gating
  5. physically prune a small per-module ratio
  6. report pre/post teacher-forced prediction loss

The script reuses the existing physical-pruning and safetensors save utilities
from sp_core, but score accumulation and ranking are local to this file.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prune import sp_core


DEFAULT_TIMESTEP_WEIGHTS = {
    999: 0.70,
    749: 0.25,
    499: 0.05,
    249: 0.05,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="models/sd-turbo")
    p.add_argument("--model-path", default=None, help="Optional previous-round .safetensors")
    p.add_argument("--model-config", default=None, help="Optional previous-round config JSON")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--max-samples", type=int, default=512, help="Prompt count for scoring")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--prune-ratio", type=float, default=0.02,
                   help="Per-module pruning ratio for this single progressive round")
    p.add_argument("--bottom-pool-ratio", type=float, default=0.10,
                   help="A/B stability pool. A channel must be in both bottom pools to be pruneable")
    p.add_argument("--round-to-val", type=int, default=32)
    p.add_argument("--ffn-ratio", type=float, default=None,
                   help="Per-round FFN ratio. Defaults to --prune-ratio")
    p.add_argument("--protect-blocks", default="down_blocks.0,down_blocks.1,up_blocks.2,up_blocks.3",
                   help="Comma-separated blocks excluded from this round")
    p.add_argument("--candidate-blocks",
                   default="down_blocks.2,down_blocks.3,mid_block,up_blocks.0,up_blocks.1",
                   help="Comma-separated candidate blocks for this round")
    p.add_argument("--include-transformer-ffn", action="store_true",
                   help="Also prune transformer FFN layers inside attention blocks")
    p.add_argument("--timestep-weights", default="prune/timestep_weights_turbo.yaml")
    p.add_argument("--max-eval-samples", type=int, default=64,
                   help="Calib samples used for pre/post pred-loss benchmark")
    return p.parse_args()


def load_timestep_weights(path: str | None) -> Dict[int, float]:
    if not path:
        return dict(DEFAULT_TIMESTEP_WEIGHTS)
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("timestep_weights", cfg)
        weights = {int(k): float(v) for k, v in raw.items()}
        return weights or dict(DEFAULT_TIMESTEP_WEIGHTS)
    except Exception as exc:
        print(f"WARNING: failed to load timestep weights from {path}: {exc}")
        return dict(DEFAULT_TIMESTEP_WEIGHTS)


def comma_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def prunable_named_modules(
    model: nn.Module,
    candidate_blocks: Iterable[str],
    include_transformer_ffn: bool,
) -> List[Tuple[str, nn.Module]]:
    out = []
    for name, module in model.named_modules():
        if not any(name.startswith(b) for b in candidate_blocks):
            continue
        if sp_core._is_attention_name(name):
            continue
        if (not include_transformer_ffn) and sp_core._is_transformer_ffn_name(name):
            continue
        if sp_core._is_sampler_module(name):
            continue
        if sp_core._is_interface_conv(name):
            continue
        if not sp_core._is_prunable_module(name, module):
            continue
        if sp_core._is_ffn_out_proj(name):
            continue
        out.append((name, module))
    return out


def channel_score(weight: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    if weight.dim() >= 2:
        return (weight * grad).abs().mean(dim=tuple(range(1, weight.dim()))).detach()
    return (weight * grad).abs().detach()


def accumulate_scores(
    student_unet: nn.Module,
    calib_data: List[dict],
    modules: List[Tuple[str, nn.Module]],
    timestep_weights: Dict[int, float],
    device: str,
) -> Tuple[Dict[nn.Module, torch.Tensor], Dict[nn.Module, torch.Tensor]]:
    """Return weighted average scores and weighted max scores."""
    module_set = {m for _, m in modules}
    avg_scores: Dict[nn.Module, torch.Tensor] = {}
    max_scores: Dict[nn.Module, torch.Tensor] = {}
    total_w: Dict[nn.Module, float] = {}
    dtype = next(student_unet.parameters()).dtype

    student_unet.train()
    for i, sample in enumerate(calib_data, 1):
        student_unet.zero_grad(set_to_none=True)
        noisy_latents = sample["noisy_latents"].to(device=device, dtype=dtype)
        timesteps = sample["timesteps"].to(device)
        encoder_hidden_states = sample["encoder_hidden_states"].to(device=device, dtype=dtype)
        teacher_pred = sample["teacher_pred"].to(device=device, dtype=dtype)
        t_key = int(timesteps.flatten()[0].item())
        w = float(timestep_weights.get(t_key, 1.0))

        pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(pred, teacher_pred)
        loss.backward()

        for _, module in modules:
            if module not in module_set:
                continue
            if not hasattr(module, "weight") or module.weight is None or module.weight.grad is None:
                continue
            score = channel_score(module.weight, module.weight.grad).float()
            weighted = score * w
            if module not in avg_scores:
                avg_scores[module] = weighted.clone()
                max_scores[module] = weighted.clone()
                total_w[module] = w
            else:
                avg_scores[module] += weighted
                max_scores[module] = torch.maximum(max_scores[module], weighted)
                total_w[module] += w

        if i % 100 == 0:
            print(f"  scored {i}/{len(calib_data)} samples")

    for module in list(avg_scores):
        avg_scores[module] = avg_scores[module] / max(total_w[module], 1e-8)
    return avg_scores, max_scores


def rank_percentile(score: torch.Tensor) -> torch.Tensor:
    """Low score -> low percentile, high score -> high percentile."""
    flat = score.detach().float().flatten()
    n = flat.numel()
    order = torch.argsort(flat, descending=False)
    ranks = torch.empty(n, dtype=torch.float32, device=flat.device)
    if n == 1:
        ranks[order] = 1.0
    else:
        ranks[order] = torch.linspace(0.0, 1.0, n, device=flat.device)
    return ranks.reshape(score.shape)


def build_rank_importance(
    modules: List[Tuple[str, nn.Module]],
    avg_scores: Dict[nn.Module, torch.Tensor],
    max_scores: Dict[nn.Module, torch.Tensor],
    a_scores: Dict[nn.Module, torch.Tensor],
    b_scores: Dict[nn.Module, torch.Tensor],
    bottom_pool_ratio: float,
) -> Dict[nn.Module, torch.Tensor]:
    """Create final per-channel importance.

    Existing physical prune code removes lowest values.  We therefore use
    percentile rank as importance, then set unstable channels to a huge value
    so they are protected.
    """
    out: Dict[nn.Module, torch.Tensor] = {}
    huge = 1e6
    for name, module in modules:
        if module not in avg_scores or module not in max_scores:
            print(f"  WARNING: missing score for {name}, module will be skipped/protected")
            continue
        avg_rank = rank_percentile(avg_scores[module])
        max_rank = rank_percentile(max_scores[module])
        final_rank = torch.maximum(avg_rank, max_rank)

        if module in a_scores and module in b_scores:
            a_rank = rank_percentile(a_scores[module])
            b_rank = rank_percentile(b_scores[module])
            stable = (a_rank <= bottom_pool_ratio) & (b_rank <= bottom_pool_ratio)
            final_rank = torch.where(stable, final_rank, torch.full_like(final_rank, huge))
            stable_n = int(stable.sum().item())
        else:
            stable_n = 0
            final_rank = torch.full_like(final_rank, huge)

        need = max(1, int(final_rank.numel() * 0.02))
        if stable_n < need:
            print(
                f"  WARNING: {name}: only {stable_n}/{final_rank.numel()} stable bottom-pool "
                f"channels; requested pruning may exhaust stable candidates"
            )
        out[module] = final_rank.cpu()
    return out


def split_by_prompt(calib_data: List[dict], steps_per_prompt: int = 4) -> Tuple[List[dict], List[dict]]:
    groups = [calib_data[i:i + steps_per_prompt] for i in range(0, len(calib_data), steps_per_prompt)]
    a, b = [], []
    for idx, group in enumerate(groups):
        (a if idx % 2 == 0 else b).extend(group)
    return a, b


@torch.no_grad()
def eval_pred_loss(student_unet: nn.Module, calib_data: List[dict], device: str, max_samples: int) -> float:
    dtype = next(student_unet.parameters()).dtype
    student_unet.eval()
    losses = []
    for sample in calib_data[:max_samples]:
        noisy_latents = sample["noisy_latents"].to(device=device, dtype=dtype)
        timesteps = sample["timesteps"].to(device)
        encoder_hidden_states = sample["encoder_hidden_states"].to(device=device, dtype=dtype)
        teacher_pred = sample["teacher_pred"].to(device=device, dtype=dtype)
        pred = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
        losses.append(F.mse_loss(pred.float(), teacher_pred.float()).item())
    return sum(losses) / max(1, len(losses))


def make_zones(
    candidate_blocks: List[str],
    protect_blocks: List[str],
    prune_ratio: float,
    ffn_ratio: float,
    round_to: int,
    include_transformer_ffn: bool,
):
    zones = []
    for block in candidate_blocks:
        if block in protect_blocks:
            continue
        zones.append({
            "name": block,
            "keywords": [block],
            "step_ratio": prune_ratio,
            "max_conv_prune": prune_ratio,
            "ffn_ratio": ffn_ratio,
            "round_to": round_to,
            "include_transformer_ffn": include_transformer_ffn,
        })
    return zones


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    timestep_weights = load_timestep_weights(args.timestep_weights)
    protect_blocks = comma_list(args.protect_blocks)
    candidate_blocks = comma_list(args.candidate_blocks)
    ffn_ratio = args.ffn_ratio if args.ffn_ratio is not None else args.prune_ratio

    print("=== Progressive pruning v1 ===")
    print(f"Device: {device}")
    print(f"Student start: {args.model_path or 'base UNet'}")
    print(f"Output: {args.output}")
    print(f"Candidate blocks: {candidate_blocks}")
    print(f"Protected blocks: {protect_blocks}")
    print(f"Per-round conv ratio: {args.prune_ratio:.3f}, ffn ratio: {ffn_ratio:.3f}")
    print(f"Include transformer FFN: {args.include_transformer_ffn}")
    print(f"Timestep weights: {timestep_weights}")

    base_model = sp_core.resolve_base_model_source(args.base_model)
    model_cfg = args.model_config or (
        args.model_path.replace(".safetensors", ".config.json") if args.model_path else None
    )
    pipe, student_unet = sp_core.load_pipeline_and_student(
        base_model, device, pruned_st=args.model_path, pruned_cfg=model_cfg
    )

    print(f"\nGenerating calibration data from {args.dataset}...")
    calib_data = sp_core.generate_calibration_data(
        pipe, args.dataset, device, max_samples=args.max_samples
    )
    a_data, b_data = split_by_prompt(calib_data)
    print(f"Calib split: A={len(a_data)} samples, B={len(b_data)} samples")

    modules = prunable_named_modules(student_unet, candidate_blocks, args.include_transformer_ffn)
    print(f"Prunable candidate modules: {len(modules)}")
    for name, module in modules[:20]:
        size = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
        print(f"  {name}: channels={size}")
    if len(modules) > 20:
        print(f"  ... {len(modules) - 20} more")

    print("\nPre-prune teacher-forced pred loss benchmark...")
    pre_loss = eval_pred_loss(student_unet, calib_data, device, args.max_eval_samples)
    print(f"  pre_pred_loss={pre_loss:.6g}")

    print("\nScoring full calibration data...")
    avg_scores, max_scores = accumulate_scores(student_unet, calib_data, modules, timestep_weights, device)
    print("\nScoring split A for stability...")
    a_avg, _ = accumulate_scores(student_unet, a_data, modules, timestep_weights, device)
    print("\nScoring split B for stability...")
    b_avg, _ = accumulate_scores(student_unet, b_data, modules, timestep_weights, device)

    taylor_scores = build_rank_importance(
        modules, avg_scores, max_scores, a_avg, b_avg, args.bottom_pool_ratio
    )

    zones = make_zones(
        candidate_blocks, protect_blocks, args.prune_ratio, ffn_ratio,
        args.round_to_val, args.include_transformer_ffn
    )
    print("\nPhysical pruning zones:")
    for zone in zones:
        print(
            f"  {zone['name']}: ratio={zone['step_ratio']}, ffn={zone['ffn_ratio']}, "
            f"round_to={zone['round_to']}, include_transformer_ffn={zone['include_transformer_ffn']}"
        )

    grad_sample = calib_data[0]
    grad_inputs = (
        grad_sample["noisy_latents"].to(device=device, dtype=next(student_unet.parameters()).dtype),
        grad_sample["timesteps"].to(device),
        grad_sample["encoder_hidden_states"].to(device=device, dtype=next(student_unet.parameters()).dtype),
    )

    sp_core.prune_zones(
        student_unet=student_unet,
        pipe=pipe,
        zones=zones,
        grad_inputs=grad_inputs,
        importance=None,
        round_to_val=args.round_to_val,
        allow_conv_shortcut_prune=False,
        ffn_max_prune=ffn_ratio,
        max_prune_ratio_per_layer=args.prune_ratio,
        output_path=None,
        device=device,
        taylor_scores=taylor_scores,
    )

    print("\nPost-prune teacher-forced pred loss benchmark...")
    post_loss = eval_pred_loss(student_unet, calib_data, device, args.max_eval_samples)
    rel = (post_loss / pre_loss - 1.0) * 100.0 if pre_loss > 0 else 0.0
    print(f"  post_pred_loss={post_loss:.6g} ({rel:+.2f}% vs pre)")

    print(f"\nSaving pruned model: {args.output}")
    sp_core.save_checkpoint(student_unet, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
