#!/usr/bin/env python3
"""
Pre-compute teacher trajectories for attention map distillation.

Runs the teacher UNet once per (prompt, seed) pair and saves per-step tensors:
  - input_latent : UNet input latent at this step   [1, 4, H/8, W/8] fp16
  - t_pred       : teacher epsilon prediction        [1, 4, H/8, W/8] fp16
  - next_latent  : teacher latent after scheduler.step [1, 4, H/8, W/8] fp16
  - attn_maps    : {layer_key: [1, seq_q, seq_k] fp16}  head-averaged

Cache layout on disk:
    cache_dir/
        00000_0000/    ← prompt_idx=0, seed=0
            step_0.pt  ← timestep t=999
            step_1.pt  ← timestep t=749
            step_2.pt  ← timestep t=499
            step_3.pt  ← timestep t=249
        00000_0001/    ← prompt_idx=0, seed=1
        ...

Storage estimate: ~1 MB per trajectory × n_seeds × n_prompts
  e.g. 5 seeds × 1000 prompts ≈ 5 GB

VRAM benefit during training: removing the teacher UNet from GPU frees ~1.7 GB
of weights plus ~2–3 GB of forward activations, allowing batch_size to increase.

Resume: existing trajectory directories (where all step files exist) are skipped.

Usage:
    python prune/gen_teacher_cache.py --run-config prune/sp_distill_config.yaml \\
        --cache-dir teacher_cache --n-seeds 5

    # Override attention layers:
    python prune/gen_teacher_cache.py --cache-dir teacher_cache --n-seeds 3 \\
        --attn-layers mid_block.attentions.0.transformer_blocks.0.attn1 \\
                      mid_block.attentions.0.transformer_blocks.0.attn2
"""

import os
import sys
import copy
import argparse
import random
import glob as _glob
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN_CONFIG = os.path.join(_SCRIPT_DIR, 'sp_distill_config.yaml')

# Default layers matching sp_distill_config.yaml
DEFAULT_ATTN_LAYERS = [
    "mid_block.attentions.0.transformer_blocks.0.attn1",
    "mid_block.attentions.0.transformer_blocks.0.attn2",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2",
    "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
    "up_blocks.0.attentions.0.transformer_blocks.0.attn2",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
]


def _load_run_config(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return {k: v for k, v in cfg.items() if v is not None}
    except ImportError:
        print("WARNING: PyYAML not installed — run-config ignored.")
        return {}
    except Exception as e:
        print(f"WARNING: Failed to load run config {path}: {e}")
        return {}


def main():
    # ── Config loading (same two-pass approach as sp_distill.py) ──────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--run-config', default=DEFAULT_RUN_CONFIG)
    pre_args, _ = pre.parse_known_args()
    rc = _load_run_config(pre_args.run_config)

    def _rc(key, default=None):
        return rc.get(key, default)

    # Apply cache paths before HuggingFace imports
    os.environ.setdefault('HF_HOME',           _rc('hf_cache', '.hf_cache'))
    os.environ.setdefault('TRANSFORMERS_CACHE', _rc('hf_cache', '.hf_cache'))
    os.environ.setdefault('TMPDIR',             _rc('tmp_dir',  '.tmp'))

    parser = argparse.ArgumentParser(
        description="Pre-compute SD-Turbo teacher trajectories for attention map distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run-config',  default=DEFAULT_RUN_CONFIG)
    parser.add_argument('--base-model',  default=_rc('base_model',  'models/sd-turbo'))
    parser.add_argument('--dataset',     default=_rc('dataset',     'dataset'))
    parser.add_argument('--max-samples', type=int, default=_rc('max_samples', 5000))
    parser.add_argument('--device',      default=_rc('device'))
    parser.add_argument('--resolution',  type=int, default=_rc('resolution', 512))
    parser.add_argument('--cache-dir',
                        default=_rc('teacher_cache_dir', 'teacher_cache'),
                        help="Directory to write cached trajectories into.")
    parser.add_argument('--n-seeds', type=int, default=5,
                        help="Number of noise seeds per prompt. "
                             "Total trajectories = n_prompts × n_seeds.")
    parser.add_argument('--n-steps', type=int, default=4,
                        help="Denoising steps per trajectory (must match n_steps in distillation).")
    parser.add_argument('--attn-layers', nargs='*',
                        default=_rc('attn_layers', DEFAULT_ATTN_LAYERS),
                        help="Dot-paths of attention modules to capture. "
                             "Pass empty list to only cache t_pred/latents.")
    args = parser.parse_args()

    device     = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Device      : {device}")
    print(f"Cache dir   : {cache_root.resolve()}")
    print(f"Seeds/prompt: {args.n_seeds}")
    print(f"Steps       : {args.n_steps}")
    print(f"Attn layers : {len(args.attn_layers or [])}")

    # ── Load teacher pipeline ─────────────────────────────────────────────────
    print(f"\nLoading teacher from: {args.base_model}")
    from diffusers import StableDiffusionPipeline
    from prune import sp_core
    base_model_id = sp_core.resolve_base_model_source(args.base_model)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    teacher_unet = pipe.unet
    teacher_unet.eval()
    teacher_unet.requires_grad_(False)
    scheduler = copy.deepcopy(pipe.scheduler)

    # ── Install attention capture hooks on teacher ────────────────────────────
    registry = None
    if args.attn_layers:
        from prune.attn_map_capture import AttentionMapRegistry
        registry = AttentionMapRegistry()
        registry.register(teacher_unet, args.attn_layers)
        print(f"Hooked {len(args.attn_layers)} attention layers")

    # ── Load prompts ──────────────────────────────────────────────────────────
    all_txts = sorted(_glob.glob(os.path.join(args.dataset, "*.txt")))
    if not all_txts:
        raise ValueError(f"No .txt prompt files found in {args.dataset}")
    if len(all_txts) > args.max_samples:
        random.seed(42)
        random.shuffle(all_txts)
        all_txts = all_txts[:args.max_samples]

    total = len(all_txts) * args.n_seeds
    print(f"\nPrompts: {len(all_txts)}  ×  {args.n_seeds} seeds  =  {total} trajectories\n")

    latent_h = args.resolution // 8
    latent_w = args.resolution // 8

    # ── Progress bar (tqdm optional) ──────────────────────────────────────────
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Caching", unit="traj")

        def tick():
            pbar.update(1)

        def close():
            pbar.close()
    except ImportError:
        _done_count = [0]

        def tick():
            _done_count[0] += 1
            if _done_count[0] % 50 == 0:
                print(f"  {_done_count[0]}/{total}")

        def close():
            pass

    # ── Cache generation loop ─────────────────────────────────────────────────
    skipped = done = 0

    for p_idx, txt_path in enumerate(all_txts):
        with open(txt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        # Encode text once per prompt — same across all seeds for this prompt
        with torch.no_grad():
            text_input = pipe.tokenizer(
                [prompt],
                padding="max_length", max_length=77,
                truncation=True, return_tensors="pt",
            ).to(device)
            enc = pipe.text_encoder(text_input.input_ids)[0]  # [1, 77, 768] fp16

        for seed in range(args.n_seeds):
            traj_dir = cache_root / f"{p_idx:05d}_{seed:04d}"

            # Skip if all step files already exist (allow resuming)
            if traj_dir.exists() and all(
                (traj_dir / f"step_{i}.pt").exists() for i in range(args.n_steps)
            ):
                skipped += 1
                tick()
                continue

            traj_dir.mkdir(exist_ok=True)

            # Deterministic initial noise — (prompt_idx, seed) → unique trajectory
            torch.manual_seed(seed * 10000 + p_idx)
            latent = torch.randn(
                (1, 4, latent_h, latent_w), device=device, dtype=torch.float16,
            ) * pipe.scheduler.init_noise_sigma

            scheduler.set_timesteps(args.n_steps, device=device)

            with torch.no_grad():
                for step_i, t in enumerate(scheduler.timesteps):
                    input_latent = latent.clone()

                    # Clear previous step's captured maps
                    if registry is not None:
                        registry.clear()

                    # Teacher forward
                    scheduler._step_index = step_i
                    t_in   = scheduler.scale_model_input(input_latent, t)
                    t_pred = teacher_unet(t_in, t, enc).sample               # fp16

                    # Teacher scheduler step
                    scheduler._step_index = step_i
                    next_latent = scheduler.step(t_pred, t, input_latent).prev_sample  # fp16

                    # Collect captured attention maps as fp16 on CPU
                    attn_maps_cpu = {}
                    if registry is not None:
                        for k, v in registry.captured.items():
                            attn_maps_cpu[k] = v.half().cpu()

                    # Save step cache
                    try:
                        torch.save(
                            {
                                'input_latent': input_latent.cpu().half(),
                                't_pred':       t_pred.cpu().half(),
                                'next_latent':  next_latent.cpu().half(),
                                'attn_maps':    attn_maps_cpu,
                            },
                            traj_dir / f"step_{step_i}.pt",
                        )
                    except Exception as e:
                        print(f"\nWARNING: failed to save {traj_dir.name}/step_{step_i}.pt: {e} — skipping trajectory")
                        break

                    latent = next_latent  # advance teacher's own trajectory

            done += 1
            tick()

    close()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if registry is not None:
        registry.restore(teacher_unet)

    print(f"\nDone.")
    print(f"  Generated : {done}")
    print(f"  Skipped   : {skipped} (already cached)")
    print(f"  Cache dir : {cache_root.resolve()}")


if __name__ == "__main__":
    main()
