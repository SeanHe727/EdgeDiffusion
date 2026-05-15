#!/usr/bin/env python3
"""Unified pruning entry point (Taylor + softmask only).

All settings live in prune/sp_apply_config.yaml — just run:

  python prune/sp_apply.py

CLI flags always override the config file:

  python prune/sp_apply.py --warmup 50 --softmask 100   # quick test
  python prune/sp_apply.py --run-config other.yaml       # swap config

Calib data priority:
  1. calib_data: <path>   in config (or --calib-data)  — pre-saved .pt
  2. dataset: <dir>       in config (or --dataset)     — generate via teacher inference
  3. (fallback)                                         — random data, smoke test only
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default run-config: same directory as this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN_CONFIG = os.path.join(_SCRIPT_DIR, 'sp_apply_config.yaml')


def _load_run_config(path: str) -> dict:
    """Load YAML run config. Returns empty dict if file missing or PyYAML absent."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return {k: v for k, v in cfg.items() if v is not None}
    except ImportError:
        print("WARNING: PyYAML not installed — run-config ignored. pip install pyyaml")
        return {}
    except Exception as e:
        print(f"WARNING: Failed to load run config {path}: {e}")
        return {}


def _get_text_embed_dim(pipe) -> int:
    """Detect text embedding dim from the pipeline's text encoder config.

    SD-Turbo / SD 2.x uses OpenCLIP (dim=1024).
    SD 1.5 / SD 1.x uses CLIP ViT-L (dim=768).
    Falls back to 1024 if config is unavailable.
    """
    try:
        return pipe.text_encoder.config.hidden_size
    except Exception:
        return 1024


def make_random_calib_data(pipe, device, n_samples=8):
    """Generate minimal random calib data for smoke testing (no real teacher trajectory).

    Automatically detects text embedding dim so the same function works for
    both SD-Turbo (dim=1024) and SD 1.5 (dim=768).
    """
    import torch
    dtype = torch.float16
    unet = pipe.unet
    scheduler = pipe.scheduler
    scheduler.set_timesteps(4, device=device)
    fixed_ts = scheduler.timesteps

    # Auto-detect: SD-Turbo=1024, SD 1.5=768
    text_dim = _get_text_embed_dim(pipe)

    data = []
    for i in range(n_samples):
        t_idx = i % len(fixed_ts)
        t = fixed_ts[t_idx]
        noisy  = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
        enc_hs = torch.randn(1, 77, text_dim, device=device, dtype=dtype)
        with torch.no_grad():
            teacher_pred = unet(noisy, t, encoder_hidden_states=enc_hs).sample
            scheduler._step_index = t_idx
            teacher_next = scheduler.step(teacher_pred, t, noisy).prev_sample
        data.append({
            'noisy_latents':        noisy.detach(),
            'latents_in':           noisy.detach(),
            'timesteps':            t.unsqueeze(0),
            'step_index':           t_idx,
            'encoder_hidden_states': enc_hs.detach(),
            'teacher_pred':         teacher_pred.detach(),
            'teacher_next_latent':  teacher_next.detach().float(),
        })
    print(f"  Random calib data: {len(data)} samples, text_dim={text_dim} (smoke-test mode)")
    return data


def main():
    # ── Step 1: find --run-config before building the full parser ─────────────
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--run-config', default=DEFAULT_RUN_CONFIG)
    pre_args, _ = pre.parse_known_args()
    rc = _load_run_config(pre_args.run_config)

    if rc:
        print(f"Run config: {pre_args.run_config}")

    # Apply env vars from config before any downstream imports read them
    hf_cache = rc.get('hf_cache', '.hf_cache')
    tmp_dir  = rc.get('tmp_dir',  '.tmp')
    os.environ.setdefault('HF_HOME',            hf_cache)
    os.environ.setdefault('TRANSFORMERS_CACHE',  hf_cache)
    os.environ.setdefault('TMPDIR',              tmp_dir)
    if 'base_model' in rc:
        os.environ.setdefault('BASE_MODEL_ID', str(rc['base_model']))

    from prune import sp_core

    def _rc(key, default=None):
        return rc.get(key, default)

    # ── Step 2: full argument parser (config file values are the defaults) ─────
    parser = argparse.ArgumentParser(
        description="Unified pruning entry point — configure via sp_apply_config.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run-config', default=DEFAULT_RUN_CONFIG,
                        help='YAML run config path (auto-loaded if present)')

    # Paths
    parser.add_argument('--hf-cache',  default=_rc('hf_cache',  '.hf_cache'), help='HuggingFace cache dir')
    parser.add_argument('--tmp-dir',   default=_rc('tmp_dir',   '.tmp'),       help='Temp dir')

    # Model
    parser.add_argument('--base-model',          default=_rc('base_model', 'models/sd-turbo'))
    parser.add_argument('--inference-timesteps', default=_rc('inference_timesteps', '999,749,499,249'))
    parser.add_argument('--device',              default=_rc('device'))

    # Pruning target
    parser.add_argument('--config',       default=_rc('config'),       help='Pruning config JSON')
    parser.add_argument('--target',       default=_rc('target'),       help='Target key in config (auto if only one)')
    parser.add_argument('--round',        type=int, default=_rc('round'), help='Round number for history record')
    parser.add_argument('--model-path',   default=_rc('model_path'),   help='Previous round .safetensors')
    parser.add_argument('--model-config', default=_rc('model_config'), help='Previous round config JSON')

    # Calibration data
    parser.add_argument('--calib-data',  default=_rc('calib_data'),           help='Pre-saved calib_data.pt')
    parser.add_argument('--dataset',     default=_rc('dataset',   'dataset'), help='.txt prompt directory')
    parser.add_argument('--max-samples', type=int, default=_rc('max_samples', 64), help='Max calib prompts')

    # Output
    parser.add_argument('--output', default=_rc('output'), help='Output .safetensors path')

    # Taylor pipeline
    parser.add_argument('--warmup',          type=int,   default=_rc('warmup',          200))
    parser.add_argument('--softmask',        type=int,   default=_rc('softmask',        200))
    parser.add_argument('--rampup',          type=int,   default=_rc('rampup',         1000))
    parser.add_argument('--reeval-interval', type=int,   default=_rc('reeval_interval', 100))
    parser.add_argument('--round-to-val',    type=int,   default=_rc('round_to_val',     32),    help='Channel rounding granularity')
    parser.add_argument('--ffn-max-prune',   type=float, default=_rc('ffn_max_prune',  0.10),   help='Max pruning ratio per FFN layer')
    parser.add_argument('--max-prune-ratio', type=float, default=_rc('max_prune_ratio', 0.40),  help='Global cap on pruning ratio per layer')
    parser.add_argument('--taylor-mode',     default=_rc('taylor_mode', 'avg'),                 help='Score aggregation across timesteps: "avg" or "max"')

    # Smoke test
    parser.add_argument('--smoke-test-samples', type=int, default=_rc('smoke_test_samples', 8),
                        help='Random calib samples when dataset is unavailable')

    args = parser.parse_args()

    if not args.config:
        parser.error("'config' is required — set it in sp_apply_config.yaml or pass --config")
    if not args.output:
        parser.error("'output' is required — set it in sp_apply_config.yaml or pass --output")

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load pruning config JSON ───────────────────────────────────────────────
    print(f"Loading pruning config: {args.config}")
    with open(args.config) as f:
        all_configs = json.load(f)

    configs = all_configs['configs']
    target_key = args.target or list(configs.keys())[0]
    if target_key not in configs:
        parser.error(f"Target '{target_key}' not in config. Available: {list(configs.keys())}")

    config    = configs[target_key]
    per_block = config['per_block']
    print(f"Target: {target_key} — actual {config['actual']*100:.1f}%, "
          f"estimated removal {config['params_removed_M']:.1f}M params")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # ── Load pipeline + student UNet ──────────────────────────────────────────
    base_model = sp_core.resolve_base_model_source(args.base_model)
    pruned_st  = args.model_path
    pruned_cfg = args.model_config or (
        pruned_st.replace('.safetensors', '.config.json') if pruned_st else None
    )
    pipe, student_unet = sp_core.load_pipeline_and_student(
        base_model, device, pruned_st=pruned_st, pruned_cfg=pruned_cfg)

    zones = sp_core.load_zones_from_config(args.config)
    print("Zone config:")
    for z in zones:
        print(f"  {z['name']}: step_ratio={z.get('step_ratio')}, "
              f"ffn={z.get('ffn_ratio', 0)}, "
              f"self_attn={z.get('attn_self_ratio', 0)}, "
              f"cross_attn={z.get('attn_cross_ratio', 0)}")

    # ── Calibration data ──────────────────────────────────────────────────────
    if args.calib_data and os.path.exists(args.calib_data):
        print(f"\nLoading calib data from {args.calib_data} ...")
        calib_data = torch.load(args.calib_data, map_location=device)
        if len(calib_data) > args.max_samples:
            import random; random.seed(42)
            calib_data = random.sample(calib_data, args.max_samples)
        print(f"  Loaded {len(calib_data)} samples")
    elif os.path.isdir(args.dataset):
        if args.calib_data:
            print(f"WARNING: {args.calib_data} not found — generating from dataset")
        else:
            print(f"\nGenerating calib data from {args.dataset} (teacher inference) ...")
        calib_data = sp_core.generate_calibration_data(
            pipe, args.dataset, device, max_samples=args.max_samples)
    else:
        print(f"\nDataset '{args.dataset}' not found — using random calib data (smoke-test mode)")
        calib_data = make_random_calib_data(pipe, device, n_samples=args.smoke_test_samples)

    # ── Run Taylor + softmask pipeline ────────────────────────────────────────
    sp_core.taylor_softmask_pipeline(
        pipe=pipe,
        student_unet=student_unet,
        zones=zones,
        calib_data=calib_data,
        device=device,
        warmup_grad_steps=args.warmup,
        softmask_steps=args.softmask,
        per_block_config=per_block,
        output_path=args.output,
        reeval_interval=args.reeval_interval,
        rampup_steps=args.rampup,
        round_to_val=args.round_to_val,
        ffn_max_prune=args.ffn_max_prune,
        max_prune_ratio=args.max_prune_ratio,
        inference_timesteps=args.inference_timesteps,
        taylor_mode=args.taylor_mode,
    )

    # ── Write pruning history record ──────────────────────────────────────────
    if args.round is not None:
        from safetensors.torch import load_file as _load
        params_after = sum(v.numel() for v in _load(args.output).values()) / 1e6

        record_path = os.path.join(os.path.dirname(args.config), 'pruning_config.json')
        if os.path.exists(record_path):
            with open(record_path) as f:
                record = json.load(f)
            baseline_m = record.get('original_params_M')
        else:
            from diffusers import UNet2DConditionModel
            _u = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet", torch_dtype=torch.float32)
            baseline_m = sum(p.numel() for p in _u.parameters()) / 1e6
            del _u
            record = {
                "model": base_model,
                "original_params_M": round(baseline_m, 2),
                "protected_blocks": _rc('protected_blocks', config.get('protected_blocks', [])),
                "pruning_history": [],
            }

        params_before = baseline_m
        if pruned_st:
            params_before = sum(v.numel() for v in _load(pruned_st).values()) / 1e6

        record["pruning_history"] = [r for r in record["pruning_history"] if r.get("round") != args.round]
        record["pruning_history"].append({
            "round":                args.round,
            "target":               target_key,
            "params_before_M":      round(params_before, 1),
            "params_after_M":       round(params_after,  1),
            "params_removed_M":     round(params_before - params_after, 1),
            "actual_removal_pct":   round((params_before - params_after) / params_before * 100, 2),
            "cumulative_removal_pct": round((baseline_m - params_after) / baseline_m * 100, 2),
            "config_file":          os.path.basename(args.config),
            "output_path":          args.output,
            "training": {
                "warmup_steps":    args.warmup,
                "softmask_steps":  args.softmask,
                "rampup_steps":    args.rampup,
                "reeval_interval": args.reeval_interval,
            },
            "per_block": per_block,
        })
        record["pruning_history"].sort(key=lambda r: r["round"])

        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)
        print(f"  Updated pruning record: {record_path}")

    print(f"\nDone. Output: {args.output}")


if __name__ == '__main__':
    main()
