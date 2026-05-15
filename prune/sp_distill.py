#!/usr/bin/env python3
"""
SD-Turbo UNet Step-wise Knowledge Distillation.

Goal: recover image quality lost during structural pruning by training the
pruned student UNet to mimic the original (teacher) UNet's denoising outputs.

Core idea — per-step teacher-student matching:
  For each training step, both teacher and student start from the same pure
  Gaussian noise and run 1 to 4 denoising steps. At EVERY denoising step:
    1. Save the teacher's current latent as the shared input (t_latent_in).
    2. Teacher runs one step → t_latent (teacher's next latent).
    3. Student predicts from the SAME t_latent_in (no scheduler.step).
    4. Loss = MSE(s_pred, t_pred) + lambda_l1 * L1(s_pred, t_pred).
       Prediction-space loss avoids EulerAncestral stochastic noise that
       would corrupt latent-space targets.
    5. Student's next step uses t_latent (teacher's output) as input,
       not its own output — so errors never accumulate across steps.

  This is the same "per-step independent" approach used in sensitivity_ld.py,
  applied here as a training signal.

Loss design:
  - MSE + L1 mixed loss: MSE minimises mean error; L1 preserves high-frequency
    detail that MSE blurs away by averaging over valid outputs.
  - Per-timestep loss normalisation: each timestep's loss is divided by a
    running EMA of that timestep's historical loss magnitude. This prevents
    high-noise steps (t=999, large raw MSE) from dominating gradient direction
    at the expense of near-clean steps (t=249, small raw MSE but critical
    for fine detail).

Model EMA (Exponential Moving Average):
  - A shadow copy of student weights is maintained as a weighted average of
    all past training snapshots. EMA weights are used for checkpointing and
    inference; the live weights are used only for gradient updates.
  - Without EMA, a checkpoint saved mid-training may happen to land on an
    oscillation peak and produce noticeably worse inference results.

Teacher/Student dtype split:
  - Teacher: fp16, no grad  (saves memory, frozen weights)
  - Student: fp32, grad     (higher precision for stable gradient updates)
  - Separate schedulers:    teacher_scheduler = deepcopy(pipe.scheduler)
                            Prevents internal state from bleeding between models.

All settings live in prune/sp_distill_config.yaml — just run:

  python prune/sp_distill.py

CLI flags always override the config file:

  python prune/sp_distill.py --main-steps 500    # quick test
  python prune/sp_distill.py --run-config other.yaml
"""
import os
import sys
import gc
import copy
import time
import math
import random
import argparse
import logging
import glob as _glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN_CONFIG = os.path.join(_SCRIPT_DIR, 'sp_distill_config.yaml')


def _load_run_config(path: str) -> dict:
    """Load YAML config file. Returns empty dict if file missing or PyYAML not installed."""
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


# —— Dataset ——————————————————————————————————————————————————————————————————

class DistillDataset(Dataset):
    """Prompt-only dataset.

    SD-Turbo generates purely from text + Gaussian noise (no image conditioning),
    so we only need .txt prompt files. Images in the dataset directory are ignored.

    Directory format:
        dataset/
            prompt_001.txt   ← one prompt per file
            prompt_002.txt
            ...
    """
    def __init__(self, root_dir, max_samples=5000):
        all_txts = sorted(_glob.glob(os.path.join(root_dir, "*.txt")))
        if len(all_txts) == 0:
            raise ValueError(f"No .txt prompt files found in {root_dir}")
        if len(all_txts) > max_samples:
            random.seed(42)           # fixed seed → deterministic subset across runs
            random.shuffle(all_txts)
            self.txt_paths = all_txts[:max_samples]
        else:
            self.txt_paths = all_txts

    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, idx):
        with open(self.txt_paths[idx], "r", encoding="utf-8") as f:
            return f.read().strip()   # strip trailing newline / whitespace


class CachedTeacherDataset(Dataset):
    """Dataset for cache-mode distillation.

    Scans cache_dir for fully-populated trajectory directories and exposes
    them as (prompt_text, prompt_idx, seed) tuples.  Only trajectories where
    every step_0..step_{n_steps-1}.pt file exists are included, so a partially-
    written cache from a previous interrupted run is handled gracefully.

    Use gen_teacher_cache.py to populate the cache before training.
    """

    def __init__(self, prompt_dir, cache_dir, n_steps=4, max_samples=5000):
        from pathlib import Path as _Path
        all_txts = sorted(_glob.glob(os.path.join(prompt_dir, "*.txt")))
        if not all_txts:
            raise ValueError(f"No .txt prompt files found in {prompt_dir}")
        if len(all_txts) > max_samples:
            random.seed(42)           # same seed as DistillDataset for consistent subset
            random.shuffle(all_txts)
            all_txts = all_txts[:max_samples]
        self.txt_paths = all_txts     # list of .txt paths, index = prompt_idx on disk

        # Scan cache_dir for complete trajectory folders (format: "{p_idx:05d}_{seed:04d}/")
        self.items = []
        cache_path = _Path(cache_dir)
        for traj_dir in sorted(cache_path.glob("*_*")):
            if not traj_dir.is_dir():
                continue
            parts = traj_dir.name.split("_")
            if len(parts) != 2:       # skip any dir that doesn't match "int_int" pattern
                continue
            try:
                p_idx, seed = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if p_idx >= len(self.txt_paths):  # cache was built with a larger prompt set
                continue
            # Only include trajectories where ALL n_steps step files exist.
            # A partially-written dir (e.g. interrupted gen_teacher_cache run) is silently skipped.
            if all((traj_dir / f"step_{i}.pt").exists() for i in range(n_steps)):
                self.items.append((p_idx, seed))

        if not self.items:
            raise ValueError(
                f"No complete trajectories found in {cache_dir}. "
                f"Run gen_teacher_cache.py first."
            )
        print(f"CachedTeacherDataset: {len(self.items)} trajectories "
              f"({len(self.txt_paths)} prompts)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p_idx, seed = self.items[idx]
        with open(self.txt_paths[p_idx], "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        # Return (prompt, p_idx, seed) so the training loop can reconstruct the cache path
        return prompt, p_idx, seed


# —— LR schedule ——————————————————————————————————————————————————————————————

def build_cosine_lr_lambda(warmup_steps, total_steps, lr_max, lr_min, decay_start=None):
    """Build a LambdaLR multiplier: linear warmup → optional plateau → cosine decay.

    Timeline:
      [0, warmup_steps)      : LR rises linearly from lr_min to lr_max
      [warmup_steps,
       decay_start)          : LR stays at lr_max  (plateau, skipped if decay_start=warmup_steps)
      [decay_start,
       total_steps]          : LR decays via cosine from lr_max down to lr_min

    Args:
        warmup_steps:  number of linear warmup steps
        total_steps:   total training steps (end of cosine decay)
        lr_max:        peak learning rate (used as base for LambdaLR)
        lr_min:        floor learning rate at end of decay
        decay_start:   step to start cosine decay (default = warmup_steps, no plateau)

    Returns a function f(step) → multiplier in [lr_min/lr_max, 1.0].
    """
    # LambdaLR multiplies the base lr (lr_max) by the returned ratio, so the ratio range is [min_ratio, 1.0]
    min_ratio = lr_min / lr_max
    if decay_start is None:
        decay_start = warmup_steps  # no plateau: decay begins right after warmup

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: ratio grows from min_ratio → 1.0 over warmup_steps
            return min_ratio + (1.0 - min_ratio) * (step / warmup_steps)
        if step < decay_start:
            # Plateau: hold at peak lr_max (ratio = 1.0) until decay_start
            return 1.0
        # Cosine decay: ratio falls smoothly from 1.0 → min_ratio
        # progress ∈ [0, 1] measures how far into the decay window we are
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        # cos(0)=1 → ratio=1.0 at start; cos(π)=-1 → ratio=min_ratio at end
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


# —— Feature distillation helpers ————————————————————————————————————————————

def _get_module_by_dotpath(model: nn.Module, dotpath: str) -> nn.Module:
    """Navigate a module tree by dot-path, supporting ModuleList indices."""
    module = model
    for part in dotpath.split('.'):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _first_tensor(output):
    """Extract the primary hidden-state tensor from a block forward output."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class FeatureMapRegistry:
    """Forward-hook registry for large UNet block output features."""

    def __init__(self, detach: bool) -> None:
        self.detach = detach
        self.captured: dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self, model: nn.Module, layer_keys: list[str]) -> None:
        for key in layer_keys:
            module = _get_module_by_dotpath(model, key)

            def _hook(_module, _inputs, output, layer_key=key):
                feat = _first_tensor(output)
                if feat is not None:
                    feat = feat.float()
                    self.captured[layer_key] = feat.detach() if self.detach else feat

            self._handles.append(module.register_forward_hook(_hook))

    def clear(self) -> None:
        self.captured.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def normalized_feature_mse(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Feature KD loss.

    If shapes match, align z-score-normalized tensors.  If pruning changed the
    channel count, fall back to a channel-agnostic spatial energy map.
    """
    s = student_feat.float()
    t = teacher_feat.float().detach()

    if s.shape == t.shape:
        reduce_dims = tuple(range(1, s.ndim))
        s_mean = s.mean(dim=reduce_dims, keepdim=True)
        t_mean = t.mean(dim=reduce_dims, keepdim=True)
        s_std = s.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        t_std = t.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        return F.mse_loss((s - s_mean) / s_std, (t - t_mean) / t_std)

    if s.ndim == 4 and t.ndim == 4:
        s_map = s.pow(2).mean(dim=1, keepdim=True)
        t_map = t.pow(2).mean(dim=1, keepdim=True)
        if s_map.shape[-2:] != t_map.shape[-2:]:
            s_map = F.interpolate(s_map, size=t_map.shape[-2:], mode="bilinear", align_corners=False)
        s_map = F.normalize(s_map.flatten(1), p=2, dim=1)
        t_map = F.normalize(t_map.flatten(1), p=2, dim=1)
        return F.mse_loss(s_map, t_map)

    return s.new_zeros(())


# —— Model loading ————————————————————————————————————————————————————————————

def load_teacher_and_pipe(base_model_id, device):
    """Load the original (unpruned) SD-Turbo pipeline as the teacher.

    Teacher UNet is:
      - fp16 (saves ~half VRAM vs fp32)
      - eval mode, no gradients (frozen — we never update the teacher)

    Returns (pipe, teacher_unet). pipe is kept for tokenizer, text_encoder,
    VAE, and scheduler — all shared with the student training loop.
    """
    from diffusers import StableDiffusionPipeline
    from prune import sp_core
    base_model_id = sp_core.resolve_base_model_source(base_model_id)  # resolves local path or HF hub ID
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16,  # fp16: halves VRAM (~1.7 GB vs 3.4 GB)
        safety_checker=None, requires_safety_checker=False,  # not needed for training
    ).to(device)
    pipe.unet.eval()                    # disable dropout / BatchNorm training mode
    pipe.unet.requires_grad_(False)     # freeze all teacher weights — no gradients ever computed
    return pipe, pipe.unet


def load_student_unet(safetensors_path, device):
    """Load the pruned student UNet for fine-tuning.

    Student UNet is:
      - fp32 (higher precision for gradient updates)
      - train mode, gradients enabled
      - gradient checkpointing ON (trades compute for memory — recomputes
        activations during backward instead of storing them)

    Expects a matching .config.json alongside the .safetensors file,
    written by the pruning pipeline (sp_apply.py / sp_core.py).
    """
    from prune.pruned_rebuild import create_unet_from_safetensors
    from prune.sp_core import sync_diffusers_attributes
    config_path = safetensors_path.replace('.safetensors', '.config.json')  # written by sp_apply.py
    student_unet = create_unet_from_safetensors(safetensors_path, config_path)
    student_unet = student_unet.to(device=device, dtype=torch.float32)  # fp32 for stable gradient updates
    student_unet.train()                          # enable dropout/BN training mode
    student_unet.enable_gradient_checkpointing()  # recompute activations on backward → saves activation VRAM
    sync_diffusers_attributes(student_unet)        # copy config/dtype attrs expected by diffusers scheduler
    print(f"Student UNet: {sum(p.numel() for p in student_unet.parameters())/1e6:.1f}M params, fp32, grad_ckpt ON")
    return student_unet


# —— EMA helpers ——————————————————————————————————————————————————————————————

def init_ema_params(student_unet):
    """Initialise the EMA shadow weights as an exact copy of student parameters.

    Only tracks named parameters (weight/bias tensors), not buffers
    (running_mean, running_var, etc.). Buffers are read directly from the
    live model when saving.

    Stored on CPU (float32) to save ~3.25 GB of GPU VRAM — the EMA copy is
    only read at checkpoint time, so the CPU↔GPU sync cost is negligible.
    """
    # detach: no grad needed for shadow copy; cpu().float(): saves ~3.25 GB GPU VRAM
    return {name: param.detach().cpu().float()
            for name, param in student_unet.named_parameters()}


def update_ema_params(ema_params, student_unet, decay):
    """EMA update: ema = decay * ema + (1 - decay) * current.

    Called once after every optimizer.step(). decay=0.9999 means the EMA
    weights lag ~10,000 steps behind the live weights — slow enough to smooth
    out oscillations, fast enough to track genuine improvement.
    """
    with torch.no_grad():
        for name, param in student_unet.named_parameters():
            # In-place: ema = decay * ema + (1 - decay) * live_param
            # mul_ and add_ avoid allocating a new tensor each step
            ema_params[name].mul_(decay).add_(param.data.cpu().float(), alpha=1.0 - decay)


def get_ema_state_dict(student_unet, ema_params):
    """Build a full state dict using EMA weights for parameters.

    Starts from the live model's state dict (which includes all buffers such
    as BatchNorm statistics), then overwrites every parameter entry with its
    EMA counterpart. The result is safe to pass directly to save_file().
    """
    state = student_unet.state_dict()  # includes buffers (e.g. BN running_mean) from live model
    for name in ema_params:
        state[name] = ema_params[name]  # overwrite parameter entries with their EMA counterparts
    return state


# —— Checkpoint saving ————————————————————————————————————————————————————————

def save_checkpoint(student_unet, path, step, loss_val, ema_params=None):
    """Save student weights + metadata to safetensors + JSON sidecar.

    If ema_params is provided, the EMA (averaged) weights are saved instead
    of the live training weights. EMA weights should always be used for
    inference — they are more stable than any individual training snapshot.
    """
    import json
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if ema_params is not None:
        # EMA weights: stable, averaged over training history → preferred for inference
        state = get_ema_state_dict(student_unet, ema_params)
    else:
        # Fallback: live training weights (used only if EMA not yet initialised)
        state = student_unet.state_dict()
    save_file({k: v.cpu() for k, v in state.items()}, path)
    meta = {
        "distill_step": step,
        "distill_loss": float(loss_val),
        "is_ema": ema_params is not None,
    }
    if hasattr(student_unet, 'config') and student_unet.config is not None:
        try:
            meta['model_config'] = (
                student_unet.config.to_dict()
                if hasattr(student_unet.config, 'to_dict')
                else dict(student_unet.config)
            )
        except Exception:
            pass
    with open(path.replace('.safetensors', '.config.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    tag = " (EMA)" if ema_params is not None else ""
    print(f"  Saved{tag}: {path} ({os.path.getsize(path)/1024**3:.2f}GB) step={step}")


# —— Main ——————————————————————————————————————————————————————————————————————

def main():
    # ── Config loading ────────────────────────────────────────────────────────
    # Two-pass argument parsing: first read --run-config to load YAML defaults,
    # then parse all args with those defaults. CLI flags always win over YAML.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--run-config', default=DEFAULT_RUN_CONFIG)
    pre_args, _ = pre.parse_known_args()
    rc = _load_run_config(pre_args.run_config)
    if rc:
        print(f"Run config: {pre_args.run_config}")

    # Apply cache paths from config before HuggingFace imports
    hf_cache = rc.get('hf_cache', '.hf_cache')
    tmp_dir  = rc.get('tmp_dir',  '.tmp')
    os.environ.setdefault('HF_HOME',           hf_cache)
    os.environ.setdefault('TRANSFORMERS_CACHE', hf_cache)
    os.environ.setdefault('TMPDIR',             tmp_dir)

    def _rc(key, default=None):
        """Get value from YAML config, falling back to default."""
        return rc.get(key, default)

    parser = argparse.ArgumentParser(
        description="SD-Turbo distillation — configure via sp_distill_config.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--run-config', default=DEFAULT_RUN_CONFIG)

    # Paths
    parser.add_argument('--hf-cache',     default=_rc('hf_cache',    '.hf_cache'))
    parser.add_argument('--tmp-dir',      default=_rc('tmp_dir',     '.tmp'))
    parser.add_argument('--base-model',   default=_rc('base_model',  'models/sd-turbo'))
    parser.add_argument('--pruned-model', default=_rc('pruned_model', 'models/round1_pruned.safetensors'))
    parser.add_argument('--device',       default=_rc('device'))

    # Data
    parser.add_argument('--dataset',     default=_rc('dataset',     'dataset'))
    parser.add_argument('--max-samples', type=int, default=_rc('max_samples', 5000))

    # Output
    parser.add_argument('--output-dir',        default=_rc('output_dir',        'models/distill_r1'))
    parser.add_argument('--save-every',        type=int, default=_rc('save_every',        2000))
    parser.add_argument('--keep-checkpoints',  type=int, default=_rc('keep_checkpoints',  3))
    parser.add_argument('--log-every',         type=int, default=_rc('log_every',         100))

    # Training phases — both at native 512px resolution
    # Phase 1 (Main):     primary quality recovery, higher LR range
    # Phase 2 (Finetune): low LR final polish, set to 0 to skip
    parser.add_argument('--resolution',      type=int, default=_rc('resolution',      512))
    parser.add_argument('--main-steps',      type=int, default=_rc('main_steps',      7000))
    parser.add_argument('--finetune-steps',  type=int, default=_rc('finetune_steps',  1000))

    # Optimizer (AdamW + cosine LR decay)
    parser.add_argument('--lr-max',          type=float, default=_rc('lr_max',          2e-5))
    parser.add_argument('--lr-min',          type=float, default=_rc('lr_min',          1e-6))
    parser.add_argument('--lr-warmup-steps', type=int,   default=_rc('lr_warmup_steps', 200))
    parser.add_argument('--lr-decay-start',  type=int,   default=_rc('lr_decay_start'))   # None = start after warmup
    parser.add_argument('--weight-decay',    type=float, default=_rc('weight_decay',    0.01))
    parser.add_argument('--grad-clip',       type=float, default=_rc('grad_clip',       1.0))  # L2 norm cap

    # Loss
    parser.add_argument('--lambda-pred', type=float, default=_rc('lambda_pred', 1.0),
                        help="Weight for prediction-space teacher-student loss.")
    parser.add_argument('--lambda-l1',  type=float, default=_rc('lambda_l1', 0.1),
                        help="Weight of L1 loss term added alongside MSE. "
                             "L1 preserves high-frequency detail that pure MSE blurs away. "
                             "0.1 is conservative; try up to 0.2 if output is still blurry.")

    # Training details
    parser.add_argument('--batch-size',        type=int,   default=_rc('batch_size',        2))
    parser.add_argument('--grad-accum-steps',  type=int,   default=_rc('grad_accum_steps',  1),
                        help="Accumulate gradients over N batches before optimizer.step(). "
                             "Effective batch size = batch_size × grad_accum_steps. "
                             "Use 4 to simulate batch_size=8 without extra VRAM.")
    parser.add_argument('--n-steps-min',  type=int, default=_rc('n_steps_min',  4))  # min denoising steps per sample
    parser.add_argument('--n-steps-max',  type=int, default=_rc('n_steps_max',  4))  # max denoising steps per sample
    parser.add_argument('--num-workers',  type=int, default=_rc('num_workers',  2))
    parser.add_argument('--gc-every',     type=int, default=_rc('gc_every',     200))

    # Model EMA
    parser.add_argument('--ema-decay',      type=float, default=_rc('ema_decay',      0.9999),
                        help="EMA decay for model weights. 0.9999 means the shadow weights "
                             "lag ~10,000 optimizer steps behind live weights. "
                             "EMA weights are always used for checkpointing.")
    parser.add_argument('--ema-norm-decay', type=float, default=_rc('ema_norm_decay', 0.99),
                        help="EMA decay for per-timestep loss normalisation. "
                             "Tracks the running mean loss per timestep so each timestep "
                             "contributes ~equally to the gradient regardless of noise scale.")

    # Attention map distillation
    parser.add_argument('--teacher-cache-dir', default=_rc('teacher_cache_dir'),
                        help="Path to pre-computed teacher cache (gen_teacher_cache.py). "
                             "When set, teacher UNet is offloaded from GPU — saves ~3-5 GB VRAM.")
    parser.add_argument('--lambda-attn', type=float, default=_rc('lambda_attn', 0.0),
                        help="Weight of attention map MSE loss. 0 = disabled. "
                             "Recommended start: 0.02. Applies to mid/deep zone layers only.")
    parser.add_argument('--attn-warmup-steps', type=int,
                        default=_rc('attn_warmup_steps', 500),
                        help="Ramp lambda_attn linearly from 0 → target over this many steps.")
    parser.add_argument('--attn-layers', nargs='*',
                        default=_rc('attn_layers', []),
                        help="Dot-paths of attention modules to distil. "
                             "Empty list disables attention distillation.")
    parser.add_argument('--lambda-feat', type=float, default=_rc('lambda_feat', 0.0),
                        help="Weight of normalized block feature distillation. 0 = disabled.")
    parser.add_argument('--feat-layers', nargs='*',
                        default=_rc('feat_layers', []),
                        help="Dot-paths of large UNet blocks whose output features are distilled.")

    args = parser.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'distill.log'), mode='w'),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger(__name__)

    total_steps = args.main_steps + args.finetune_steps
    log.info(f"Distillation: Main={args.main_steps} + Finetune={args.finetune_steps} = {total_steps} total @ {args.resolution}px")
    log.info(f"LR: {args.lr_max} -> {args.lr_min}, warmup={args.lr_warmup_steps}")
    log.info(f"Loss: {args.lambda_pred} * (MSE + {args.lambda_l1} * L1) "
             f"+ {args.lambda_feat} * feature  |  Timestep normalisation EMA decay={args.ema_norm_decay}")
    log.info(f"Batch: {args.batch_size} × grad_accum={args.grad_accum_steps} → effective_bs={args.batch_size * args.grad_accum_steps}")
    log.info(f"Model EMA decay={args.ema_decay}")
    log.info(f"Base model: {args.base_model}  |  Student: {args.pruned_model}")

    use_cache       = bool(args.teacher_cache_dir)
    use_attn_distill = args.lambda_attn > 0 and bool(args.attn_layers)
    use_feat_distill = args.lambda_feat > 0 and bool(args.feat_layers)
    if use_cache:
        if not Path(args.teacher_cache_dir).exists():
            raise FileNotFoundError(
                f"--teacher-cache-dir not found: {args.teacher_cache_dir}\n"
                f"Run gen_teacher_cache.py first to pre-compute teacher trajectories."
            )
        log.info(f"Cache mode ON: teacher UNet will be offloaded from GPU — cache: {args.teacher_cache_dir}")
        if use_feat_distill:
            raise ValueError("Feature distillation is not supported with teacher_cache_dir yet; run live teacher mode.")
    if use_attn_distill:
        log.info(f"Attn distillation ON: λ={args.lambda_attn}, warmup={args.attn_warmup_steps}, "
                 f"layers={len(args.attn_layers)}")
    if use_feat_distill:
        log.info(f"Feature distillation ON: λ={args.lambda_feat}, layers={len(args.feat_layers)}")

    # ── Model loading ─────────────────────────────────────────────────────────
    log.info("Loading teacher pipeline...")
    pipe, teacher_unet = load_teacher_and_pipe(args.base_model, device)

    # Deep-copy the scheduler for teacher so its internal state (sigma tables,
    # _step_index) is completely isolated from the student's scheduler.
    # Without this, teacher.step() and student.step() would share state and
    # corrupt each other's sigma lookups.
    teacher_scheduler = copy.deepcopy(pipe.scheduler)

    # In cache mode: offload teacher UNet to CPU immediately — the training loop
    # will load t_pred and attention maps from disk instead of running it on GPU.
    # This frees ~1.7 GB (weights) + ~2-3 GB (forward activations), enabling
    # a larger batch_size on the same hardware.
    if use_cache:
        log.info("Offloading teacher UNet to CPU (cache mode)...")
        pipe.unet = pipe.unet.to('cpu')
        teacher_unet = None    # prevent accidental GPU usage

    log.info("Loading student UNet...")
    student_unet = load_student_unet(args.pruned_model, device)

    # Initialise EMA shadow weights from the student's current parameters.
    # From this point on, ema_params is updated after every optimizer.step().
    ema_params = init_ema_params(student_unet)
    log.info(f"Model EMA initialised — {len(ema_params)} parameter tensors tracked")

    # ── Attention map distillation setup ─────────────────────────────────────
    # student_registry: always installed when attn distillation is active.
    # teacher_registry: only for live (non-cache) mode — captures teacher maps
    #                   during each teacher forward pass.
    student_registry = None
    teacher_registry = None
    if use_attn_distill:
        from prune.attn_map_capture import AttentionMapRegistry
        student_registry = AttentionMapRegistry()
        student_registry.register(student_unet, args.attn_layers)
        log.info(f"Student attention hooks registered on {len(args.attn_layers)} layers")
        if not use_cache:
            teacher_registry = AttentionMapRegistry()
            teacher_registry.register(teacher_unet, args.attn_layers)
            log.info("Teacher attention hooks registered (live mode)")

    # Per-timestep EMA normalisation state for attention loss
    # Key format: "a{timestep}" to avoid collision with pred-loss keys
    ema_attn_per_t: dict[str, float] = {}

    student_feature_registry = None
    teacher_feature_registry = None
    if use_feat_distill:
        student_feature_registry = FeatureMapRegistry(detach=False)
        teacher_feature_registry = FeatureMapRegistry(detach=True)
        student_feature_registry.register(student_unet, args.feat_layers)
        teacher_feature_registry.register(teacher_unet, args.feat_layers)
        log.info(f"Feature hooks registered on {len(args.feat_layers)} large blocks")

    # Per-timestep EMA normalisation state for feature loss
    ema_feat_per_t: dict[str, float] = {}

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    # Only update parameters that require grad (student weights, not frozen layers)
    # Only pass params with requires_grad=True; frozen layers (if any) are excluded
    trainable_params = [p for p in student_unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr_max, weight_decay=args.weight_decay)

    # LR schedule is continuous across both phases — the phase boundary does NOT
    # reset the LR. This means Finetune naturally runs at a lower LR since it
    # starts after the cosine decay has already progressed through Main.
    lr_lambda = build_cosine_lr_lambda(
        args.lr_warmup_steps, total_steps, args.lr_max, args.lr_min, args.lr_decay_start)
    # LambdaLR multiplies the optimizer's base lr (lr_max) by lr_lambda(step) each call
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Per-timestep loss normalisation state ─────────────────────────────────
    # Maps each timestep integer (999, 749, 499, 249) to its running EMA of
    # raw loss magnitude. Used to normalise each step's contribution so that
    # high-noise steps (large raw MSE) don't dominate gradient direction.
    # Initialised lazily on first encounter to avoid assuming a fixed schedule.
    ema_loss_per_t: dict[int, float] = {}

    # ── Training phases ───────────────────────────────────────────────────────
    phases = [
        {"name": "Main",     "resolution": args.resolution, "steps": args.main_steps},
        {"name": "Finetune", "resolution": args.resolution, "steps": args.finetune_steps},
    ]

    global_step  = 0   # counts optimizer steps (increments every grad_accum_steps batches)
    accum_step   = 0   # position within the current accumulation window [0, grad_accum_steps)
    running_loss = 0.0 # cumulative loss for final avg_loss calculation
    loss_val     = 0.0 # loss at last optimizer step (used for checkpoint metadata)
    t_start      = time.time()

    for phase in phases:
        res         = phase["resolution"]
        phase_steps = phase["steps"]
        phase_name  = phase["name"]
        if phase_steps <= 0:
            continue  # allow skipping a phase by setting its steps to 0

        log.info(f"\n{'='*60}\nStarting {phase_name}: {phase_steps} steps @ {res}px\n{'='*60}")
        if use_cache:
            _n_steps_for_cache = args.n_steps_max  # cache was built with this step count
            dataset = CachedTeacherDataset(
                args.dataset, args.teacher_cache_dir,
                n_steps=_n_steps_for_cache, max_samples=args.max_samples,
            )
        else:
            dataset = DistillDataset(args.dataset, max_samples=args.max_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        data_iter  = iter(dataloader)
        phase_t    = time.time()

        # SD-Turbo latent space is 1/8 of pixel resolution (VAE factor)
        latent_h = res // 8
        latent_w = res // 8

        for _ in range(phase_steps * args.grad_accum_steps):
            step_t = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart dataloader when exhausted (dataset loops indefinitely)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Unpack batch — DistillDataset yields strings; CachedTeacherDataset
            # yields (prompt, prompt_idx, seed) triples.
            if use_cache:
                prompts, prompt_indices, seeds = batch
                prompts        = list(prompts)
                prompt_indices = prompt_indices.tolist()
                seeds          = seeds.tolist()
            else:
                prompts        = list(batch)
                prompt_indices = None
                seeds          = None

            # ── Denoising step count ──────────────────────────────────────────
            # Defaults to fixed 4 steps (n_steps_min = n_steps_max = 4 in config),
            # matching SD-Turbo's native schedule [999, 749, 499, 249].
            # Fixed steps give a stable, consistent gradient signal.
            # Optionally set n_steps_min < n_steps_max to randomize (e.g. 1–4),
            # but lower step counts produce very blurry outputs so fixed 4 is preferred.
            # Both schedulers must be synced to the same step count.
            _n_steps = random.randint(args.n_steps_min, args.n_steps_max)
            pipe.scheduler.set_timesteps(_n_steps, device=device)         # student scheduler
            teacher_scheduler.set_timesteps(_n_steps, device=device)      # teacher scheduler (isolated copy)
            ts_schedule = pipe.scheduler.timesteps  # e.g. tensor([999, 749, 499, 249])

            # ── Text conditioning (shared by teacher and student) ─────────────
            with torch.no_grad():  # text encoder is frozen; no need to track gradients
                text_input = pipe.tokenizer(
                    list(prompts), padding="max_length", max_length=77,  # CLIP max token length
                    truncation=True, return_tensors="pt",
                ).to(device)
                # [batch, 77, 768] — cross-attention context fed to both UNets
                encoder_hidden_states = pipe.text_encoder(
                    text_input.input_ids)[0].to(dtype=torch.float32)

            # ── Initialize latents ────────────────────────────────────────────
            bsz = len(prompts)
            if not use_cache:
                # Live mode: fresh Gaussian noise each iteration.
                # init_noise_sigma scales to match scheduler's expected sigma at t=999.
                latents = torch.randn(
                    (bsz, 4, latent_h, latent_w),  # 4 = VAE latent channels
                    device=device, dtype=torch.float32,
                ) * pipe.scheduler.init_noise_sigma  # scale to noise level at first timestep
                # t_latent is the teacher's current denoising state; updated each step by teacher.step()
                t_latent = latents.clone()
            # Cache mode: input_latent is loaded per-step from disk (no running state needed).

            total_loss      = torch.zeros(1, device=device, dtype=torch.float32)  # normalised pred loss, accumulates over steps
            total_attn_loss = torch.zeros(1, device=device, dtype=torch.float32)  # normalised attn loss, accumulates over steps
            total_feat_loss = torch.zeros(1, device=device, dtype=torch.float32)  # normalised feature loss, accumulates over steps
            total_raw_loss  = 0.0   # un-normalised MSE+L1 sum — detached float, logging only
            total_raw_attn  = 0.0   # un-normalised attn MSE sum — logging only
            total_raw_feat  = 0.0   # un-normalised feature MSE sum — logging only

            # ── Pre-load full trajectory from disk (cache mode only) ──────────
            # Loading all steps here instead of inside the timestep loop avoids
            # blocking the GPU N times per training iteration on synchronous I/O.
            # Each torch.load() call stalls the CUDA stream until the read
            # completes; front-loading them lets the GPU run the student forward
            # passes without waiting on disk.
            if use_cache:
                cache_root = Path(args.teacher_cache_dir)
                all_step_data = []     # list[list[dict]] — [step_i][b_i]
                for si in range(_n_steps):
                    step_batch = []
                    for b_i in range(bsz):
                        sp = (cache_root
                              / f"{prompt_indices[b_i]:05d}_{seeds[b_i]:04d}"
                              / f"step_{si}.pt")
                        step_batch.append(
                            torch.load(sp, map_location='cpu', weights_only=True)
                        )
                    all_step_data.append(step_batch)

            # ── Per-step denoising loop ───────────────────────────────────────
            for step_i, t in enumerate(ts_schedule):

                # ── Teacher outputs ───────────────────────────────────────────
                t_attn_maps: dict = {}

                if use_cache:
                    # Data was pre-loaded above — just index, no I/O here.
                    step_data_batch = all_step_data[step_i]
                    t_latent_in = torch.cat(
                        [d['input_latent'].float() for d in step_data_batch]
                    ).to(device)
                    t_pred = torch.cat(
                        [d['t_pred'].float() for d in step_data_batch]
                    ).to(device)
                    t_latent = torch.cat(
                        [d['next_latent'].float() for d in step_data_batch]
                    ).to(device)
                    # Attention maps: stack per-layer across batch items
                    if use_attn_distill:
                        for key in args.attn_layers:
                            maps = [d['attn_maps'].get(key) for d in step_data_batch]
                            if all(m is not None for m in maps):
                                t_attn_maps[key] = torch.cat(
                                    [m.float() for m in maps]
                                ).to(device)

                else:
                    # Live mode: run teacher on GPU this step.
                    # _step_index must be set explicitly because both schedulers share
                    # the same timestep table but are called in sequence.
                    t_latent_in = t_latent.detach().clone()  # snapshot teacher's current latent as shared input

                    with torch.no_grad():
                        if teacher_registry is not None:
                            teacher_registry.clear()  # discard previous step's captured maps
                        if teacher_feature_registry is not None:
                            teacher_feature_registry.clear()
                        # Must set _step_index manually; EulerAncestral tracks it internally and
                        # would use wrong sigma if we call scale_model_input out of order.
                        teacher_scheduler._step_index = step_i
                        t_in   = teacher_scheduler.scale_model_input(t_latent_in, t).to(torch.float16)  # sigma-scaled input
                        t_pred = teacher_unet(t_in, t, encoder_hidden_states.to(torch.float16)).sample.to(torch.float32)
                        teacher_scheduler._step_index = step_i  # reset again before .step() (scheduler advances index internally)
                        t_latent = teacher_scheduler.step(t_pred, t, t_latent_in).prev_sample.to(torch.float32)  # teacher's next latent

                        if teacher_registry is not None:
                            for key in args.attn_layers:
                                if key in teacher_registry.captured:
                                    t_attn_maps[key] = teacher_registry.captured[key].detach()  # no grad needed for target
                        t_feat_maps = (
                            dict(teacher_feature_registry.captured)
                            if teacher_feature_registry is not None else {}
                        )

                # ── Student step (fp32, grad) ─────────────────────────────────
                # Student uses t_latent_in (same as teacher's input this step),
                # NOT its own previous output. This prevents error accumulation:
                # a blurry student step-1 output would degrade step-2 inputs
                # and corrupt the gradient signal across all remaining steps.
                if student_registry is not None:
                    student_registry.clear()          # clear previous step's captured maps before forward
                if student_feature_registry is not None:
                    student_feature_registry.clear()
                pipe.scheduler._step_index = step_i   # sync student scheduler index to match teacher
                s_in   = pipe.scheduler.scale_model_input(t_latent_in, t)  # same sigma scaling as teacher
                s_pred = student_unet(s_in, t, encoder_hidden_states).sample  # fp32 forward, grad tracked
                # No scheduler.step() for student — EulerAncestral samples fresh
                # random noise inside step(), so student and teacher latents would
                # diverge stochastically. Loss on predictions bypasses that entirely.

                # ── Prediction loss with timestep normalisation ───────────────
                # Compare UNet epsilon predictions directly (prediction space).
                # t_pred is a constant target (no_grad / from cache); s_pred has grad.
                # L1 alongside MSE preserves high-frequency detail that MSE blurs.
                s_f = s_pred.float()   # ensure fp32 for stable loss computation
                t_f = t_pred.float()   # teacher target — no gradient flows through this
                # MSE: minimises mean squared error (good for overall accuracy)
                # L1:  preserves sharp edges / high-frequency detail that MSE blurs by averaging
                raw_loss = F.mse_loss(s_f, t_f) + args.lambda_l1 * F.l1_loss(s_f, t_f)

                # Per-timestep EMA normalisation:
                # raw MSE at t=999 (high noise) can be 100× larger than at t=249 (near-clean).
                # Without normalisation, gradients at noisy steps drown out near-clean steps.
                # Dividing by the EMA makes each timestep contribute ~equally to the gradient.
                t_key = int(t.item())  # e.g. 999, 749, 499, 249
                if t_key not in ema_loss_per_t:
                    ema_loss_per_t[t_key] = raw_loss.item()  # initialise on first encounter
                else:
                    # Running EMA of loss magnitude for this specific timestep
                    ema_loss_per_t[t_key] = (
                        args.ema_norm_decay * ema_loss_per_t[t_key]
                        + (1.0 - args.ema_norm_decay) * raw_loss.item()
                    )
                norm_loss = raw_loss / (ema_loss_per_t[t_key] + 1e-8)  # +1e-8 prevents div-by-zero

                total_loss     = total_loss + norm_loss   # accumulate over all denoising steps
                total_raw_loss += raw_loss.item()          # float — no grad, for logging only

                # ── Large-block feature distillation ─────────────────────────
                # Compare normalized intermediate block outputs at the same
                # timestep.  Shapes that changed after pruning fall back to a
                # channel-agnostic spatial energy map inside normalized_feature_mse.
                if student_feature_registry is not None and t_feat_maps:
                    step_feat = torch.zeros(1, device=device, dtype=torch.float32)
                    n_feat = 0
                    for key in args.feat_layers:
                        if key in student_feature_registry.captured and key in t_feat_maps:
                            step_feat = step_feat + normalized_feature_mse(
                                student_feature_registry.captured[key],
                                t_feat_maps[key].to(device),
                            )
                            n_feat += 1
                    if n_feat > 0:
                        step_feat = step_feat / n_feat
                        feat_t_key = f"f{t_key}"
                        if feat_t_key not in ema_feat_per_t:
                            ema_feat_per_t[feat_t_key] = step_feat.item()
                        else:
                            ema_feat_per_t[feat_t_key] = (
                                args.ema_norm_decay * ema_feat_per_t[feat_t_key]
                                + (1.0 - args.ema_norm_decay) * step_feat.item()
                            )
                        norm_feat = step_feat / (ema_feat_per_t[feat_t_key] + 1e-8)
                        total_feat_loss = total_feat_loss + norm_feat * args.lambda_feat
                        total_raw_feat += step_feat.item()

                # ── Attention map loss ────────────────────────────────────────
                # MSE between student and teacher head-averaged attention maps,
                # summed over hooked layers and normalised per timestep (same EMA
                # scheme as prediction loss).  Lambda is ramped up gradually so the
                # student first establishes basic prediction quality.
                if student_registry is not None and t_attn_maps:
                    step_attn = torch.zeros(1, device=device, dtype=torch.float32)
                    n_matched = 0
                    for key in args.attn_layers:
                        if key in student_registry.captured and key in t_attn_maps:
                            # MSE between student and teacher head-averaged attention maps for this layer
                            step_attn = step_attn + F.mse_loss(
                                student_registry.captured[key],
                                t_attn_maps[key].to(device).detach(),  # teacher map is a fixed target
                            )
                            n_matched += 1
                    if n_matched > 0:
                        step_attn = step_attn / n_matched  # average across all matched layers
                        attn_t_key = f"a{t_key}"           # "a" prefix avoids collision with pred-loss keys
                        if attn_t_key not in ema_attn_per_t:
                            ema_attn_per_t[attn_t_key] = step_attn.item()
                        else:
                            # Same per-timestep EMA normalisation scheme as prediction loss
                            ema_attn_per_t[attn_t_key] = (
                                args.ema_norm_decay * ema_attn_per_t[attn_t_key]
                                + (1.0 - args.ema_norm_decay) * step_attn.item()
                            )
                        norm_attn = step_attn / (ema_attn_per_t[attn_t_key] + 1e-8)
                        # Linear ramp-up: prevent attn loss from interfering before
                        # prediction quality stabilises (first attn_warmup_steps steps)
                        # min(1.0, ...) clamps ramp to [0, 1] so lambda never exceeds target
                        lambda_now = args.lambda_attn * min(
                            1.0, global_step / max(1, args.attn_warmup_steps)
                        )
                        total_attn_loss = total_attn_loss + norm_attn * lambda_now
                        total_raw_attn  += step_attn.item()

            # Average across denoising steps, then scale down for gradient accumulation.
            # Both pred loss and attn loss are averaged over ts_schedule length.
            # Dividing by grad_accum_steps keeps gradient magnitude consistent with
            # a single-step update — gradients accumulate additively over N batches.
            loss = (
                (total_loss * args.lambda_pred + total_feat_loss + total_attn_loss) / len(ts_schedule)
            ) / args.grad_accum_steps

            # ── NaN guard ─────────────────────────────────────────────────────
            # Skip the update if loss is invalid (rare but can happen early in
            # training with an aggressively pruned model).
            if torch.isnan(loss) or torch.isinf(loss):
                log.warning(f"NaN/Inf at batch {global_step * args.grad_accum_steps + accum_step + 1}, skipping")
                if accum_step == 0:
                    optimizer.zero_grad(set_to_none=True)
                continue

            # ── Gradient accumulation ─────────────────────────────────────────
            # zero_grad only at the START of each accumulation window.
            # Gradients from subsequent batches in the same window ADD to the
            # existing gradient buffer — this is the accumulation mechanism.
            if accum_step == 0:
                optimizer.zero_grad(set_to_none=True)

            loss.backward()
            accum_step += 1

            # Only update weights once the full accumulation window is complete.
            if accum_step < args.grad_accum_steps:
                continue
            accum_step = 0

            # ── Optimizer step ────────────────────────────────────────────────
            # Gradient clipping: rescale all gradients if their global L2 norm
            # exceeds grad_clip. Prevents a single bad batch from blowing up weights.
            torch.nn.utils.clip_grad_norm_(student_unet.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            lr_sched.step()

            # ── Model EMA update ──────────────────────────────────────────────
            # Update EMA shadow weights with the newly updated live weights.
            # Must happen AFTER optimizer.step() so the EMA sees the updated params.
            update_ema_params(ema_params, student_unet, args.ema_decay)

            global_step += 1
            # Scale loss back up for logging (undo the grad_accum division so the
            # displayed value reflects the true per-sample loss magnitude)
            loss_val      = loss.item() * args.grad_accum_steps
            raw_loss_val  = total_raw_loss / len(ts_schedule)
            raw_attn_val  = total_raw_attn  / len(ts_schedule) if use_attn_distill else 0.0
            raw_feat_val  = total_raw_feat  / len(ts_schedule) if use_feat_distill else 0.0
            running_loss += loss_val

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % args.log_every == 0 or global_step <= 3:
                elapsed   = time.time() - t_start
                step_time = time.time() - step_t
                eta       = step_time * (total_steps - global_step)
                attn_str  = f" attn={raw_attn_val:.5f}" if use_attn_distill else ""
                feat_str  = f" feat={raw_feat_val:.5f}" if use_feat_distill else ""
                log.info(
                    f"[{phase_name}] {global_step}/{total_steps} | "
                    f"loss={loss_val:.6f} raw={raw_loss_val:.6f}{feat_str}{attn_str} n={_n_steps} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"{step_time:.2f}s/it | elapsed={elapsed/3600:.2f}h ETA={eta/3600:.2f}h"
                )

            # ── Periodic checkpoint ───────────────────────────────────────────
            if global_step % args.save_every == 0:
                ckpt = os.path.join(args.output_dir, f"distill_step_{global_step}.safetensors")
                save_checkpoint(student_unet, ckpt, global_step, loss_val, ema_params)
                # Keep only the N most recent periodic checkpoints to save disk space
                old_ckpts = sorted(_glob.glob(os.path.join(args.output_dir, "distill_step_*.safetensors")))
                while len(old_ckpts) > args.keep_checkpoints:
                    old = old_ckpts.pop(0)
                    os.remove(old)
                    old_cfg = old.replace('.safetensors', '.config.json')
                    if os.path.exists(old_cfg): os.remove(old_cfg)
                    log.info(f"  Removed old checkpoint: {os.path.basename(old)}")

            # ── Periodic memory cleanup ───────────────────────────────────────
            if global_step % args.gc_every == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # ── End of phase ──────────────────────────────────────────────────────
        log.info(f"{phase_name} done: {phase_steps} steps in {(time.time()-phase_t)/3600:.2f}h")
        save_checkpoint(
            student_unet,
            os.path.join(args.output_dir, f"distill_{phase_name}_done.safetensors"),
            global_step, loss_val, ema_params,
        )
        del dataset, dataloader, data_iter
        gc.collect(); torch.cuda.empty_cache()

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "distill_final.safetensors")
    avg_loss   = running_loss / max(1, global_step)
    save_checkpoint(student_unet, final_path, global_step, avg_loss, ema_params)

    # Restore original attention processors (clean up CaptureAttnProcessor hooks)
    if student_registry is not None:
        student_registry.restore(student_unet)
    if teacher_registry is not None and teacher_unet is not None:
        teacher_registry.restore(teacher_unet)
    if student_feature_registry is not None:
        student_feature_registry.remove()
    if teacher_feature_registry is not None:
        teacher_feature_registry.remove()

    log.info(f"\nDone: {global_step} steps, {(time.time()-t_start)/3600:.2f}h, avg_loss={avg_loss:.6f}")
    log.info(f"Final model (EMA): {final_path}")


if __name__ == "__main__":
    main()
