#!/usr/bin/env python3
"""
LD-Pruner Sensitivity Analysis for SD-Turbo UNet pruning ratio arrangement.

Pipeline:
  1. For each calibration prompt, run the teacher (unpruned) UNet through a full SD-Turbo 4-step inference (timesteps fixed to [999, 749, 499, 249]). 
    At every step, we capture the pre-step latent as the start. And compare the post-step latent between student and teacher. 
  2. For each (block, component, ratio) candidate, install magnitude-based channel masks on the student UNet, run the student at the same inputs,
     call scheduler.step to obtain the student's post-step latent, then remove the masks.
  3. LD score is computed per-timestep (to avoid high-noise timesteps dominating the std term) and summed across the 4 timesteps:
         avg_dist_t = ||mean_teacher_t - mean_student_t||
         std_dist_t = ||std_teacher_t  - std_student_t||
         ld_t       = avg_dist_t + std_dist_t
         ld_score   = sum_t(ld_t)

Channel selection for masking: weight magnitude (LD-Pruner paper formula).
"""

import os
import sys
import json
import gc
import glob
import argparse
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_HF_CACHE = ".hf_cache"
DEFAULT_TMP = ".tmp"
os.environ.setdefault('HF_HOME', DEFAULT_HF_CACHE)
os.environ.setdefault('TRANSFORMERS_CACHE', DEFAULT_HF_CACHE)
os.environ.setdefault('TMPDIR', DEFAULT_TMP)
from diffusers import StableDiffusionPipeline
from prune import sp_core

# Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = os.environ.get("SA_BASE_MODEL", "stabilityai/sd-turbo")
# Config
NUM_PROMPTS = int(os.environ.get("SA_NUM_PROMPTS", 64))
DATASET_DIR = os.environ.get("SA_DATASET_DIR", "dataset")   # read prompts from here if available
FIXED_TIMESTEPS = [999, 749, 499, 249]
PRUNING_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
# DIR
OUTPUT_DIR = "gen_test_output"
REPORT_PATH = "prune/sensitivity_ld_report.json"


# ── Block / Component definitions ────────────────────────────────────
BLOCKS = [
    "down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
    "mid_block",
    "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3",
]
COMPONENT_TYPES = ["resnet", "ffn", "self_attn", "cross_attn"]
BLOCK_SHORT = {
    'down_blocks.0': 'D0', 'down_blocks.1': 'D1',
    'down_blocks.2': 'D2', 'down_blocks.3': 'D3',
    'mid_block': 'Mid',
    'up_blocks.0': 'U0', 'up_blocks.1': 'U1',
    'up_blocks.2': 'U2', 'up_blocks.3': 'U3',
}
BLOCK_COLORS = {
    'down_blocks.0': '#1f77b4', 'down_blocks.1': '#ff7f0e',
    'down_blocks.2': '#2ca02c', 'down_blocks.3': '#d62728',
    'mid_block':     '#9467bd',
    'up_blocks.0':   '#8c564b', 'up_blocks.1':   '#e377c2',
    'up_blocks.2':   '#7f7f7f', 'up_blocks.3':   '#bcbd22',
}


# ── Module classification ────────────────────────────────────────────
"""
Match block and module by name, and match module and block
"""
def classify_module(name: str) -> str:
    parts = name.lower()
    if "ff.net" in parts or "ff." in parts:
        return "ffn"
    if "attn2" in parts:
        return "cross_attn"
    if "attn1" in parts:
        return "self_attn"
    if "resnets" in parts:
        return "resnet"
    return "other"

def get_block_name(name: str) -> str:
    for block in BLOCKS:
        if name.startswith(block):
            return block
    return "other"

def collect_block_components(unet):
    """Returns dict[block_name][comp_type] -> list[(name, module)]"""
    components = defaultdict(lambda: defaultdict(list))
    for name, module in unet.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        block = get_block_name(name)
        if block == "other":
            continue
        comp = classify_module(name)
        if comp == "other":
            continue
        components[block][comp].append((name, module))
    return components


# ── Calibration data ─────────────────────────────────────────────────
CALIBRATION_PROMPTS = [
    "a photo of a cat sitting on a windowsill",
    "a portrait of a woman with flowers in her hair",
    "a child playing with a puppy in a green park",
    "a golden retriever playing in autumn leaves",
    "a beautiful landscape with mountains and lake",
    "a cyberpunk city skyline at night, neon lights",
    "a red sports car parked on a rainy street",
    "an astronaut riding a horse on the moon",
    "a cozy coffee shop interior, morning light",
    "a dragon flying over a medieval fantasy city",
    "a watercolor painting of a rainy city street",
    "a bright red umbrella against a gray cityscape",
    "a chef preparing sushi in a traditional kitchen",
    "a majestic eagle soaring over mountain peaks",
    "a Japanese garden with cherry blossoms",
    "a medieval castle on a hilltop at sunset",
    "a sailboat on calm waters at golden hour",
    "an enchanted forest with glowing mushrooms",
    "a minimalist black and white photograph",
    "a white horse galloping through a meadow",
    "a dramatic sunset over the ocean with orange clouds",
    "a modern glass skyscraper reflecting clouds",
    "a bouquet of sunflowers in a ceramic vase",
    "a space station orbiting a distant planet",
    "an oil painting of a stormy seascape",
    "a panda eating bamboo in a misty forest",
    "rolling green hills with scattered wildflowers",
    "a bustling street market in southeast asia",
    "a vintage steam locomotive crossing a bridge",
    "a steampunk airship floating through clouds",
    "an abstract painting with geometric shapes",
    "a purple aurora borealis over a frozen lake",
]


def _scheduler_step_at(scheduler, step_idx, pred, t, latents):
    """
    Call scheduler.step with an explicit step_index.
    Force Euler-family scheduler using the specific step_idx and process all the samples in this timestep, then move on. 
    (all t=999 samples, then all t=749, etc.)
    """
    scheduler._step_index = step_idx
    return scheduler.step(pred, t, latents).prev_sample


def generate_calibration_data(pipe, device, dtype):
    """Run teacher 4-step inference per prompt and collect per-timestep samples.

    Each returned sample carries:
      - noisy_latents:       scaled UNet input at step i
      - latents_in:          pre-step latent (fed to scheduler.step)
      - timesteps:           (1,) tensor with the fixed timestep
      - step_index:          integer 0..3 for resetting scheduler state later
      - encoder_hidden_states
      - teacher_next_latent: scheduler.step output at step i (LD target)
    """
    scheduler = pipe.scheduler
    unet = pipe.unet
    data = []

    # Load prompts: prefer dataset/ .txt files (diverse, real-world captions),
    # fall back to the hardcoded CALIBRATION_PROMPTS list.
    dataset_prompts = []
    if os.path.isdir(DATASET_DIR):
        txt_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.txt")))
        if txt_files:
            import random as _random
            _random.seed(42)
            _random.shuffle(txt_files)
            for tf in txt_files:
                try:
                    p = open(tf).read().strip()
                    if p:
                        dataset_prompts.append(p)
                except Exception:
                    pass

    if len(dataset_prompts) >= NUM_PROMPTS:
        prompts = dataset_prompts[:NUM_PROMPTS]
        print(f"  Prompts: {NUM_PROMPTS} loaded from {DATASET_DIR}/")
    elif dataset_prompts:
        # supplement with hardcoded list
        combined = dataset_prompts + CALIBRATION_PROMPTS
        seen, unique = set(), []
        for p in combined:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        prompts = unique[:NUM_PROMPTS]
        print(f"  Prompts: {len(dataset_prompts)} from dataset + {len(prompts)-len(dataset_prompts)} hardcoded = {len(prompts)}")
    else:
        prompts = CALIBRATION_PROMPTS[:NUM_PROMPTS]
        print(f"  Prompts: {len(prompts)} hardcoded (no dataset found at {DATASET_DIR})")

    print(f"  Generating {len(prompts)} prompts x 4 timesteps = {len(prompts) * 4} samples")
    print(f"  Timesteps fixed to {FIXED_TIMESTEPS}")

    for prompt in tqdm(prompts, desc="  Calibration"):
        with torch.no_grad():
            text_input = pipe.tokenizer(
                [prompt], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            ).to(device)
            enc_hs = pipe.text_encoder(text_input.input_ids)[0].to(dtype=dtype)

            # Run full 4-step teacher inference. We need the actual semantic
            # denoising trajectory — NOT randn+add_noise — so that channels
            # responsible for high-frequency detail (mid_block, down_blocks.2)
            # are probed on inputs that have real content for them to reconstruct.
            scheduler.set_timesteps(4, device=device)
            assert list(scheduler.timesteps.long().tolist()) == FIXED_TIMESTEPS, (
                f"SD-Turbo schedule mismatch: got {scheduler.timesteps.tolist()}, "
                f"expected {FIXED_TIMESTEPS}"
            )

            latents = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
            latents = latents * scheduler.init_noise_sigma

            for step_idx, t in enumerate(scheduler.timesteps):
                # latents_in is the pre-step latent. We keep it so the student
                # can replay scheduler.step from the exact same starting point;
                # any divergence in the post-step latent is then attributable
                # solely to the channel mask, not to a different trajectory.
                latents_in = latents.clone()

                # scale_model_input applies the 1/sqrt(sigma^2+1) normalization
                # required by Euler schedulers before the UNet forward pass.
                scaled = scheduler.scale_model_input(latents, t).to(dtype=dtype)
                teacher_pred = unet(
                    scaled, t, encoder_hidden_states=enc_hs
                ).sample

                # Teacher traverses the schedule sequentially, so step_index
                # matches the loop index. scheduler.step returns the latent
                # *after* timestep t — this is exactly the "latent after target
                # timestep" used as the LD comparison target.
                scheduler._step_index = step_idx
                teacher_next_latent = scheduler.step(
                    teacher_pred, t, latents_in
                ).prev_sample

                data.append({
                    'noisy_latents': scaled.detach(),
                    'latents_in': latents_in.detach(),
                    'timesteps': t.unsqueeze(0),
                    'step_index': step_idx,
                    'encoder_hidden_states': enc_hs.detach(),
                    'teacher_pred': teacher_pred.detach(),
                    'teacher_next_latent': teacher_next_latent.detach().float(),
                })

                latents = teacher_next_latent

    return data


# ── Channel masking ──────────────────────────────────────────────────
#
# Channel selection criterion (controls which channels get masked during the
# sensitivity sweep):
#
#   "magnitude" (LD-Pruner paper, original default):
#       imp = |w|.mean(non_channel_dims)
#       Pros: reproducible w/ paper; no extra forward/backward; no data needed.
#       Cons: criterion ≠ actual pruning criterion (Taylor) → sensitivity
#             numbers describe a different counterfactual than the real prune.
#
#   "taylor" (matches the actual pruning criterion in sp_core.py):
#       imp = |w · ∂L/∂w|.mean(non_channel_dims), L = MSE(student, teacher_pred)
#       Pros: sensitivity is measured under the exact channel-selection rule
#             that the downstream pipeline applies, so LD numbers reflect the
#             true cost of pruning.
#       Cons: needs calib_data with `teacher_pred`; one backward per sample.
#
# Use `--channel-selection taylor` for methodologically-aligned profiling
# (recommended). Use `--channel-selection magnitude` to reproduce LD-Pruner
# paper baselines exactly.

def compute_taylor_scores_for_components(unet, components, calib_data, device):
    """Compute per-channel Taylor scores |w · g|.mean(non_channel_dims) for
    every Linear/Conv2d module reachable through `components`.

    Loss = F.mse_loss(student_pred, teacher_pred), where teacher_pred is the
    raw UNet noise prediction stored in calib_data by generate_calibration_data().
    This matches the loss used in sp_core.py's Taylor pruning, so the
    importance signal here is the SAME signal that drives real channel removal.

    Returns:
        dict[fqn -> 1-D tensor of length out_channels].
    """
    # Collect target modules and their original requires_grad state
    target_modules = {}
    saved_requires_grad = {}
    for block_dict in components.values():
        for mod_list in block_dict.values():
            for fqn, mod in mod_list:
                target_modules[fqn] = mod
                saved_requires_grad[fqn] = mod.weight.requires_grad
                mod.weight.requires_grad_(True)

    scores = {fqn: torch.zeros(mod.weight.shape[0], device=device)
              for fqn, mod in target_modules.items()}

    unet_was_training = unet.training
    unet.eval()  # eval mode is fine; we only need grads, not dropout

    for sample in calib_data:
        # Clear previous grads
        for mod in target_modules.values():
            if mod.weight.grad is not None:
                mod.weight.grad = None

        noisy = sample['noisy_latents'].to(device)
        t = sample['timesteps'].to(device)
        enc_hs = sample['encoder_hidden_states'].to(device)
        teacher_pred = sample['teacher_pred'].to(device)

        student_pred = unet(noisy, t, encoder_hidden_states=enc_hs).sample
        loss = F.mse_loss(student_pred.float(), teacher_pred.float())
        loss.backward()

        for fqn, mod in target_modules.items():
            g = mod.weight.grad
            if g is None:
                continue
            taylor = (mod.weight.data * g).abs()
            if taylor.dim() >= 2:
                # Collapse all non-channel dims (dim 0 = out_channels)
                taylor = taylor.mean(dim=tuple(range(1, taylor.dim())))
            scores[fqn] += taylor.detach()

    # Restore state
    for fqn, mod in target_modules.items():
        mod.weight.requires_grad_(saved_requires_grad[fqn])
        if mod.weight.grad is not None:
            mod.weight.grad = None
    if unet_was_training:
        unet.train()

    # Average over samples (units don't matter for ranking, but keeps numbers
    # invariant to calib set size — useful when comparing sensitivity reports)
    n = max(1, len(calib_data))
    for fqn in scores:
        scores[fqn] /= n
    return scores


def create_channel_mask(module, ratio, precomputed_score=None):
    """Return a per-channel mask (1=keep, 0=prune) selecting the `ratio` lowest
    importance channels. If `precomputed_score` is given, use it directly
    (Taylor mode); otherwise fall back to weight magnitude (LD-Pruner default).
    """
    if ratio <= 0:
        return None

    if precomputed_score is not None:
        # Taylor-based ranking: provided by compute_taylor_scores_for_components
        imp = precomputed_score
    else:
        # Magnitude-based (LD-Pruner paper default; retained for reproducibility)
        w = module.weight.detach()
        if w.dim() >= 2:
            imp = w.abs().mean(dim=tuple(range(1, w.dim())))
        else:
            imp = w.abs()

    n_channels = imp.shape[0]
    n_prune = int(n_channels * ratio)
    if n_prune == 0:
        return None

    _, indices = torch.topk(imp, k=n_prune, largest=False)
    mask = torch.ones(n_channels, device=imp.device)
    mask[indices] = 0.0
    return mask


def make_channel_mask_hook(mask):
    def hook(module, input, output):
        if output.dim() == 4:
            return output * mask[None, :, None, None]
        elif output.dim() == 3:
            return output * mask[None, None, :]
        elif output.dim() == 2:
            return output * mask[None, :]
        return output
    return hook


def install_masks(components, block, comp, ratio, taylor_scores=None):
    """Install forward hooks that zero the `ratio` lowest-importance channels.

    If `taylor_scores` is provided (dict[fqn -> per-channel tensor]), each
    module's mask is built from its cached Taylor score; otherwise we fall
    back to weight magnitude inside create_channel_mask.
    """
    modules_list = components.get(block, {}).get(comp, [])
    handles = []
    for fqn, module in modules_list:
        score = taylor_scores.get(fqn) if taylor_scores is not None else None
        mask = create_channel_mask(module, ratio, precomputed_score=score)
        if mask is not None:
            h = module.register_forward_hook(make_channel_mask_hook(mask))
            handles.append(h)
    return handles


# ── Sensitivity computation (LD-Pruner, per-timestep bucketed) ────────
def compute_sensitivity(unet, scheduler, calib_data, components,
                        max_samples=None, channel_selection="taylor"):
    """Compute LD scores for all (block, component, ratio) combinations.

    Args:
        channel_selection: "taylor" (recommended, matches actual prune criterion)
                           or "magnitude" (LD-Pruner paper original).

    LD is computed per-timestep to prevent the high-noise timestep from
    dominating the std term, then summed:
        avg_dist_t = ||mean_teacher_t - mean_student_t||_2
        std_dist_t = ||std_teacher_t  - std_student_t||_2
        ld_score   = sum_t (avg_dist_t + std_dist_t)
    """
    samples = calib_data[:max_samples] if max_samples else calib_data

    # Channel-selection criterion. In Taylor mode we compute per-channel
    # |w · grad| scores ONCE up front (one backward pass per calib sample) and
    # cache them; these get passed to install_masks below so every (block,
    # comp, ratio) sweep uses the SAME channel ranking as the real pruning
    # pipeline (sp_core.py's Taylor-softmask).
    taylor_scores = None
    if channel_selection == "taylor":
        print(f"  Channel selection: Taylor (|w·grad|, computed over {len(samples)} samples)")
        device = next(unet.parameters()).device
        taylor_scores = compute_taylor_scores_for_components(
            unet, components, samples, device,
        )
        print(f"  Cached Taylor scores for {len(taylor_scores)} modules")
    elif channel_selection == "magnitude":
        print(f"  Channel selection: magnitude (|w|.mean, LD-Pruner paper)")
    else:
        raise ValueError(f"channel_selection must be 'taylor' or 'magnitude', got {channel_selection!r}")

    # Pre-bucket sample indices by timestep. Bucketing is essential: the four
    # timesteps have wildly different latent magnitudes (t=999 is nearly pure
    # noise with sigma ~14, t=249 is near-clean with sigma ~0.5). If we pooled
    # all 128 samples into a single mean/std, the t=999 variance would drown
    # out the detail-reconstruction signal from t=249 in the std term.
    indices_by_t = defaultdict(list)
    for i, s in enumerate(samples):
        indices_by_t[int(s['timesteps'].item())].append(i)
    unique_ts = sorted(indices_by_t.keys(), reverse=True)  # [999,749,499,249]

    # Teacher stacks are constant across all (block, comp, ratio) sweeps, so
    # precompute once. Avoids N_sweeps * N_samples redundant teacher forwards.
    teacher_stacks = {
        t: torch.stack([samples[i]['teacher_next_latent'].flatten() for i in indices_by_t[t]])
        for t in unique_ts
    }

    total = sum(
        1 for b in BLOCKS for c in COMPONENT_TYPES
        if components.get(b, {}).get(c)
    ) * len(PRUNING_RATIOS)
    print(f"\n  Sweeping: {total} measurements over {len(samples)} samples ({len(unique_ts)} timesteps)")
    print(f"  Metric: LD (paper formula, per-timestep bucketed)")

    results = {}
    idx = 0
    for block in BLOCKS:
        for comp in COMPONENT_TYPES:
            if not components.get(block, {}).get(comp):
                continue

            key = (block, comp)
            results[key] = {}

            for ratio in PRUNING_RATIOS:
                idx += 1

                if ratio == 0.0:
                    results[key][0.0] = {
                        'ld_score': 0.0, 'avg_dist': 0.0, 'std_dist': 0.0,
                        'per_timestep': {str(t): 0.0 for t in unique_ts},
                    }
                    continue

                # Install masks → student forward on the exact same inputs the
                # teacher saw → scheduler.step to obtain the student's post-step
                # latent → remove masks. Because the masks are forward hooks on
                # the unet modules, the SAME unet instance acts as both teacher
                # and student depending on whether hooks are currently attached.
                student_latents = [None] * len(samples)
                mask_handles = install_masks(components, block, comp, ratio,
                                             taylor_scores=taylor_scores)

                with torch.no_grad():
                    for i, sample in enumerate(samples):
                        student_pred = unet(
                            sample['noisy_latents'], sample['timesteps'],
                            encoder_hidden_states=sample['encoder_hidden_states']
                        ).sample
                        student_next = _scheduler_step_at(
                            scheduler,
                            sample['step_index'],
                            student_pred,
                            sample['timesteps'][0],
                            sample['latents_in'],
                        )
                        student_latents[i] = student_next.detach().float().flatten()

                for h in mask_handles:
                    h.remove()  # critical: leftover hooks would corrupt later sweeps

                # Per-timestep LD: apply the LD-Pruner (avg_dist + std_dist)
                # formula independently within each timestep bucket, then sum.
                # per_timestep is kept in the report so downstream code can
                # inspect which timestep a given block/comp is most sensitive at.
                total_avg = 0.0
                total_std = 0.0
                per_t = {}
                for t in unique_ts:
                    t_idxs = indices_by_t[t]
                    s_stack = torch.stack([student_latents[i] for i in t_idxs])
                    teach = teacher_stacks[t]

                    avg_dist = torch.norm(teach.mean(dim=0) - s_stack.mean(dim=0)).item()
                    std_dist = torch.norm(teach.std(dim=0) - s_stack.std(dim=0)).item()

                    total_avg += avg_dist
                    total_std += std_dist
                    per_t[str(t)] = avg_dist + std_dist

                ld_score = total_avg + total_std
                results[key][ratio] = {
                    'ld_score': ld_score,
                    'avg_dist': total_avg,
                    'std_dist': total_std,
                    'per_timestep': per_t,
                }

                if idx % 10 == 0 or idx <= 3:
                    print(f"  [{idx}/{total}] {BLOCK_SHORT.get(block, block)}.{comp} "
                          f"r={ratio:.0%} | ld={ld_score:.2f} "
                          f"(avg={total_avg:.2f} std={total_std:.2f})")

    del teacher_stacks
    torch.cuda.empty_cache()

    return results


# ── Plotting ─────────────────────────────────────────────────────────
COMP_LINESTYLES = {
    'resnet': '-', 'ffn': '--', 'self_attn': '-.', 'cross_attn': ':',
}


def plot_sensitivity_curves(results, output_prefix=""):
    """Plot LD sensitivity curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle("Sensitivity Analysis: Latent Divergence (LD-Pruner, teacher-aligned)", fontsize=14)

    ax.set_xlabel("Pruning Ratio")
    ax.set_ylabel("LD Score (sum over 4 timesteps)")

    for block in BLOCKS:
        for comp in COMPONENT_TYPES:
            key = (block, comp)
            if key not in results:
                continue
            ratios_dict = results[key]
            ratios = sorted(ratios_dict.keys())
            scores = [ratios_dict[r]['ld_score'] for r in ratios]
            linestyle = COMP_LINESTYLES.get(comp, '-')
            label = f"{BLOCK_SHORT.get(block, block)}.{comp}"
            ax.plot(ratios, scores, 'o' + linestyle,
                    color=BLOCK_COLORS.get(block, '#333'),
                    label=label,
                    markersize=3, linewidth=1.2)

    ax.legend(fontsize=6, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{output_prefix}sensitivity_ld.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved chart: {path}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None,
                        help="Path to pruned model safetensors (if analyzing a pruned model)")
    parser.add_argument("--config_path", default=None,
                        help="Path to pruned model config JSON")
    parser.add_argument("--output", default=REPORT_PATH,
                        help="Output JSON report path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of calibration samples")
    parser.add_argument("--round", type=int, default=None,
                        help="Round number (for appending to sensitivity_record.json)")
    parser.add_argument("--label", default=None,
                        help="Label for this round in the record (e.g. 'pruned_7pct')")
    parser.add_argument("--channel-selection", default="taylor",
                        choices=["taylor", "magnitude"],
                        help="How to rank channels for masking during the sensitivity "
                             "sweep. 'taylor' (default) matches the real prune criterion "
                             "in sp_core.py — recommended. 'magnitude' reproduces the "
                             "LD-Pruner paper baseline.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("Loading pipeline...")
    model_source = sp_core.resolve_base_model_source(BASE_MODEL)
    if args.model_path:
        from prune.pruned_rebuild import create_unet_from_safetensors

        pipe = StableDiffusionPipeline.from_pretrained(
            model_source, torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False,
        ).to(DEVICE)

        config_path = args.config_path
        if not config_path:
            config_path = args.model_path.replace('.safetensors', '.config.json')
        unet = create_unet_from_safetensors(args.model_path, config_path)
        unet = unet.to(device=DEVICE, dtype=torch.float32)
        print(f"  Loaded pruned model: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M params")
        pipe.unet = unet
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_source, torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False,
        ).to(DEVICE)

    unet = pipe.unet
    unet.eval()
    dtype = torch.float32

    # Calibration data (teacher 4-step inference)
    print("\n1. Generating calibration data (teacher 4-step inference)...")
    calib_data = generate_calibration_data(pipe, DEVICE, dtype)

    # Collect components
    components = collect_block_components(unet)
    print(f"\n  Components found:")
    for block in BLOCKS:
        comps = components.get(block, {})
        parts = [f"{c}: {len(comps[c])} layers" for c in COMPONENT_TYPES if c in comps]
        if parts:
            print(f"    {BLOCK_SHORT[block]}: {', '.join(parts)}")

    # Sweep
    print("\n2. Computing LD sensitivity...")
    results = compute_sensitivity(
        unet, pipe.scheduler, calib_data, components,
        max_samples=args.max_samples,
        channel_selection=args.channel_selection,
    )

    # Plot
    print("\n3. Generating chart...")
    plot_sensitivity_curves(results)

    # Save report
    serializable = {}
    for (block, comp), ratios_dict in results.items():
        if block not in serializable:
            serializable[block] = {}
        serializable[block][comp] = {
            str(r): v for r, v in ratios_dict.items()
        }

    report = {
        "config": {
            "num_prompts": NUM_PROMPTS,
            "timesteps": FIXED_TIMESTEPS,
            "total_samples": len(calib_data),
            "pruning_ratios": PRUNING_RATIOS,
            "metric": "ld_paper",
            "channel_selection": "weight_magnitude",
        },
        "results": serializable,
    }

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {args.output}")

    # Append to sensitivity_record.json
    record_path = os.path.join(os.path.dirname(args.output) or 'prune',
                               'sensitivity_record.json')
    if args.round is not None:
        model_params = sum(p.numel() for p in unet.parameters()) / 1e6
        entry = {
            "round": args.round,
            "label": args.label or f"round_{args.round}",
            "model_params_M": round(model_params, 1),
            "metric": "ld_paper",
            "config": report["config"],
            "results": serializable,
        }
        if os.path.exists(record_path):
            with open(record_path) as f:
                record = json.load(f)
        else:
            record = {"description": "Sensitivity analysis record across pruning rounds.", "rounds": []}
        # Replace existing round entry if present
        record["rounds"] = [r for r in record["rounds"] if r.get("round") != args.round]
        record["rounds"].append(entry)
        record["rounds"].sort(key=lambda r: r["round"])
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)
        print(f"  Appended round {args.round} to: {record_path}")

    # Cleanup
    del pipe, unet
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
