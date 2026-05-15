#!/usr/bin/env python3
"""
Stage 4: QLoRA Recovery Training.

After Stage 3, the model has its weights rounded to the quant grid (fake-quant).
This stage recovers quality lost in PTQ by training a LoRA adapter on top of
the frozen quantized base, using a step-wise teacher-student distillation
loss against the original fp16 model.

Design choices (mirrors prune/sp_distill.py for consistency):
  - Loss: MSE + λ·L1 in PREDICTION SPACE (compare s_pred vs t_pred at each
    denoising step, no scheduler.step). Same as the pruning recovery stage.
  - Frozen quantized base + trainable LoRA adapters on every quantized Linear.
  - Teacher: original fp16 UNet (no quantization, no LoRA).
  - Teacher trajectory drives both teacher and student inputs at every step,
    so errors never compound across steps (per-step independent).

Why LoRA (instead of full QAT on scales / zero-points):
  - Stronger expressiveness — low-rank correction can fix more than a scale shift.
  - Mature tooling (peft).
  - LoRA adapters can be merged into the base at inference time → zero overhead.

Output:
  - lora_adapter.pt           the trained LoRA weights (small file)
  - distill_log.txt           training log
  - (optional) merged.safetensors  base weights with LoRA folded in

Run:
  python mp_quant/qlora_recovery.py
  python mp_quant/qlora_recovery.py --steps 500     # quick test
"""
import os
import sys
import gc
import json
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mp_quant.sensitivity import (
    _load_config,
    _resolve_unet_source,
    load_unet,
)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "mp_quant_config.yaml")


# —— Dataset (prompt-only, mirrors prune/sp_distill.py) ───────────────────────

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_samples=5000, seed=42):
        import glob as _glob
        import random as _rand
        paths = sorted(_glob.glob(os.path.join(root, "*.txt")))
        if not paths:
            raise ValueError(f"No prompts in {root}")
        if len(paths) > max_samples:
            _rand.seed(seed)
            _rand.shuffle(paths)
            paths = paths[:max_samples]
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], encoding="utf-8") as f:
            return f.read().strip()


# —— Load the quantized base UNet from Stage 3 output ─────────────────────────

def load_quantized_base(weights_path, config_path, device):
    """Load the mp_quant output (fake-quant safetensors) and its dtype.

    The dtype is stored in the sidecar JSON under quantization.model_dtype.
    """
    with open(config_path, encoding="utf-8") as f:
        meta = json.load(f)
    dt_str = meta.get("quantization", {}).get("model_dtype", "float16")
    dtype = torch.bfloat16 if dt_str == "bfloat16" else torch.float16

    unet = load_unet(weights_path, config_path, device, dtype)
    return unet, dtype, meta


# —— Build cosine LR schedule (matches sp_distill.py) ─────────────────────────

def build_cosine_lr(warmup, total, lr_max, lr_min):
    min_ratio = lr_min / lr_max
    def lr_lambda(step):
        if step < warmup:
            return min_ratio + (1.0 - min_ratio) * (step / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return lr_lambda


# —— Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QLoRA recovery training for the mp-quantized UNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",           default=DEFAULT_CONFIG)
    parser.add_argument("--quantized-weights", default=None,
                        help="Path to mp_quant safetensors from Stage 3 "
                             "(default: latest in output_dir)")
    parser.add_argument("--quantized-config",  default=None,
                        help="Sidecar JSON path (default: matches weights file)")
    parser.add_argument("--teacher-weights",   default=None,
                        help="Original fp16 UNet weights (default: HF unet_repo/unet_filename from config)")
    parser.add_argument("--teacher-config",    default=None)
    # Training
    parser.add_argument("--steps",            type=int,   default=None,
                        help="Total training steps (default: qat_steps from config)")
    parser.add_argument("--batch-size",       type=int,   default=None)
    parser.add_argument("--grad-accum-steps", type=int,   default=None)
    parser.add_argument("--lr-max",           type=float, default=None)
    parser.add_argument("--lr-min",           type=float, default=None)
    parser.add_argument("--warmup",           type=int,   default=None)
    parser.add_argument("--lambda-l1",        type=float, default=0.1)
    # LoRA
    parser.add_argument("--lora-rank",        type=int,   default=8)
    parser.add_argument("--lora-alpha",       type=int,   default=None,
                        help="LoRA alpha (default: 2*rank, standard ratio)")
    parser.add_argument("--lora-dropout",     type=float, default=0.0)
    # I/O
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument("--save-every",       type=int,   default=500)
    parser.add_argument("--log-every",        type=int,   default=20)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: GPU required for QLoRA training")
        sys.exit(1)

    # Resolve hyper-params (CLI > config > defaults)
    steps        = args.steps             or cfg.get("qat_steps",        2000)
    bs           = args.batch_size        or cfg.get("qat_batch_size",   1)
    grad_accum   = args.grad_accum_steps  or cfg.get("qat_grad_accum",   4)
    lr_max       = args.lr_max            or cfg.get("qat_lr_max",       1e-4)
    lr_min       = args.lr_min            or cfg.get("qat_lr_min",       1e-6)
    warmup       = args.warmup            or cfg.get("qat_warmup_steps", 100)
    output_dir   = args.output_dir        or cfg.get("output_dir",       "mp_quant/output")
    n_steps_inf  = cfg.get("inference_steps", 4)
    resolution   = cfg.get("resolution",      512)
    lora_alpha   = args.lora_alpha or args.lora_rank * 2
    lambda_l1    = args.lambda_l1

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "qlora_log.txt")
    log_f = open(log_path, "w", encoding="utf-8")
    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Steps={steps} bs={bs} grad_accum={grad_accum} eff_bs={bs*grad_accum}")
    log(f"LR {lr_max} -> {lr_min}, warmup={warmup}")
    log(f"LoRA: rank={args.lora_rank} alpha={lora_alpha} dropout={args.lora_dropout}")
    log(f"Loss: MSE + {lambda_l1}*L1, prediction-space")

    # —— Resolve & load quantized student ─────────────────────────────────────
    if args.quantized_weights is None:
        import glob as _glob
        candidates = sorted(_glob.glob(os.path.join(output_dir, "*_mp_r*.safetensors")))
        if not candidates:
            log("ERROR: no Stage 3 output found in output_dir")
            sys.exit(1)
        args.quantized_weights = candidates[-1]
        args.quantized_config  = args.quantized_weights.replace(".safetensors", ".config.json")
    log(f"\nQuantized student: {args.quantized_weights}")

    student, dtype, q_meta = load_quantized_base(
        args.quantized_weights, args.quantized_config, device,
    )
    log(f"  dtype={dtype}  params={sum(p.numel() for p in student.parameters())/1e6:.1f}M")

    # —— Load fp16 teacher ────────────────────────────────────────────────────
    log(f"\nTeacher (fp16, frozen):")
    # Use the original unquantized weights from HF if not specified
    if args.teacher_weights is None:
        from huggingface_hub import hf_hub_download
        repo     = cfg.get("unet_repo")
        filename = cfg.get("unet_filename", "distill_final.safetensors")
        args.teacher_weights = hf_hub_download(repo, filename)
        args.teacher_config  = hf_hub_download(repo, filename.replace(".safetensors", ".config.json"))
    teacher = load_unet(args.teacher_weights, args.teacher_config, device, dtype)
    teacher.eval()
    teacher.requires_grad_(False)
    log(f"  source: {args.teacher_weights}")

    # —— Load base pipeline (scheduler, text_encoder, tokenizer) ─────────────
    log(f"\nLoading base pipeline (scheduler + text_encoder)...")
    import copy
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.get("base_model", "stabilityai/sd-turbo"),
        torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    teacher_scheduler = copy.deepcopy(pipe.scheduler)

    # —— Attach LoRA to quantized layers ──────────────────────────────────────
    assignment = q_meta["quantization"]["assignment"]
    target_fqns = [fqn for fqn, bit in assignment.items() if bit != "fp16"]
    log(f"\nAttaching LoRA to {len(target_fqns)} quantized layers...")

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_fqns,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    student = get_peft_model(student, lora_cfg)
    student.train()
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    log(f"  Trainable: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M  "
        f"({100*n_train/n_total:.3f}%)")

    # Force LoRA params to fp32 for stable training (peft default may be dtype-mixed)
    for p in student.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # —— Optimizer + LR schedule ──────────────────────────────────────────────
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr_max, weight_decay=0.01)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_cosine_lr(warmup, steps, lr_max, lr_min)
    )

    # —— Dataset ──────────────────────────────────────────────────────────────
    log(f"\nLoading prompts dataset...")
    dataset = PromptDataset(
        cfg.get("calib_dataset", "dataset"),
        max_samples=cfg.get("calib_max_prompts", 5000),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True,
                                         num_workers=0, pin_memory=False, drop_last=True)
    data_iter = iter(loader)
    log(f"  {len(dataset)} prompts")

    # —— Training loop ────────────────────────────────────────────────────────
    log(f"\nTraining {steps} optimizer steps...")
    t_start = time.time()
    global_step = 0
    accum_step  = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while global_step < steps:
        step_t = time.time()
        try:
            prompts = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            prompts = next(data_iter)

        # Text encoding
        with torch.no_grad():
            text_ids = pipe.tokenizer(list(prompts), padding="max_length",
                                       max_length=77, truncation=True,
                                       return_tensors="pt").to(device)
            enc_hs = pipe.text_encoder(text_ids.input_ids)[0].to(dtype=dtype)

        # Initial noise
        latents = torch.randn(
            (len(prompts), 4, resolution // 8, resolution // 8),
            device=device, dtype=dtype,
        ) * pipe.scheduler.init_noise_sigma

        pipe.scheduler.set_timesteps(n_steps_inf, device=device)
        teacher_scheduler.set_timesteps(n_steps_inf, device=device)
        ts = pipe.scheduler.timesteps

        t_latent = latents.clone()
        total_loss = torch.zeros(1, device=device, dtype=torch.float32)

        for step_i, t in enumerate(ts):
            t_latent_in = t_latent.detach().clone()

            # Teacher forward (no grad, fp16/bf16)
            with torch.no_grad():
                teacher_scheduler._step_index = step_i
                t_in = teacher_scheduler.scale_model_input(t_latent_in, t)
                t_pred = teacher(t_in, t, encoder_hidden_states=enc_hs).sample
                # Advance teacher trajectory for the NEXT step's input
                teacher_scheduler._step_index = step_i
                t_latent = teacher_scheduler.step(t_pred, t, t_latent_in).prev_sample.to(dtype)

            # Student forward (LoRA gradient flows)
            pipe.scheduler._step_index = step_i
            s_in = pipe.scheduler.scale_model_input(t_latent_in, t)
            s_pred = student(s_in, t, encoder_hidden_states=enc_hs).sample

            # Prediction-space loss (avoids EulerAncestral stochastic noise)
            s_f = s_pred.float()
            t_f = t_pred.float()
            step_loss = F.mse_loss(s_f, t_f) + lambda_l1 * F.l1_loss(s_f, t_f)
            total_loss = total_loss + step_loss

        loss = (total_loss / len(ts)) / grad_accum

        if not torch.isfinite(loss):
            log(f"  NaN/Inf at substep {global_step*grad_accum + accum_step + 1}, skipping")
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0
            continue

        loss.backward()
        accum_step += 1
        if accum_step < grad_accum:
            continue
        accum_step = 0

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        loss_val = loss.item() * grad_accum
        running_loss += loss_val

        if global_step % args.log_every == 0 or global_step <= 3:
            elapsed = time.time() - t_start
            it_t = time.time() - step_t
            eta = it_t * (steps - global_step)
            cur_lr = optimizer.param_groups[0]['lr']
            log(f"  [{global_step:>4}/{steps}] loss={loss_val:.5f} "
                f"lr={cur_lr:.2e} {it_t:.1f}s/it elapsed={elapsed/60:.1f}min "
                f"ETA={eta/60:.1f}min")

        if global_step % args.save_every == 0:
            ckpt = os.path.join(output_dir, f"lora_step_{global_step}.pt")
            torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()
                        if "lora_" in k}, ckpt)
            log(f"  Saved LoRA checkpoint: {ckpt}")

    # —— Final save ───────────────────────────────────────────────────────────
    log(f"\nTraining done in {(time.time()-t_start)/60:.1f}min")
    log(f"Average loss: {running_loss/max(1, global_step):.5f}")

    # Save LoRA adapter (small file)
    final_lora = os.path.join(output_dir, "lora_adapter.pt")
    torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()
                if "lora_" in k}, final_lora)
    sz_mb = os.path.getsize(final_lora) / 1024 ** 2
    log(f"\nLoRA adapter: {final_lora}  ({sz_mb:.1f} MB)")

    log_f.close()


if __name__ == "__main__":
    main()
