#!/usr/bin/env python3
"""Q-LoRA training loop for packed SDXL-Lightning UNet.

Loads:
  - Packed UNet via build_packed_unet (PackedInt4/Int8 modules, frozen).
  - Teacher cache from qlora_cache_teacher.py.

Wraps every PackedLinear with a LoRA adapter (rank=16 by default), and trains
the LoRA parameters to minimize MSE between student noise_pred and the cached
teacher noise_pred. Gradient checkpointing on UNet keeps activation VRAM low.

After training, saves the LoRA state dict to a small .safetensors file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantization.main.packed.build_packed_unet import build_packed_unet
from quantization.main.packed.qlora_wrapper import (
    collect_lora_params, get_lora_state_dict, wrap_packed_linears_with_lora,
)


class TeacherCacheDataset:
    """In-memory index into the on-disk teacher cache; loads each sample lazily."""

    def __init__(self, cache_dir: Path, device: str = "cpu"):
        self.cache_dir = cache_dir
        self.device = device
        index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        self.samples = index["samples"]
        self.meta = {k: v for k, v in index.items() if k != "samples"}

    def __len__(self):
        return len(self.samples)

    def get(self, i: int) -> dict:
        info = self.samples[i]
        return load_file(str(self.cache_dir / info["file"]), device=self.device)


def total_param_count(params) -> int:
    return sum(p.numel() for p in params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-safetensors", type=Path, required=True)
    ap.add_argument("--packed-manifest", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Samples per optimizer step (gradient accumulation if >1).")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260518)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0,
                    help="0 = only save at end; N = checkpoint every N steps.")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="If >0, cap total optimizer steps (for quick smoke tests).")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(args.repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(args.repo_root / ".tmp"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda"
    dtype = torch.float16

    print(f"Loading packed UNet manifest...")
    manifest = json.loads(args.packed_manifest.read_text(encoding="utf-8"))
    print(f"Loading packed state...")
    packed_state = load_file(str(args.packed_safetensors), device="cpu")
    print(f"Building packed UNet (PackedInt4/Int8) on GPU...")
    unet, stats = build_packed_unet(packed_state, manifest, device, dtype)
    del packed_state
    torch.cuda.empty_cache()
    print(f"  replaced: {stats['n_w4']} W4 + {stats['n_w8']} W8 packed layers")

    print(f"Wrapping packed layers with LoRA r={args.rank}, alpha={args.alpha}...")
    n_wrapped = wrap_packed_linears_with_lora(unet, r=args.rank, alpha=args.alpha, dtype=dtype)
    print(f"  wrapped {n_wrapped} layers")
    lora_params = collect_lora_params(unet)
    print(f"  trainable LoRA params: {total_param_count(lora_params)/1e6:.2f} M "
          f"(rank={args.rank} over {n_wrapped} layers)")

    print("Enabling gradient checkpointing on UNet for low activation VRAM...")
    if hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    unet.train()

    print(f"Loading teacher cache: {args.teacher_cache}")
    ds = TeacherCacheDataset(args.teacher_cache, device="cpu")
    print(f"  {len(ds)} samples; meta={ds.meta}")

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    total_steps = (len(ds) // args.batch_size) * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    print(f"\nTraining: {args.epochs} epochs x {len(ds)} samples / batch {args.batch_size} = ~{total_steps} steps")
    print(f"Optimizer: AdamW lr={args.lr}, wd={args.weight_decay}")

    sample_order = list(range(len(ds)))
    step = 0
    losses_window = []
    t_start = time.time()
    log_history = []

    grad_accum_steps = max(1, args.batch_size)
    for epoch in range(args.epochs):
        random.shuffle(sample_order)
        for batch_start in range(0, len(sample_order), grad_accum_steps):
            optimizer.zero_grad(set_to_none=True)
            batch_idxs = sample_order[batch_start:batch_start + grad_accum_steps]
            batch_loss = 0.0
            for sub_idx in batch_idxs:
                s = ds.get(sub_idx)
                latent_in = s["latent_in"].to(device, dtype, non_blocking=True)
                timestep = s["timestep"].to(device)
                prompt_embeds = s["prompt_embeds"].to(device, dtype, non_blocking=True)
                pooled = s["pooled_embeds"].to(device, dtype, non_blocking=True)
                time_ids = s["time_ids"].to(device, dtype, non_blocking=True)
                teacher_pred = s["teacher_pred"].to(device, dtype, non_blocking=True)
                pred = unet(latent_in, timestep,
                            encoder_hidden_states=prompt_embeds,
                            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                loss = F.mse_loss(pred.float(), teacher_pred.float()) / len(batch_idxs)
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            losses_window.append(batch_loss)
            step += 1
            if step % args.log_every == 0 or step == 1:
                avg = sum(losses_window[-args.log_every:]) / min(len(losses_window), args.log_every)
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                elapsed = time.time() - t_start
                eta = (elapsed / step) * (total_steps - step) if total_steps > step else 0
                print(f"  epoch {epoch+1}/{args.epochs}  step {step}/{total_steps}  "
                      f"loss={avg:.5f}  peak_vram={peak_mb:.0f}MB  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
                log_history.append({"step": step, "epoch": epoch + 1, "loss": avg,
                                    "peak_vram_mb": peak_mb, "elapsed_s": elapsed})
            if args.save_every > 0 and step % args.save_every == 0:
                ckpt = args.output_dir / f"lora_step{step}.safetensors"
                save_file(get_lora_state_dict(unet), str(ckpt))
                print(f"    [checkpoint] {ckpt.name}")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final = args.output_dir / "lora_final.safetensors"
    save_file(get_lora_state_dict(unet), str(final))

    summary = {
        "rank": args.rank, "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "n_wrapped_layers": n_wrapped,
        "n_lora_params": total_param_count(lora_params),
        "n_samples": len(ds), "steps_done": step,
        "final_loss": sum(losses_window[-50:]) / max(1, min(len(losses_window), 50)),
        "log_history": log_history,
        "packed_safetensors": str(args.packed_safetensors),
        "packed_manifest": str(args.packed_manifest),
        "teacher_cache": str(args.teacher_cache),
    }
    (args.output_dir / "qlora_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinal loss: {summary['final_loss']:.5f}")
    print(f"Saved: {final}")
    print(f"Done. {time.time() - t_start:.0f}s elapsed.")


if __name__ == "__main__":
    main()
