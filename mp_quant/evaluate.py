#!/usr/bin/env python3
"""
Stage 5: End-to-end evaluation framework.

Generates images from each variant (fp16 baseline, mp_quant PTQ, mp_quant + QLoRA)
on a fixed prompt/seed set, then computes quantitative metrics:

  - **Latency**: median inference time per image (5 runs, with warmup)
  - **VRAM**: peak GPU memory during inference
  - **Size**: on-disk model size
  - **LPIPS**: perceptual similarity to fp16 baseline (lower = closer to baseline)
  - (optional) **FID / CLIP**: deferred to a separate, dataset-heavy run

Also produces a side-by-side comparison grid (rows=variants, cols=prompts) so
quality differences are visually inspectable.

Run:
  python mp_quant/evaluate.py
  python mp_quant/evaluate.py --variants fp16,mp_quant_ptq,mp_quant_qlora
"""
import os
import sys
import gc
import json
import time
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mp_quant.sensitivity import _load_config, load_unet


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "mp_quant_config.yaml")


# —— Variant loaders ──────────────────────────────────────────────────────────
# Each loader returns a Stable Diffusion pipeline with the appropriate UNet.
# Variants form a chain showing the full compression pipeline:
#   sd_turbo_original  — pristine reference, no pruning/distill/quant (860M)
#   fp16               — pruned + distilled (our baseline for mp_quant, 642M)
#   mp_quant_ptq       — above + GPTQ-applied mixed-precision (37% Linear reduction)
#   mp_quant_qlora     — above + LoRA recovery (Stage 4)

def load_sd_turbo_original(cfg, device):
    """Original stabilityai/sd-turbo — never touched by our pipeline.

    Used as the Pareto-floor reference: shows what quality we *could* have
    if we hadn't compressed anything. The gap from sd_turbo_original to fp16
    is the cost of pruning + distillation; the gap from fp16 to mp_quant_*
    is the additional cost of quantization.
    """
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.get("base_model", "stabilityai/sd-turbo"),
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    # We need to size the UNet file on disk for the report; resolve it from cache.
    from huggingface_hub import try_to_load_from_cache, hf_hub_download
    cached = try_to_load_from_cache(cfg.get("base_model", "stabilityai/sd-turbo"),
                                     "unet/diffusion_pytorch_model.safetensors")
    if cached is None or cached is False:
        cached = hf_hub_download(cfg.get("base_model", "stabilityai/sd-turbo"),
                                  "unet/diffusion_pytorch_model.safetensors")
    size_mb = os.path.getsize(cached) / 1024 ** 2 if cached else 0.0
    return pipe, size_mb


def load_fp16_baseline(cfg, device):
    """Original fp16 UNet (un-quantized) from HF or local."""
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import hf_hub_download
    repo     = cfg.get("unet_repo")
    filename = cfg.get("unet_filename", "distill_final.safetensors")
    weights  = hf_hub_download(repo, filename)
    sidecar  = hf_hub_download(repo, filename.replace(".safetensors", ".config.json"))

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.get("base_model", "stabilityai/sd-turbo"),
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.unet = load_unet(weights, sidecar, device, torch.float16)
    size_mb = os.path.getsize(weights) / 1024 ** 2
    return pipe, size_mb


def load_mp_quant_ptq(cfg, device):
    """Mixed-precision quantized UNet from Stage 3 output (no LoRA)."""
    import glob as _glob
    from diffusers import StableDiffusionPipeline
    output_dir = cfg.get("output_dir", "mp_quant/output")
    candidates = sorted(_glob.glob(os.path.join(output_dir, "*_mp_r*.safetensors")))
    if not candidates:
        raise FileNotFoundError(f"No mp_quant safetensors in {output_dir}")
    weights = candidates[-1]
    sidecar = weights.replace(".safetensors", ".config.json")

    with open(sidecar, encoding="utf-8") as f:
        meta = json.load(f)
    dt_str = meta.get("quantization", {}).get("model_dtype", "float16")
    dtype = torch.bfloat16 if dt_str == "bfloat16" else torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.get("base_model", "stabilityai/sd-turbo"),
        torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.unet = load_unet(weights, sidecar, device, dtype)
    size_mb = os.path.getsize(weights) / 1024 ** 2
    return pipe, size_mb


def load_mp_quant_qlora(cfg, device):
    """Mixed-precision quantized UNet + trained LoRA adapter (Stage 4)."""
    pipe, base_size_mb = load_mp_quant_ptq(cfg, device)

    output_dir = cfg.get("output_dir", "mp_quant/output")
    lora_path = os.path.join(output_dir, "lora_adapter.pt")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"No LoRA adapter at {lora_path}")
    lora_state = torch.load(lora_path, map_location="cpu", weights_only=False)

    # Re-attach LoRA structure to the loaded UNet, then load adapter weights.
    # The LoRA target_modules can be recovered from the Stage 3 sidecar.
    import glob as _glob
    cand = sorted(_glob.glob(os.path.join(output_dir, "*_mp_r*.config.json")))[-1]
    with open(cand, encoding="utf-8") as f:
        sidecar = json.load(f)
    assignment = sidecar["quantization"]["assignment"]
    target_fqns = [fqn for fqn, b in assignment.items() if b != "fp16"]

    from peft import LoraConfig, get_peft_model
    # Rank is encoded in the LoRA state dict shape — read from the first lora_A
    sample_key = next(k for k in lora_state if "lora_A" in k)
    rank = lora_state[sample_key].shape[0]
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2,
        target_modules=target_fqns, lora_dropout=0.0, bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)

    # Load adapter weights (need them in pipe device + dtype)
    own_state = pipe.unet.state_dict()
    for k, v in lora_state.items():
        if k in own_state:
            own_state[k].copy_(v.to(own_state[k].device, dtype=own_state[k].dtype))
    pipe.unet.eval()

    lora_size_mb = os.path.getsize(lora_path) / 1024 ** 2
    return pipe, base_size_mb + lora_size_mb


VARIANT_LOADERS = {
    "sd_turbo_original": load_sd_turbo_original,
    "fp16":              load_fp16_baseline,
    "mp_quant_ptq":      load_mp_quant_ptq,
    "mp_quant_qlora":    load_mp_quant_qlora,
}


# —— Inference + metrics ──────────────────────────────────────────────────────

def measure(pipe, prompts, seed, steps, guidance, resolution, device,
            warmup=1, runs=5):
    """Generate images for each prompt; measure time + VRAM."""
    images = []
    times  = []

    for _ in range(warmup):
        gen = torch.Generator(device=device).manual_seed(seed)
        _ = pipe(prompts[0], num_inference_steps=steps, guidance_scale=guidance,
                 height=resolution, width=resolution, generator=gen).images[0]
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for prompt in prompts:
        per_prompt_times = []
        img = None
        for _ in range(runs):
            gen = torch.Generator(device=device).manual_seed(seed)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            img = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance,
                       height=resolution, width=resolution, generator=gen).images[0]
            if device == "cuda":
                torch.cuda.synchronize()
            per_prompt_times.append(time.time() - t0)
        per_prompt_times.sort()
        times.append(per_prompt_times[len(per_prompt_times) // 2])
        images.append(img)

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0.0
    return images, times, peak_vram


def compute_lpips(images_a, images_b, device):
    """Mean LPIPS distance between two image sets. Returns None if LPIPS package missing."""
    try:
        import lpips
    except ImportError:
        return None

    fn = lpips.LPIPS(net="alex").to(device).eval()
    scores = []
    with torch.no_grad():
        for a, b in zip(images_a, images_b):
            # PIL → tensor [-1, 1], shape [1, 3, H, W]
            ta = torch.from_numpy(__import__("numpy").array(a)).permute(2, 0, 1).unsqueeze(0).float()
            tb = torch.from_numpy(__import__("numpy").array(b)).permute(2, 0, 1).unsqueeze(0).float()
            ta = (ta / 127.5 - 1.0).to(device)
            tb = (tb / 127.5 - 1.0).to(device)
            scores.append(fn(ta, tb).item())
    return sum(scores) / max(1, len(scores))


def build_grid(variant_images, prompts, out_path, thumb=256):
    """Side-by-side grid: rows=variants, cols=prompts."""
    variants = list(variant_images.keys())
    label_w = 220
    header_h = 60
    W = label_w + len(prompts) * thumb
    H = header_h + len(variants) * thumb
    grid = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default(); small = ImageFont.load_default()

    for c, p in enumerate(prompts):
        short = p if len(p) <= 30 else p[:27] + "..."
        draw.text((label_w + c * thumb + 4, 4), f"#{c}", fill="black", font=font)
        draw.text((label_w + c * thumb + 4, 22), short, fill="gray", font=small)

    for r, v in enumerate(variants):
        y = header_h + r * thumb
        draw.text((4, y + thumb // 2 - 8), v, fill="black", font=font)
        for c, img in enumerate(variant_images[v]):
            grid.paste(img.resize((thumb, thumb), Image.LANCZOS),
                       (label_w + c * thumb, y))
    grid.save(out_path)


# —— Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fp16 baseline, mp_quant PTQ, and mp_quant + QLoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default=DEFAULT_CONFIG)
    parser.add_argument("--variants",    default=None,
                        help="Comma-separated subset (default: all)")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--thumb-size",  type=int, default=256)
    parser.add_argument("--steps",       type=int, default=None,
                        help="Override inference_steps from config (e.g. 2 for SD-Turbo "
                             "2-step inference)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CPU mode is slow.")

    steps      = args.steps if args.steps else cfg.get("inference_steps", 4)
    guidance   = cfg.get("guidance_scale",  0.0)
    resolution = cfg.get("resolution",      512)
    seed       = cfg.get("seed",            42)
    runs       = cfg.get("benchmark_runs",   5)
    warmup     = cfg.get("benchmark_warmup", 1)
    results_dir = cfg.get("results_dir", "mp_quant/results")
    prompts    = cfg.get("eval_prompts", [])
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]
    if not prompts:
        print("ERROR: no eval_prompts in config")
        sys.exit(1)

    variants = (args.variants.split(",") if args.variants
                else list(VARIANT_LOADERS.keys()))
    print(f"Variants: {variants}")
    print(f"Prompts:  {len(prompts)}")

    os.makedirs(results_dir, exist_ok=True)

    metrics = {}
    images_by_variant = {}
    for v in variants:
        if v not in VARIANT_LOADERS:
            print(f"  [skip] unknown variant: {v}")
            continue
        print(f"\n=== {v} ===")
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            pipe, size_mb = VARIANT_LOADERS[v](cfg, device)
            images, times, peak_vram = measure(
                pipe, prompts, seed, steps, guidance, resolution, device,
                warmup=warmup, runs=runs,
            )
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            metrics[v] = {"error": f"{type(e).__name__}: {e}"}
            continue

        median = sorted(times)[len(times)//2]
        metrics[v] = {
            "size_mb":             round(size_mb, 1),
            "median_latency_sec":  round(median, 3),
            "per_prompt_sec":      [round(t, 3) for t in times],
            "peak_vram_gb":        round(peak_vram, 2),
        }
        images_by_variant[v] = images
        print(f"  size:    {size_mb:.1f} MB")
        print(f"  latency: {median:.3f}s (median of {runs}, {steps} steps)")
        print(f"  VRAM:    {peak_vram:.2f} GB")

        # Save per-variant images for visual inspection. Tag the subfolder with
        # step count so multi-step-count runs don't overwrite each other.
        sub = os.path.join(results_dir, f"{v}_{steps}step")
        os.makedirs(sub, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(sub, f"{i:02d}.png"))

        del pipe; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # LPIPS computation — compute vs two reference points when available:
    #   - vs sd_turbo_original : "total compression cost" (cumulative pipeline)
    #   - vs fp16              : "mp_quant added cost" (this project's contribution)
    references = [r for r in ("sd_turbo_original", "fp16") if r in images_by_variant]
    if references:
        for v in images_by_variant:
            for ref in references:
                if v == ref:
                    metrics[v][f"lpips_vs_{ref}"] = 0.0
                    continue
                score = compute_lpips(images_by_variant[v], images_by_variant[ref], device)
                if score is not None:
                    metrics[v][f"lpips_vs_{ref}"] = round(score, 4)
                    print(f"LPIPS({v} vs {ref}) = {score:.4f}")
                else:
                    print(f"LPIPS unavailable (pip install lpips). Skipping.")
                    break  # don't retry if not installed

    # Save metrics + grid (step-tagged so multi-step runs coexist)
    metrics_name = f"eval_metrics_{steps}step.json"
    grid_name    = f"eval_grid_{steps}step.png"
    with open(os.path.join(results_dir, metrics_name), "w") as f:
        json.dump({"variants": metrics, "config": {
            "steps": steps, "guidance": guidance, "resolution": resolution,
            "seed": seed, "prompts": prompts,
        }}, f, indent=2)
    if images_by_variant:
        build_grid(images_by_variant, prompts,
                   os.path.join(results_dir, grid_name),
                   thumb=args.thumb_size)

    print(f"\n──── Summary ────")
    print(f"{'variant':<20} {'size MB':>10} {'latency s':>10} {'VRAM GB':>10} "
          f"{'vs orig':>9} {'vs fp16':>9}")
    for v, m in metrics.items():
        if "error" in m:
            print(f"{v:<20}      FAIL: {m['error']}")
        else:
            lp_orig = m.get("lpips_vs_sd_turbo_original", "-")
            lp_fp16 = m.get("lpips_vs_fp16", "-")
            print(f"{v:<20} {m['size_mb']:>10} {m['median_latency_sec']:>10} "
                  f"{m['peak_vram_gb']:>10} {lp_orig!s:>9} {lp_fp16!s:>9}")


if __name__ == "__main__":
    main()
