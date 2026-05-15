#!/usr/bin/env python3
"""
对比所有量化方案的生成质量 + 性能。

流程：
  1. 加载 SD-Turbo base pipeline
  2. 对每个 recipe：
     - 加载量化后的 UNet（fp16 是 .safetensors，量化方案是 .pt）
     - 跑 warmup
     - 在固定 prompts + 固定 seed 下生成 N 张图
     - 测推理时间中位数 + 峰值 VRAM
  3. 输出：
     - quantize/results/comparison.png  （行=recipe，列=prompt 的对比网格图）
     - quantize/results/metrics.json    （大小/速度/显存）
     - quantize/results/<recipe>/<i>.png

用法：
  python quantize/evaluate.py
  python quantize/evaluate.py --recipes fp16,int8_weight_only,fp8_dynamic
  python quantize/evaluate.py --num-prompts 4
"""
import os
import sys
import gc
import json
import time
import glob
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "quantize_config.yaml")


def _load_config(path):
    if not os.path.exists(path):
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_base_name(cfg, args):
    """复用 quantize.py 的命名逻辑：HF repo short name + filename stem。"""
    if args.unet_weights:
        return os.path.splitext(os.path.basename(args.unet_weights))[0]
    if cfg.get("unet_weights"):
        return os.path.splitext(os.path.basename(cfg["unet_weights"]))[0]
    repo     = cfg.get("unet_repo")
    filename = cfg.get("unet_filename", "distill_final.safetensors")
    if not repo:
        raise ValueError("Need unet_repo (or unet_weights) in config to derive base_name")
    return f"{repo.split('/')[-1]}_{os.path.splitext(filename)[0]}"


def _find_saved_recipes(output_dir, base_name):
    """扫描 output_dir 下匹配 base_name 的所有 recipe 文件。"""
    recipes = {}
    for ext in ("safetensors", "pt"):
        for path in sorted(glob.glob(os.path.join(output_dir, f"{base_name}_*.{ext}"))):
            name = os.path.basename(path)
            recipe = name.replace(f"{base_name}_", "").rsplit(".", 1)[0]
            recipes[recipe] = path
    return recipes


def build_unet_for_recipe(weights_path, sidecar_path, recipe, dtype, device):
    """重建 UNet 并加载权重（fp16 / 量化方案统一入口）。

    fp16:    create_unet_from_safetensors，直接 load .safetensors
    量化方案: 先按 model_config 建空 UNet → 形状对齐 → 搬到 device → apply_recipe
              （替换 Linear 类型，INT4 kernel 要求 CUDA 上执行）→ load_state_dict
    """
    from prune.pruned_rebuild import (
        _replace_layers_to_match_shapes,
        _fix_internal_attrs,
    )
    from diffusers import UNet2DConditionModel

    with open(sidecar_path, encoding="utf-8") as f:
        meta = json.load(f)
    model_config = meta["model_config"]

    if recipe == "fp16":
        st = load_file(weights_path)
    else:
        # 量化方案保存为 .pt，state_dict 里有 AffineQuantizedTensor 等 subclass
        st = torch.load(weights_path, map_location="cpu", weights_only=False)

    unet = UNet2DConditionModel(**model_config)
    _replace_layers_to_match_shapes(unet, st)
    _fix_internal_attrs(unet)
    unet = unet.to(device=device, dtype=dtype)

    if recipe != "fp16":
        # apply_recipe 必须在最终 device 上执行（INT4 的 _convert_weight_to_int4pack
        # 只有 CUDA 实现）。在 CPU 上调用会报 NotImplementedError。
        from quantize.quantize import apply_recipe
        unet = apply_recipe(unet, recipe)

    missing, unexpected = unet.load_state_dict(st, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    return unet


def load_pipeline_with_quantized_unet(base_model, weights_path, sidecar_path,
                                       device, recipe):
    """Pipeline dtype must match the dtype used at quantization time.

    - int4_weight_only requires bf16 (TILE_PACKED_TO_4D constraint)
    - All other recipes use fp16
    """
    from diffusers import StableDiffusionPipeline
    dtype = torch.bfloat16 if recipe == "int4_weight_only" else torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    unet = build_unet_for_recipe(weights_path, sidecar_path, recipe, dtype, device)
    pipe.unet = unet
    return pipe


def measure_inference(pipe, prompts, seed, steps, guidance, resolution,
                      warmup, runs, device, compile_mode=False):
    images = []
    times  = []

    # Warmup. torch.compile 的第一次 forward 会 trace + 编译，慢 30~60 秒，
    # 必须在测速前完成。我们用 warmup 次数把编译时间消化掉。
    if compile_mode and warmup < 2:
        warmup = 2

    for w in range(warmup):
        t_warm = time.time()
        gen = torch.Generator(device=device).manual_seed(seed)
        _ = pipe(prompts[0], num_inference_steps=steps, guidance_scale=guidance,
                 height=resolution, width=resolution, generator=gen).images[0]
        if compile_mode:
            print(f"  warmup {w+1}/{warmup}: {time.time()-t_warm:.1f}s")
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


def build_comparison_grid(recipe_images, prompts, out_path, thumb_size=256):
    recipes = list(recipe_images.keys())
    n_rows = len(recipes)
    n_cols = len(prompts)

    label_w = 220
    header_h = 60
    cell = thumb_size

    W = label_w + n_cols * cell
    H = header_h + n_rows * cell
    grid = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for c, p in enumerate(prompts):
        short = p if len(p) <= 30 else p[:27] + "..."
        draw.text((label_w + c * cell + 4, 4), f"#{c}", fill="black", font=font)
        draw.text((label_w + c * cell + 4, 22), short, fill="gray", font=small_font)

    for r, recipe in enumerate(recipes):
        y = header_h + r * cell
        draw.text((4, y + cell // 2 - 8), recipe, fill="black", font=font)
        for c, img in enumerate(recipe_images[recipe]):
            thumb = img.resize((cell, cell), Image.LANCZOS)
            grid.paste(thumb, (label_w + c * cell, y))

    grid.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare quality + speed across quantization recipes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default=DEFAULT_CONFIG)
    parser.add_argument("--recipes",     default=None,
                        help="Comma-separated subset of recipes")
    parser.add_argument("--unet-weights", default=None,
                        help="Override local unet weights path (skips HF base_name derivation)")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--device",      default=None)
    parser.add_argument("--thumb-size",  type=int, default=256)
    parser.add_argument("--compile",     action="store_true",
                        help="Wrap UNet with torch.compile. First inference is slow "
                             "(compilation), subsequent ones much faster. "
                             "Critical for unlocking dynamic quantization speedup.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    base_model  = cfg.get("base_model",  "stabilityai/sd-turbo")
    output_dir  = cfg.get("output_dir",  "models")
    results_dir = cfg.get("results_dir", "quantize/results")
    steps       = cfg.get("inference_steps", 4)
    guidance    = cfg.get("guidance_scale",  0.0)
    resolution  = cfg.get("resolution",      512)
    seed        = cfg.get("seed",            42)
    warmup      = cfg.get("benchmark_warmup", 1)
    runs        = cfg.get("benchmark_runs",   5)

    prompts = cfg.get("eval_prompts", [])
    if not prompts:
        print("ERROR: no eval_prompts in config")
        sys.exit(1)
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: CPU mode — very slow and FP8/INT8 may not work.")

    base_name = _get_base_name(cfg, args)
    print(f"Base name: {base_name}")
    print(f"Output dir: {output_dir}")

    available = _find_saved_recipes(output_dir, base_name)
    print(f"Found {len(available)} saved recipes: {list(available.keys())}")

    if args.recipes:
        wanted = [r.strip() for r in args.recipes.split(",") if r.strip()]
    elif cfg.get("eval_recipes"):
        wanted = list(cfg["eval_recipes"])
    else:
        wanted = list(available.keys())

    recipes = [r for r in wanted if r in available]
    skipped = [r for r in wanted if r not in available]
    for r in skipped:
        print(f"  [skip] {r}: no saved file found  ({base_name}_{r}.*)")
    if not recipes:
        print("ERROR: no quantized variants found. Run quantize.py first.")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)

    metrics = {}
    recipe_images = {}

    for recipe in recipes:
        print(f"\n=== {recipe} ===")
        weights_path = available[recipe]
        sidecar_path = os.path.join(output_dir, f"{base_name}_{recipe}.config.json")
        size_mb = os.path.getsize(weights_path) / 1024 ** 2

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 单个 recipe 失败不应该影响其他 recipe 的测量。常见崩溃：
        #  - int8_dynamic + torch.compile: kernel 要求 batch >= 16
        #  - fp8_dynamic + torch.compile : 同上 / shape 限制
        try:
            pipe = load_pipeline_with_quantized_unet(
                base_model, weights_path, sidecar_path, device, recipe,
            )

            if args.compile:
                print("  compiling UNet (first inference will be slow) ...")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)

            images, times, peak_vram = measure_inference(
                pipe, prompts, seed, steps, guidance, resolution, warmup, runs, device,
                compile_mode=args.compile,
            )
        except Exception as e:
            print(f"  [FAIL] {recipe}: {type(e).__name__}: {e}")
            metrics[recipe] = {
                "file_size_mb": round(size_mb, 1),
                "error":        f"{type(e).__name__}: {e}",
            }
            try: del pipe
            except: pass
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        median_time = sorted(times)[len(times) // 2]
        metrics[recipe] = {
            "file_size_mb":          round(size_mb, 1),
            "median_inference_sec":  round(median_time, 3),
            "per_prompt_sec":        [round(t, 3) for t in times],
            "peak_vram_gb":          round(peak_vram, 2),
        }
        recipe_images[recipe] = images

        print(f"  size      : {size_mb:.1f} MB")
        print(f"  med time  : {median_time:.3f}s ({steps} steps)")
        print(f"  peak VRAM : {peak_vram:.2f} GB")

        sub_dir = os.path.join(results_dir, recipe)
        os.makedirs(sub_dir, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(sub_dir, f"{i:02d}.png"))

        del pipe
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Tag output files when --compile is used so they don't overwrite the non-compiled baseline.
    suffix = "_compiled" if args.compile else ""
    metrics_path = os.path.join(results_dir, f"metrics{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump({"recipes": metrics, "prompts": prompts,
                   "config": {"steps": steps, "guidance": guidance,
                              "resolution": resolution, "seed": seed,
                              "base_name": base_name, "base_model": base_model,
                              "compile": args.compile}},
                  f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    grid_path = os.path.join(results_dir, f"comparison{suffix}.png")
    build_comparison_grid(recipe_images, prompts, grid_path, thumb_size=args.thumb_size)
    print(f"Grid saved   : {grid_path}")

    print("\n──── Summary ────")
    print(f"{'recipe':<25} {'size MB':>10} {'time s':>10} {'VRAM GB':>10}")
    for r, m in metrics.items():
        if "error" in m:
            print(f"{r:<25} {m['file_size_mb']:>10}      FAIL: {m['error']}")
        else:
            print(f"{r:<25} {m['file_size_mb']:>10} {m['median_inference_sec']:>10} {m['peak_vram_gb']:>10}")


if __name__ == "__main__":
    main()
