#!/usr/bin/env python3
"""Run SDXL-Lightning FP16 vs fake-quant on diverse dataset prompts."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
SKIP_PATTERNS = ("time_embedding", "time_emb_proj", "add_embedding", "conv_in", "conv_out")

CATEGORIES = {
    "people": ("person", "people", "man", "woman", "boy", "girl", "child", "couple", "family"),
    "animals": ("dog", "cat", "horse", "cow", "sheep", "zebra", "giraffe", "bear", "elephant", "bird"),
    "food": ("pizza", "sandwich", "cake", "donut", "banana", "apple", "orange", "broccoli", "food", "meal", "kitchen"),
    "vehicle": ("train", "bus", "truck", "car", "motorcycle", "airplane", "boat", "bicycle"),
    "sports": ("tennis", "baseball", "skateboard", "snowboard", "ski", "surf", "frisbee", "kite"),
    "indoor": ("room", "bed", "bathroom", "toilet", "sink", "table", "chair", "couch", "laptop", "desk"),
    "street": ("street", "traffic", "sign", "building", "city", "sidewalk", "clock", "parking"),
    "nature": ("beach", "river", "ocean", "lake", "mountain", "snow", "forest", "grass", "sky"),
    "objects": ("phone", "book", "umbrella", "suitcase", "teddy", "vase", "clock", "scissors", "keyboard"),
}

BAD_SUBSTRINGS = (
    "nude",
    "porn",
    "sex",
    "shirtless",
    "looking down",
    "between her legs",
)


@dataclass
class PromptItem:
    index: int
    source: str
    category: str
    seed: int
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--linear-bits", type=int, default=4, choices=(4, 8))
    parser.add_argument(
        "--linear-block-bits",
        type=str,
        default="",
        help="Comma-separated block=bits overrides for Linear layers, e.g. mid_block=4,down_blocks.2=4,up_blocks.0=4",
    )
    parser.add_argument(
        "--linear-name-bits",
        type=str,
        default="",
        help="Comma-separated name-substring=bits overrides, applied with HIGHEST priority. "
             "Example: attn2.to_k=8,attn2.to_v=8 forces cross-attn k/v to W8 even inside a W4 block. "
             "bits=16 keeps the layer in FP16.",
    )
    parser.add_argument("--conv-bits", type=str, default="8")
    parser.add_argument(
        "--conv-include",
        type=str,
        default="",
        help="Comma-separated Conv2d name prefixes to quantize, e.g. down_blocks.1,down_blocks.2,up_blocks.0",
    )
    parser.add_argument(
        "--conv-exclude",
        type=str,
        default="",
        help="Comma-separated Conv2d name prefixes to keep fp16 even if included.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--target", type=str, default="all", choices=("all", "attention", "ff", "linear", "conv"))
    parser.add_argument("--dtype", type=str, default="float16", choices=("float16", "float32"))
    parser.add_argument("--selection-seed", type=int, default=20260517)
    parser.add_argument("--save-unet", action="store_true")
    parser.add_argument("--method", type=str, default="rtn", choices=("rtn", "gptq"),
                        help="Quantization method: rtn (round-to-nearest, no calib) or gptq (Hessian-based PTQ).")
    parser.add_argument("--calib-prompt-count", type=int, default=16,
                        help="Number of prompts used to capture calibration samples for GPTQ.")
    parser.add_argument("--gptq-chunk-mem-gb", type=float, default=4.0,
                        help="GPU memory budget (GB) for one chunk of Hessian accumulators.")
    parser.add_argument("--gptq-damping", type=float, default=0.01,
                        help="Hessian diagonal damping for GPTQ (% of mean diag).")
    parser.add_argument("--gptq-bits", type=str, default="4",
                        help="Comma-separated bit-widths to apply GPTQ to (others use RTN). Default: 4.")
    return parser.parse_args()


def normalize_prompt(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def choose_prompts(dataset_dir: Path, count: int, selection_seed: int) -> list[PromptItem]:
    rng = random.Random(selection_seed)
    rows: list[tuple[str, str]] = []
    for path in sorted(dataset_dir.glob("*.txt")):
        text = normalize_prompt(path.read_text(encoding="utf-8", errors="ignore"))
        low = text.lower()
        if len(text) < 15 or any(bad in low for bad in BAD_SUBSTRINGS):
            continue
        rows.append((path.name, text))

    buckets: dict[str, list[tuple[str, str]]] = {name: [] for name in CATEGORIES}
    for source, text in rows:
        low = text.lower()
        for category, keys in CATEGORIES.items():
            if any(k in low for k in keys):
                buckets[category].append((source, text))
                break

    for items in buckets.values():
        rng.shuffle(items)

    chosen: list[PromptItem] = []
    seen: set[str] = set()
    while len(chosen) < count and any(buckets.values()):
        for category in CATEGORIES:
            while buckets[category]:
                source, text = buckets[category].pop()
                if source in seen:
                    continue
                seen.add(source)
                seed = 12345 + len(chosen) * 9973
                chosen.append(PromptItem(len(chosen) + 1, source, category, seed, text))
                break
            if len(chosen) >= count:
                break

    if len(chosen) < count:
        leftovers = [(s, t) for s, t in rows if s not in seen]
        rng.shuffle(leftovers)
        for source, text in leftovers[: count - len(chosen)]:
            seed = 12345 + len(chosen) * 9973
            chosen.append(PromptItem(len(chosen) + 1, source, "fallback", seed, text))

    if len(chosen) != count:
        raise RuntimeError(f"Only selected {len(chosen)} prompts from {dataset_dir}")
    return chosen


def is_skipped(name: str) -> bool:
    return any(pattern in name for pattern in SKIP_PATTERNS)


def should_quantize_linear(name: str, mod: nn.Module, target: str) -> bool:
    if not isinstance(mod, nn.Linear) or is_skipped(name):
        return False
    if target == "conv":
        return False
    if target == "attention":
        return ".attn" in name or "attentions" in name
    if target == "ff":
        return ".ff." in name or "ff.net" in name
    return target in ("all", "linear")


def split_prefixes(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def has_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def parse_block_bits(value: str) -> dict[str, int]:
    result = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --linear-block-bits item: {item}")
        prefix, bits = item.split("=", 1)
        bit_value = int(bits.strip())
        if bit_value not in (4, 8, 16):
            raise ValueError(f"Unsupported bit value in --linear-block-bits: {item}")
        result[prefix.strip()] = bit_value
    return result


def linear_bits_for_layer(
    name: str,
    default_bits: int,
    block_bits: dict[str, int],
    name_bits: dict[str, int] | None = None,
) -> int | None:
    if name_bits:
        for substr, bits in sorted(name_bits.items(), key=lambda x: len(x[0]), reverse=True):
            if substr in name:
                return None if bits == 16 else bits
    for prefix, bits in sorted(block_bits.items(), key=lambda x: len(x[0]), reverse=True):
        if name == prefix or name.startswith(prefix + "."):
            return None if bits == 16 else bits
    return default_bits


def should_quantize_conv(name: str, mod: nn.Module, target: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    if not isinstance(mod, nn.Conv2d) or is_skipped(name):
        return False
    if include and not has_prefix(name, include):
        return False
    if exclude and has_prefix(name, exclude):
        return False
    return target in ("all", "conv")


@torch.no_grad()
def fake_quant_2d_(weight_2d: torch.Tensor, bits: int, group_size: int) -> dict:
    orig_dtype = weight_2d.dtype
    w = weight_2d.detach().float()
    out_f, in_f = w.shape
    if group_size <= 0:
        group_size = in_f
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    out = torch.empty_like(w)
    total_mse = 0.0
    for start in range(0, in_f, group_size):
        end = min(start + group_size, in_f)
        block = w[:, start:end]
        scale = block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.round(block / scale).clamp(qmin, qmax)
        deq = q * scale
        out[:, start:end] = deq
        total_mse += torch.mean((block - deq) ** 2).item() * (end - start)
    weight_2d.copy_(out.to(orig_dtype))
    return {"mse": total_mse / in_f, "out_features": out_f, "in_features": in_f}


@torch.no_grad()
def fake_quant_module_weight_(mod: nn.Module, bits: int, group_size: int) -> dict:
    if isinstance(mod, nn.Linear):
        return fake_quant_2d_(mod.weight, bits, group_size)
    if isinstance(mod, nn.Conv2d):
        original_shape = mod.weight.shape
        flat = mod.weight.view(mod.weight.shape[0], -1)
        stats = fake_quant_2d_(flat, bits, group_size)
        mod.weight.copy_(flat.view(original_shape).to(mod.weight.dtype))
        stats["kernel_shape"] = list(original_shape)
        return stats
    raise TypeError(type(mod).__name__)


def load_lightning_unet(step: int, dtype: torch.dtype, device: str):
    ckpt = f"sdxl_lightning_{step}step_unet.safetensors"
    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(device, dtype)
    ckpt_path = hf_hub_download(LIGHTNING_REPO, ckpt)
    state = load_file(ckpt_path, device=device)
    missing, unexpected = unet.load_state_dict(state, strict=False)
    print(f"Loaded Lightning UNet: {ckpt}, missing={len(missing)}, unexpected={len(unexpected)}")
    return unet


def build_pipe(unet, dtype: torch.dtype, device: str):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        unet=unet,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=False)
    return pipe


@torch.no_grad()
def generate(pipe, prompt: str, seed: int, device: str, args: argparse.Namespace) -> Image.Image:
    gen = torch.Generator(device=device).manual_seed(seed)
    return pipe(
        prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        generator=gen,
    ).images[0]


@torch.no_grad()
def capture_calibration_samples(pipe, prompts: list[str], args: argparse.Namespace, device: str) -> list[dict]:
    """Hook the UNet during a real pipeline run and snapshot all forward kwargs."""
    captured: list[dict] = []

    def hook(_mod, fwd_args, fwd_kwargs):
        sample = fwd_args[0] if len(fwd_args) > 0 else fwd_kwargs["sample"]
        timestep = fwd_args[1] if len(fwd_args) > 1 else fwd_kwargs["timestep"]
        enc_hs = fwd_kwargs.get("encoder_hidden_states", fwd_args[2] if len(fwd_args) > 2 else None)
        added_cond = fwd_kwargs.get("added_cond_kwargs", {}) or {}
        ts = timestep.detach().clone() if torch.is_tensor(timestep) else torch.tensor(timestep, device=device)
        captured.append({
            "sample": sample.detach().clone(),
            "timestep": ts,
            "enc_hs": enc_hs.detach().clone() if enc_hs is not None else None,
            "added_cond": {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in added_cond.items()},
        })

    handle = pipe.unet.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for i, p in enumerate(prompts):
            gen = torch.Generator(device=device).manual_seed(1000 + i)
            _ = pipe(p, num_inference_steps=args.steps, guidance_scale=args.guidance,
                     height=args.height, width=args.width, generator=gen, output_type="latent")
    finally:
        handle.remove()
    return captured


def chunk_layers_by_hessian_mem(names: list[str], name_to_module: dict, budget_bytes: int) -> list[list[str]]:
    items = [(n, name_to_module[n].in_features) for n in names]
    items.sort(key=lambda x: -x[1])
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name, in_f in items:
        sz = in_f * in_f * 4
        if sz > budget_bytes and cur:
            chunks.append(cur); cur = []; cur_bytes = 0
        if cur and cur_bytes + sz > budget_bytes:
            chunks.append(cur); cur = []; cur_bytes = 0
        cur.append(name); cur_bytes += sz
    if cur: chunks.append(cur)
    return chunks


def apply_gptq_to_layers(unet, w4_names: list[str], calib_samples: list[dict],
                          group_size: int, damping: float, budget_gb: float, device: str) -> list[dict]:
    from quantization.main.fakequant.gptq import HessianAccumulator, gptq_quantize_linear  # type: ignore  # repo_root already on sys.path

    name_to_module = dict(unet.named_modules())
    chunks = chunk_layers_by_hessian_mem(w4_names, name_to_module, int(budget_gb * 1e9))
    print(f"GPTQ: {len(w4_names)} W4 layers in {len(chunks)} chunks "
          f"(budget {budget_gb:.1f} GB, {len(calib_samples)} calib samples)")
    stats: list[dict] = []
    for ci, chunk in enumerate(chunks):
        accs = {}
        handles = []
        chunk_mem_gb = sum(name_to_module[n].in_features ** 2 * 4 for n in chunk) / 1e9
        print(f"  chunk {ci+1}/{len(chunks)}: {len(chunk)} layers, ~{chunk_mem_gb:.2f} GB Hessian RAM")
        for name in chunk:
            mod = name_to_module[name]
            acc = HessianAccumulator(mod.in_features, device=device)
            accs[name] = acc
            def make_hook(a):
                def hk(_m, fa, _fk):
                    a.add_batch(fa[0])
                return hk
            handles.append(mod.register_forward_pre_hook(make_hook(acc), with_kwargs=True))
        with torch.no_grad():
            for s in calib_samples:
                _ = unet(s["sample"], s["timestep"], encoder_hidden_states=s["enc_hs"],
                         added_cond_kwargs=s["added_cond"]).sample
        for h in handles: h.remove()
        for name in chunk:
            mod = name_to_module[name]
            H = accs[name].finalize()
            r = gptq_quantize_linear(mod, H, bits=4, group_size=group_size, damping=damping)
            stats.append({"name": name, "type": "Linear", "bits": 4,
                          "params": mod.weight.numel(), **r})
            del H
        del accs
        torch.cuda.empty_cache()
        print(f"    chunk {ci+1} done; layer_error mean = "
              f"{sum(s['layer_error'] for s in stats[-len(chunk):])/len(chunk):.3e}")
    return stats


def apply_fake_quant(unet, args: argparse.Namespace, conv_bits: int | None,
                      skip_linear_bits: set[int] | None = None) -> list[dict]:
    stats = []
    before_params = 0
    after_bits = 0
    conv_include = split_prefixes(args.conv_include)
    conv_exclude = split_prefixes(args.conv_exclude)
    linear_block_bits = parse_block_bits(args.linear_block_bits)
    linear_name_bits = parse_block_bits(args.linear_name_bits)
    skip_linear_bits = skip_linear_bits or set()
    print(
        f"Applying fake quant: linear_bits={args.linear_bits}, conv_bits={conv_bits}, "
        f"group_size={args.group_size}, target={args.target}, "
        f"linear_block_bits={linear_block_bits or 'none'}, "
        f"linear_name_bits={linear_name_bits or 'none'}, "
        f"conv_include={list(conv_include) or 'all'}, conv_exclude={list(conv_exclude) or 'none'}"
    )
    for name, mod in unet.named_modules():
        bit = None
        kind = None
        if should_quantize_linear(name, mod, args.target):
            bit = linear_bits_for_layer(name, args.linear_bits, linear_block_bits, linear_name_bits)
            if bit is None:
                continue
            if bit in skip_linear_bits:
                continue
            kind = "Linear"
        elif conv_bits is not None and should_quantize_conv(name, mod, args.target, conv_include, conv_exclude):
            bit = conv_bits
            kind = "Conv2d"
        else:
            continue
        n = mod.weight.numel()
        before_params += n
        after_bits += n * bit
        s = fake_quant_module_weight_(mod, bit, args.group_size)
        stats.append({"name": name, "type": kind, "bits": bit, "params": n, **s})
        if len(stats) <= 5 or len(stats) % 50 == 0:
            print(f"  [{len(stats)}] {kind} {name}: W{bit}, {n/1e6:.2f}M params, mse={s['mse']:.3e}")
    print(f"Quantized {len(stats)} modules, {before_params/1e6:.1f} weights")
    fp16_bits = before_params * 16
    print(f"Theoretical reduction on targeted weights: {1 - after_bits / max(1, fp16_bits):.1%}")
    return stats


def compute_image_metrics(pairs: list[tuple[Image.Image, Image.Image]]) -> dict:
    mses = []
    maes = []
    for a_img, b_img in pairs:
        a = np.asarray(a_img.convert("RGB")).astype(np.float32) / 255.0
        b = np.asarray(b_img.convert("RGB")).astype(np.float32) / 255.0
        mses.append(float(np.mean((a - b) ** 2)))
        maes.append(float(np.mean(np.abs(a - b))))
    return {
        "mse_each": mses,
        "mae_each": maes,
        "mse_mean": float(np.mean(mses)),
        "mae_mean": float(np.mean(maes)),
        "mse_max": float(np.max(mses)),
        "mae_max": float(np.max(maes)),
    }


def save_contact_sheet(pairs: list[tuple[Image.Image, Image.Image]], prompts: list[PromptItem], out_path: Path) -> None:
    thumb_w = 256
    label_h = 42
    row_gap = 8
    thumb_h = int(pairs[0][0].height * thumb_w / pairs[0][0].width)
    sheet = Image.new("RGB", (2 * thumb_w, len(pairs) * (thumb_h + label_h + row_gap)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for idx, ((base_img, q_img), item) in enumerate(zip(pairs, prompts)):
        y = idx * (thumb_h + label_h + row_gap)
        title = f"{item.index:02d} {item.category} seed={item.seed}"
        draw.text((6, y + 5), f"fp16 | {title}", fill=(0, 0, 0), font=font)
        draw.text((thumb_w + 6, y + 5), f"fake quant | {title}", fill=(0, 0, 0), font=font)
        draw.text((6, y + 22), item.source, fill=(80, 80, 80), font=font)
        sheet.paste(base_img.resize((thumb_w, thumb_h)), (0, y + label_h))
        sheet.paste(q_img.resize((thumb_w, thumb_h)), (thumb_w, y + label_h))
    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("HF_HOME", str(repo_root / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TMPDIR", str(repo_root / ".tmp"))

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    conv_bits = args.conv_bits.lower()
    conv_bits_value = None if conv_bits in ("none", "fp16", "off", "") else int(conv_bits)
    if conv_bits_value not in (None, 4, 8):
        raise ValueError("--conv-bits must be 4, 8, or none")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" and device == "cuda" else torch.float32
    prompts = choose_prompts(repo_root / "dataset", args.prompt_count, args.selection_seed)

    with (out_dir / "selected_prompts.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(item) for item in prompts], f, indent=2, ensure_ascii=False)

    print(
        f"SDXL-Lightning dataset fake quant: prompts={len(prompts)}, steps={args.steps}, "
        f"size={args.width}x{args.height}, guidance={args.guidance}, dtype={dtype}, device={device}"
    )
    print(f"Output: {out_dir}")
    for item in prompts:
        print(f"  prompt {item.index:02d}: [{item.category}] {item.source} | {item.prompt[:100]}")

    unet = load_lightning_unet(args.steps, dtype, device)
    pipe = build_pipe(unet, dtype, device)

    baseline_imgs = []
    print("Generating fp16 baseline images...")
    for item in prompts:
        print(f"  fp16 [{item.index}/{len(prompts)}] seed={item.seed}: {item.prompt[:80]}")
        img = generate(pipe, item.prompt, item.seed, device, args)
        img.save(out_dir / f"fp16_{item.index:02d}_seed{item.seed}_{Path(item.source).stem}.png")
        baseline_imgs.append(img)

    if args.method == "gptq":
        gptq_bits = {int(b) for b in args.gptq_bits.split(",") if b.strip()}
        # Enumerate which Linear layers fall into gptq_bits per the bit-assignment logic
        linear_block_bits = parse_block_bits(args.linear_block_bits)
        linear_name_bits = parse_block_bits(args.linear_name_bits)
        gptq_targets: list[str] = []
        for name, mod in pipe.unet.named_modules():
            if not should_quantize_linear(name, mod, args.target):
                continue
            bit = linear_bits_for_layer(name, args.linear_bits, linear_block_bits, linear_name_bits)
            if bit in gptq_bits:
                gptq_targets.append(name)
        if not gptq_targets:
            print("WARNING: --method gptq selected but no target layers; falling back to RTN only")
        else:
            calib_prompts = [p.prompt for p in prompts[: args.calib_prompt_count]]
            print(f"Capturing GPTQ calibration: {len(calib_prompts)} prompts x {args.steps} steps "
                  f"at {args.width}x{args.height}...")
            calib_samples = capture_calibration_samples(pipe, calib_prompts, args, device)
            print(f"  captured {len(calib_samples)} samples")
            gptq_stats = apply_gptq_to_layers(
                pipe.unet, gptq_targets, calib_samples,
                group_size=args.group_size, damping=args.gptq_damping,
                budget_gb=args.gptq_chunk_mem_gb, device=device,
            )
            del calib_samples
            torch.cuda.empty_cache()
        rtn_stats = apply_fake_quant(pipe.unet, args, conv_bits_value, skip_linear_bits=gptq_bits)
        stats = (gptq_stats if gptq_targets else []) + rtn_stats
    else:
        stats = apply_fake_quant(pipe.unet, args, conv_bits_value)

    q_prefix = f"linearw{args.linear_bits}" + (f"_convw{conv_bits_value}" if conv_bits_value else "")
    if args.method == "gptq":
        q_prefix = "gptq_" + q_prefix
    quant_imgs = []
    print("Generating fake-quant images...")
    for item in prompts:
        print(f"  {q_prefix} [{item.index}/{len(prompts)}] seed={item.seed}: {item.prompt[:80]}")
        img = generate(pipe, item.prompt, item.seed, device, args)
        img.save(out_dir / f"{q_prefix}_{item.index:02d}_seed{item.seed}_{Path(item.source).stem}.png")
        quant_imgs.append(img)

    pairs = list(zip(baseline_imgs, quant_imgs))
    metrics = compute_image_metrics(pairs)
    save_contact_sheet(pairs, prompts, out_dir / "contact_sheet.png")

    with (out_dir / "quant_stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL,
                "lightning_repo": LIGHTNING_REPO,
                "steps": args.steps,
                "method": args.method,
                "calib_prompt_count": args.calib_prompt_count if args.method == "gptq" else 0,
                "gptq_damping": args.gptq_damping if args.method == "gptq" else None,
                "linear_bits": args.linear_bits,
                "linear_block_bits": parse_block_bits(args.linear_block_bits),
                "linear_name_bits": parse_block_bits(args.linear_name_bits),
                "conv_bits": conv_bits_value,
                "target": args.target,
                "group_size": args.group_size,
                "conv_include": split_prefixes(args.conv_include),
                "conv_exclude": split_prefixes(args.conv_exclude),
                "prompt_count": len(prompts),
                "height": args.height,
                "width": args.width,
                "guidance": args.guidance,
                "dtype": str(dtype),
                "device": device,
                "skip_patterns": list(SKIP_PATTERNS),
                "prompts": [asdict(item) for item in prompts],
                "image_metrics": metrics,
                "layers": stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    if args.save_unet:
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in pipe.unet.state_dict().items()},
            out_dir / f"sdxl_lightning_{args.steps}step_{q_prefix}_fakequant_unet.safetensors",
        )
    print(f"Image MSE mean: {metrics['mse_mean']:.6f}; MAE mean: {metrics['mae_mean']:.6f}")
    print(f"Image MSE max: {metrics['mse_max']:.6f}; MAE max: {metrics['mae_max']:.6f}")
    print(f"DONE: {out_dir}")


if __name__ == "__main__":
    main()
