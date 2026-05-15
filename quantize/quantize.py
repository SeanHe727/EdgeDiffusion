#!/usr/bin/env python3
"""
GPU 量化脚本：把剪枝 + 蒸馏后的 SD-Turbo UNet 量化成多种方案。

支持的方案 (--recipe)：
  fp16              基线，不量化（仅做 sanity check + 复制权重）
  int8_weight_only  仅权重 INT8 量化     ~50% 大小   Ampere+
  int8_dynamic      权重 + 动态激活 INT8 ~50% 大小   Ampere+   更快
  fp8_dynamic       权重 + 动态激活 FP8  ~50% 大小   Ada+      最接近 fp16 画质
  int4_weight_only  仅权重 INT4 量化     ~25% 大小   Ampere+   最激进

输出：
  models/EdgeDiffuse_r4_<recipe>.safetensors           （量化后 UNet 权重）
  models/EdgeDiffuse_r4_<recipe>.config.json           （元数据 + model_config）

依赖：
  pip install torchao  （所有 INT/FP8/FP4 方案都来自 torchao）

用法：
  python quantize/quantize.py --recipe int8_weight_only
  python quantize/quantize.py --recipe fp8_dynamic
  python quantize/quantize.py --recipe int4_weight_only --skip-sanity   # 跳过推理验证
"""
import os
import sys
import json
import time
import argparse
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_SCRIPT_DIR, "quantize_config.yaml")

# 所有方案的注册表：name -> (torchao Config 类名, 描述)
# torchao 0.17+ 用 Config 类替代了旧的工厂函数。运行时按需 import，避免 torchao
# 没装时 import 阶段就失败。
RECIPES = {
    "fp16":             ("baseline",                                  "fp16 baseline (no quantization)"),
    "int8_weight_only": ("Int8WeightOnlyConfig",                      "INT8 weight-only"),
    "int8_dynamic":     ("Int8DynamicActivationInt8WeightConfig",     "INT8 dynamic activation + INT8 weight (W8A8)"),
    "fp8_dynamic":      ("Float8DynamicActivationFloat8WeightConfig", "FP8 dynamic activation + FP8 weight"),
    "int4_weight_only": ("Int4WeightOnlyConfig",                      "INT4 weight-only"),
}


def _load_config(path):
    if not os.path.exists(path):
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config(class_name):
    """Lazy-resolve a torchao Config class like 'Int8WeightOnlyConfig' -> instance."""
    import torchao.quantization as q
    return getattr(q, class_name)


def resolve_unet_source(args, cfg):
    """Determine where the UNet weights come from.

    Priority:
      1. CLI / config local paths (unet_weights + unet_config) — if both set
      2. HF repo download (unet_repo + unet_filename)
    Returns (weights_path, config_path, base_name) where base_name is used
    for naming the output files (e.g. "distill_final_int8.pt").
    """
    local_w = args.unet_weights or cfg.get("unet_weights")
    local_c = args.unet_config  or cfg.get("unet_config")
    if local_w and local_c:
        base = os.path.splitext(os.path.basename(local_w))[0]
        return local_w, local_c, base

    repo     = cfg.get("unet_repo")
    filename = cfg.get("unet_filename", "distill_final.safetensors")
    if not repo:
        raise ValueError("Need either local unet_weights+unet_config or unet_repo in config")

    from huggingface_hub import hf_hub_download
    print(f"Downloading UNet from HF: {repo}/{filename}")
    weights_path = hf_hub_download(repo, filename)
    config_path  = hf_hub_download(repo, filename.replace(".safetensors", ".config.json"))
    base = os.path.splitext(filename)[0]
    # Prefix with the repo's short name so multiple repos don't collide locally
    repo_short = repo.split("/")[-1]
    base = f"{repo_short}_{base}"
    return weights_path, config_path, base


def load_pruned_unet(unet_weights, unet_config, device, dtype):
    """重建并加载剪枝后的 UNet。复用 prune.pruned_rebuild."""
    from prune.pruned_rebuild import create_unet_from_safetensors
    unet = create_unet_from_safetensors(unet_weights, unet_config)
    unet = unet.to(device=device, dtype=dtype)
    unet.eval()
    return unet


def apply_recipe(unet, recipe):
    """Apply a torchao quantization recipe in-place. Returns the modified unet."""
    if recipe == "fp16":
        return unet
    cfg_cls = _resolve_config(RECIPES[recipe][0])
    from torchao.quantization import quantize_
    # int4_weight_only: torchao 0.17 默认的 PLAIN packing 需要 mslk 库（Meta 内部，PyPI 无）。
    # 改用 TILE_PACKED_TO_4D（marlin 风格），它不依赖 mslk，CUDA 平台原生支持。
    if recipe == "int4_weight_only":
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import Int4PackingFormat
        config = cfg_cls(int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)
    else:
        config = cfg_cls()
    quantize_(unet, config)
    return unet


def sanity_forward(unet, device, dtype, base_model="ChenHe727/EdgeDiffusion_distilled_feat_attn"):
    """跑一次 forward，确认没有 NaN/Inf。

    使用真实的 CLIP text encoder 输出作为 encoder_hidden_states —— 随机张量的
    分布跟真实 CLIP 输出差异很大，INT8/FP8 量化模型对输入分布敏感，用随机数
    会产生大量 NaN 假阳性。
    """
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    with torch.no_grad():
        ids = pipe.tokenizer(
            "a photo of a cat", padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        enc_hs = pipe.text_encoder(ids)[0].to(dtype=dtype)

        sample = torch.randn(1, 4, 64, 64, device=device, dtype=dtype) * pipe.scheduler.init_noise_sigma
        t      = torch.tensor([999], device=device)
        out    = unet(sample, t, encoder_hidden_states=enc_hs).sample

    del pipe
    torch.cuda.empty_cache() if device == "cuda" else None
    assert torch.isfinite(out).all(), "Quantized UNet produced NaN/Inf in sanity forward"
    return tuple(out.shape)


def save_quantized(unet, recipe, out_dir, source_config_path, base_name):
    """Save the (possibly quantized) UNet state dict + sidecar config.

    fp16 baseline uses safetensors（标准格式）。
    量化方案使用 torch.save (.pt)，因为 torchao 的 AffineQuantizedTensor 是 subclass
    tensor，safetensors 不支持序列化这种自定义 tensor 类型。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_cfg = os.path.join(out_dir, f"{base_name}_{recipe}.config.json")

    state = unet.state_dict()
    if recipe == "fp16":
        out_weights = os.path.join(out_dir, f"{base_name}_{recipe}.safetensors")
        save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, out_weights)
    else:
        # torch.save 保留 subclass tensor 信息（量化的 scale/zero_point/packed bits）
        out_weights = os.path.join(out_dir, f"{base_name}_{recipe}.pt")
        torch.save({k: v.detach().cpu() for k, v in state.items()}, out_weights)

    # 复制源 config.json 里的 model_config，再加上量化元数据
    with open(source_config_path) as f:
        src_meta = json.load(f)
    model_config = src_meta.get("model_config", src_meta)
    sidecar = {
        "distill_step": src_meta.get("distill_step", 0),
        "distill_loss": src_meta.get("distill_loss", 0.0),
        "is_ema":       src_meta.get("is_ema", False),
        "quantization": {
            "recipe":      recipe,
            "description": RECIPES[recipe][1],
            "tool":        "torchao" if recipe != "fp16" else None,
        },
        "model_config": model_config,
    }
    with open(out_cfg, "w") as f:
        json.dump(sidecar, f, indent=2)

    size_mb = os.path.getsize(out_weights) / 1024 ** 2
    return out_weights, out_cfg, size_mb


def main():
    parser = argparse.ArgumentParser(
        description="Quantize the distilled SD-Turbo UNet via torchao.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",  default=DEFAULT_CONFIG, help="quantize_config.yaml path")
    parser.add_argument("--recipe",  required=True, choices=list(RECIPES.keys()),
                        help="Quantization recipe to apply")
    parser.add_argument("--unet-weights", default=None, help="Override unet_weights from config")
    parser.add_argument("--unet-config",  default=None, help="Override unet_config from config")
    parser.add_argument("--output-dir",   default=None, help="Override output_dir from config")
    parser.add_argument("--device",       default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--skip-sanity",  action="store_true",
                        help="Skip the forward-pass NaN check (faster but riskier)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    output_dir = args.output_dir or cfg.get("output_dir", "models")
    device     = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_model = cfg.get("base_model", "stabilityai/sd-turbo")

    # Resolve UNet source: local path (CLI/config) or HF repo
    unet_weights, unet_config, base_name = resolve_unet_source(args, cfg)

    if device == "cpu" and args.recipe != "fp16":
        # torchao 的量化 kernel 都需要 CUDA，CPU 上跑量化没意义
        print("ERROR: quantization recipes require CUDA. Run on a GPU machine.")
        sys.exit(1)

    # 选 dtype：fp16 适合大部分量化方案；INT4 tile-packed 强制要求 bf16
    if device == "cuda":
        dtype = torch.bfloat16 if args.recipe == "int4_weight_only" else torch.float16
    else:
        dtype = torch.float32

    print(f"Recipe : {args.recipe}  ({RECIPES[args.recipe][1]})")
    print(f"Device : {device}  dtype={dtype}")
    print(f"Source : {unet_weights}")
    print(f"Output base name : {base_name}")

    # 1. 加载剪枝 UNet
    unet = load_pruned_unet(unet_weights, unet_config, device, dtype)
    src_params = sum(p.numel() for p in unet.parameters())
    print(f"Loaded UNet: {src_params/1e6:.1f}M params")

    # 2. 应用量化
    if args.recipe != "fp16":
        print(f"Applying {args.recipe} ...")
        t0 = time.time()
        unet = apply_recipe(unet, args.recipe)
        print(f"  done in {time.time()-t0:.1f}s")

    # 3. Sanity forward
    if not args.skip_sanity:
        print("Sanity forward ...")
        shape = sanity_forward(unet, device, dtype, base_model=base_model)
        print(f"  OK, output shape={shape}")

    # 4. Save
    print("Saving ...")
    out_weights, out_cfg, size_mb = save_quantized(unet, args.recipe, output_dir, unet_config, base_name)
    print(f"  -> {out_weights}  ({size_mb:.1f} MB)")
    print(f"  -> {out_cfg}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
