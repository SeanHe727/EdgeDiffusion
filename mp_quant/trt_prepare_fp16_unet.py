#!/usr/bin/env python3
"""Phase A — Prepare a standard fp16 SDXL-Lightning UNet by merging the packed
weights (dequant) and the Q-LoRA contribution into ordinary nn.Linear/nn.Conv2d
modules. The result is a vanilla UNet2DConditionModel state_dict that can be
loaded by any downstream pipeline (ONNX export, TensorRT, vanilla PyTorch).

Math:
  For each LoRAWrappedPackedLinear with base = PackedIntXLinear:
    W_merged = dequant(base.weight_packed, base.scale) + scaling * (B @ A)
  scaling = alpha / r   (alpha defaults to 4, our chosen inference value)

  For each PackedInt4/Int8Linear not wrapped (none in our final config, but
  supported for completeness):
    W_merged = dequant(...)

  For each PackedInt8Conv2d:
    W_merged = dequant(...)    (reshape back to [out, in_per_group, kh, kw])
    stride / padding / dilation / groups preserved on the new nn.Conv2d.

After replacement, the UNet state_dict contains only standard '.weight' /
'.bias' keys — no '.weight_packed', '.scale', 'lora_A', 'lora_B'.

Verification step (downstream): generate images with this UNet and compare to
the FP16 baseline. Expected MSE ≈ 0.00794 (matches packed+LoRA α=4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mp_quant.build_packed_unet import build_packed_unet
from mp_quant.packed_linear import PackedInt4Linear, PackedInt8Conv2d, PackedInt8Linear
from mp_quant.qlora_lora import (
    LoRAWrappedPackedLinear, load_lora_state_dict, wrap_packed_linears_with_lora,
)


@torch.no_grad()
def merge_lora_wrapped(wrapped: LoRAWrappedPackedLinear, dtype: torch.dtype) -> nn.Linear:
    """Dequant packed base + add LoRA contribution into a standard nn.Linear."""
    base = wrapped.base
    device = base.weight_packed.device
    in_f, out_f = base.in_features, base.out_features
    w_dequant = base.dequant_weight()                              # [out, in] fp16
    A = wrapped.lora_A.to(device=device, dtype=dtype)              # [r, in]
    B = wrapped.lora_B.to(device=device, dtype=dtype)              # [out, r]
    w_lora = (B @ A) * float(wrapped.scaling)                      # [out, in]
    w_merged = (w_dequant + w_lora).to(dtype)
    new = nn.Linear(in_f, out_f, bias=base.bias is not None,
                    dtype=dtype, device=device)
    new.weight.data.copy_(w_merged)
    if base.bias is not None:
        new.bias.data.copy_(base.bias.data.to(device=device, dtype=dtype))
    return new


@torch.no_grad()
def merge_packed_linear_nolora(packed, dtype: torch.dtype) -> nn.Linear:
    device = packed.weight_packed.device
    in_f, out_f = packed.in_features, packed.out_features
    w_dequant = packed.dequant_weight().to(dtype)
    new = nn.Linear(in_f, out_f, bias=packed.bias is not None,
                    dtype=dtype, device=device)
    new.weight.data.copy_(w_dequant)
    if packed.bias is not None:
        new.bias.data.copy_(packed.bias.data.to(device=device, dtype=dtype))
    return new


@torch.no_grad()
def merge_packed_conv2d(packed: PackedInt8Conv2d, dtype: torch.dtype) -> nn.Conv2d:
    device = packed.weight_packed.device
    w_dequant = packed.dequant_weight().to(dtype)                  # [out, in_per_g, kh, kw]
    new = nn.Conv2d(
        in_channels=packed.in_channels, out_channels=packed.out_channels,
        kernel_size=packed.kernel_size, stride=packed.stride,
        padding=packed.padding, dilation=packed.dilation, groups=packed.groups,
        bias=packed.bias is not None, dtype=dtype, device=device,
    )
    new.weight.data.copy_(w_dequant)
    if packed.bias is not None:
        new.bias.data.copy_(packed.bias.data.to(device=device, dtype=dtype))
    return new


def replace_modules_in_unet(unet: nn.Module, dtype: torch.dtype) -> dict:
    """Walk unet, replace LoRAWrappedPackedLinear / PackedIntXLinear /
    PackedInt8Conv2d with standard nn modules holding the merged fp16 weights.
    Mutates in-place. Returns counts."""
    PACKED_TYPES = (PackedInt4Linear, PackedInt8Linear, PackedInt8Conv2d,
                    LoRAWrappedPackedLinear)
    # Collect first; never descend INTO a Packed/LoRA module (we replace them as a unit).
    targets: list[tuple[str, nn.Module, str, nn.Module]] = []
    for parent_mod in list(unet.modules()):
        if isinstance(parent_mod, PACKED_TYPES):
            continue
        for attr_name, child in parent_mod.named_children():
            if isinstance(child, LoRAWrappedPackedLinear):
                targets.append(("lora", parent_mod, attr_name, child))
            elif isinstance(child, (PackedInt4Linear, PackedInt8Linear)):
                targets.append(("linear", parent_mod, attr_name, child))
            elif isinstance(child, PackedInt8Conv2d):
                targets.append(("conv", parent_mod, attr_name, child))

    n = {"lora": 0, "linear": 0, "conv": 0}
    for kind, parent, name, child in targets:
        if kind == "lora":
            new = merge_lora_wrapped(child, dtype)
        elif kind == "linear":
            new = merge_packed_linear_nolora(child, dtype)
        else:
            new = merge_packed_conv2d(child, dtype)
        setattr(parent, name, new)
        n[kind] += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-safetensors", type=Path, required=True)
    ap.add_argument("--packed-manifest", type=Path, required=True)
    ap.add_argument("--lora", type=Path, default=None,
                    help="LoRA state dict path. Omit to dequant-only (no LoRA).")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=4.0,
                    help="Inference-time alpha for LoRA scaling. Default 4 = our final pick.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-stem", type=str, default="sdxl_lightning_unet_fp16_merged")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest from {args.packed_manifest.name}")
    manifest = json.loads(args.packed_manifest.read_text(encoding="utf-8"))
    print(f"Loading packed state from {args.packed_safetensors.name}")
    packed_state = load_file(str(args.packed_safetensors), device="cpu")

    print("Building packed UNet on GPU (PackedInt4Linear / PackedInt8Linear / PackedInt8Conv2d)...")
    unet, stats = build_packed_unet(packed_state, manifest, device, dtype)
    del packed_state
    torch.cuda.empty_cache()
    print(f"  {stats['n_w4']} W4 Linear, {stats['n_w8_linear']} W8 Linear, "
          f"{stats['n_w8_conv']} W8 Conv2d")

    if args.lora is not None:
        print(f"Wrapping with LoRA r={args.lora_rank}, alpha={args.lora_alpha} (scaling = {args.lora_alpha/args.lora_rank:.4f})")
        n_wrapped = wrap_packed_linears_with_lora(
            unet, r=args.lora_rank, alpha=args.lora_alpha, dtype=dtype)
        print(f"  {n_wrapped} layers wrapped, loading LoRA state from {args.lora.name}")
        lora_state = load_file(str(args.lora), device=str(device))
        load_lora_state_dict(unet, lora_state)
    else:
        print("No --lora given — dequant-only (no LoRA merge).")

    print("\nMerging Packed + LoRA into standard fp16 modules...")
    merged = replace_modules_in_unet(unet, dtype)
    print(f"  LoRA wrappers merged:     {merged['lora']}")
    print(f"  Plain Packed Linears:     {merged['linear']}")
    print(f"  Packed Conv2d dequanted:  {merged['conv']}")

    # Sanity: state_dict should now only have standard .weight / .bias keys
    sd = unet.state_dict()
    suspicious = [k for k in sd if "weight_packed" in k or "lora_A" in k or "lora_B" in k or k.endswith(".scale")]
    if suspicious:
        print(f"  WARNING: {len(suspicious)} suspicious keys remain (first 5):")
        for k in suspicious[:5]: print(f"    {k}")
    else:
        print(f"  OK: {len(sd)} tensors in state_dict, no Packed/LoRA artifacts")

    out_path = args.output_dir / f"{args.output_stem}.safetensors"
    print(f"\nSaving merged fp16 UNet → {out_path}")
    save_file({k: v.detach().cpu().contiguous() for k, v in sd.items()}, str(out_path))
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Wrote {sz_mb:.0f} MB")

    sidecar = {
        "source_packed":      str(args.packed_safetensors),
        "source_lora":        str(args.lora) if args.lora else None,
        "lora_rank":          args.lora_rank if args.lora else None,
        "lora_alpha":         args.lora_alpha if args.lora else None,
        "lora_scaling":       (args.lora_alpha / args.lora_rank) if args.lora else None,
        "merge_stats":        merged,
        "size_mb":            sz_mb,
        "base_model":         manifest.get("base_model"),
        "lightning_repo":     manifest.get("lightning_repo"),
        "steps":              manifest.get("steps"),
        "n_state_dict_keys":  len(sd),
    }
    sidecar_path = args.output_dir / f"{args.output_stem}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"  Sidecar: {sidecar_path}")
    print("\nNext: run mp_quant/eval_merged_fp16.py to verify MSE matches packed+LoRA baseline.")


if __name__ == "__main__":
    main()
