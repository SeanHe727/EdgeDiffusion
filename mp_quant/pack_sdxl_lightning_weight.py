#!/usr/bin/env python3
"""Pack a fake-quant SDXL-Lightning UNet into INT4-nibble / INT8 / FP16 safetensors.

Reads:
  - Fake-quant UNet (sdxl_lightning_*_fakequant_unet.safetensors) saved by
    tools/run_sdxl_lightning_dataset_fakequant.py with --save-unet
  - quant_stats.json (gives per-layer bits and group_size)

Writes:
  - <out_dir>/sdxl_lightning_4step_blockmp_a_gptq_packed.safetensors
  - <out_dir>/sdxl_lightning_4step_blockmp_a_gptq_packed.json (manifest)

Per-layer storage:
  W4: weight_packed uint8 [out, in/2]  (two signed nibbles per byte, offset +8)
      scale fp16 [out, n_groups]
  W8: weight_packed int8 [out, in]
      scale fp16 [out, n_groups]
  FP16 pass-through tensors unchanged.

After packing, every Linear weight is reconstructed and compared to its
fake-quant input — max absolute diff is reported (should be 0 for lossless pack).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


QMAX = {4: 7, 8: 127}


def pack_int4_nibbles(q: torch.Tensor) -> torch.Tensor:
    """q: int8 in [-7,7], shape [out, in] (in even). Returns uint8 [out, in/2]."""
    assert q.shape[1] % 2 == 0, f"in_features {q.shape[1]} must be even for nibble pack"
    q_u = (q + 8).to(torch.uint8) & 0x0F           # [-7,7] -> [1,15]
    low = q_u[:, 0::2]
    high = q_u[:, 1::2]
    return (high << 4) | low


def unpack_int4_nibbles(packed: torch.Tensor, out_f: int, in_f: int) -> torch.Tensor:
    """Reverse pack_int4_nibbles. Returns int8 [out, in]."""
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(out_f, in_f, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = low.to(torch.int8)
    out[:, 1::2] = high.to(torch.int8)
    return out


def quantize_to_grid(w: torch.Tensor, bits: int, group_size: int):
    """Re-derive (q_int, scale_fp16) from a fake-quant fp16 weight already on the
    symmetric per-group quant grid. Vectorized over groups."""
    qmax = QMAX[bits]
    out_f, in_f = w.shape
    assert in_f % group_size == 0, f"in_f={in_f} not divisible by group_size={group_size}"
    n_groups = in_f // group_size
    w_g = w.float().view(out_f, n_groups, group_size)
    max_abs = w_g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    s = max_abs / qmax                                   # [out, n_groups, 1]
    q = torch.round(w_g / s).clamp(-qmax, qmax).to(torch.int8).view(out_f, in_f)
    scale = s.squeeze(2).to(torch.float16)
    return q, scale


def rtn_quantize_per_channel(w_2d: torch.Tensor, bits: int):
    """Per-output-channel RTN quantization. Used for Conv W8 (where input
    in_channels * kh * kw may not be divisible by 128). Scale per row only.
    Returns (q [out, in], scale [out, 1] fp16)."""
    qmax = QMAX[bits]
    w_f = w_2d.float()
    max_abs = w_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    s = max_abs / qmax
    q = torch.round(w_f / s).clamp(-qmax, qmax).to(torch.int8)
    return q, s.to(torch.float16)


def verify_layer(orig: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor,
                 bits: int, group_size: int) -> float:
    # Per-channel Conv (group_size == -1): orig is 4D, packed [out, in*kh*kw], scale [out, 1]
    if group_size == -1:
        w_recon = (packed.to(torch.float16) * scale).view_as(orig)
        return (w_recon.float() - orig.float()).abs().max().item()
    # Linear path: per-group along in_features
    out_f, in_f = orig.shape
    q = unpack_int4_nibbles(packed, out_f, in_f) if bits == 4 else packed
    n_groups = in_f // group_size
    q_g = q.view(out_f, n_groups, group_size).to(torch.float16)
    w_recon = (q_g * scale.unsqueeze(2)).view(out_f, in_f)
    return (w_recon.float() - orig.float()).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakequant-dir", type=Path, required=True,
                    help="Dir holding fakequant UNet + quant_stats.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--output-stem", type=str,
                    default="sdxl_lightning_4step_blockmp_a_gptq_packed")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--conv-include", type=str, default="",
                    help="Comma-separated prefixes of Conv2d layer names to RTN-quantize to W8. "
                         "Example: 'down_blocks.1,down_blocks.2,mid_block,up_blocks.0,up_blocks.1'. "
                         "Skipped if name matches SKIP_PATTERNS (conv_in/conv_out/embeddings).")
    ap.add_argument("--conv-bits", type=int, default=8, choices=(8,),
                    help="Bit-width for Conv (W8 only, RTN, per-output-channel).")
    args = ap.parse_args()
    conv_include = tuple(p.strip() for p in args.conv_include.split(",") if p.strip())
    SKIP = ("time_embedding", "time_emb_proj", "add_embedding", "conv_in", "conv_out")

    stats_path = args.fakequant_dir / "quant_stats.json"
    if not stats_path.exists():
        sys.exit(f"Missing {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    group_size = stats["group_size"]

    unet_files = list(args.fakequant_dir.glob("sdxl_lightning_*_fakequant_unet.safetensors"))
    if not unet_files:
        sys.exit(f"No fakequant UNet in {args.fakequant_dir} (re-run runner with --save-unet)")
    unet_path = unet_files[0]
    print(f"Loading fake-quant UNet: {unet_path.name}")
    state = load_file(str(unet_path), device="cpu")
    print(f"  {len(state)} tensors")

    layer_info = {e["name"]: e for e in stats["layers"]}
    print(f"Quantization stats list: {len(layer_info)} entries (bits + type)")

    # Query Conv metadata from a UNet shell (CPU). Needed for stride/padding/etc.
    conv_meta: dict[str, dict] = {}
    if conv_include:
        print(f"Loading UNet shell (CPU) to query Conv hyperparameters...")
        from diffusers import UNet2DConditionModel  # local import
        shell = UNet2DConditionModel.from_config(stats["base_model"], subfolder="unet").to("cpu", torch.float16)
        for n, m in shell.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                conv_meta[n] = {
                    "stride":       list(m.stride),
                    "padding":      list(m.padding) if isinstance(m.padding, (tuple, list)) else [m.padding] * 2,
                    "dilation":     list(m.dilation),
                    "groups":       m.groups,
                    "kernel_size":  list(m.kernel_size),
                    "in_channels":  m.in_channels,
                    "out_channels": m.out_channels,
                    "has_bias":     m.bias is not None,
                }
        del shell
        print(f"  {len(conv_meta)} Conv2d in shell")

    def conv_should_quantize(name: str) -> bool:
        if not conv_include or name not in conv_meta: return False
        if any(p in name for p in SKIP): return False
        return any(name == p or name.startswith(p + ".") for p in conv_include)

    out_tensors: dict[str, torch.Tensor] = {}
    manifest_layers: dict[str, dict] = {}
    fp16_passthrough: list[str] = []
    n_w4 = n_w8_linear = n_w8_conv = 0
    bytes_w4 = bytes_w8_linear = bytes_w8_conv = bytes_fp16 = 0

    weight_keys = sorted(k for k in state if k.endswith(".weight"))
    for wk in weight_keys:
        layer_name = wk[: -len(".weight")]
        info = layer_info.get(layer_name)
        # 1. Linear quantization path (from GPTQ/RTN stats)
        if info is not None and info.get("type") == "Linear" and info["bits"] in (4, 8):
            bits = info["bits"]
            q, scale = quantize_to_grid(state[wk], bits, group_size)
            if bits == 4:
                packed = pack_int4_nibbles(q); n_w4 += 1
                bytes_w4 += packed.numel() + scale.numel() * 2
            else:
                packed = q; n_w8_linear += 1
                bytes_w8_linear += packed.numel() + scale.numel() * 2
            out_tensors[f"{layer_name}.weight_packed"] = packed
            out_tensors[f"{layer_name}.scale"] = scale
            manifest_layers[layer_name] = {
                "bits": bits, "type": "Linear",
                "shape": list(state[wk].shape),
                "group_size": group_size,
                "scale_shape": list(scale.shape),
                "packed_dtype": "uint8_nibble" if bits == 4 else "int8",
            }
            continue
        # 2. Conv W8 RTN per-output-channel path
        if conv_should_quantize(layer_name):
            w = state[wk]                                 # [out, in/groups, kh, kw]
            out_c = w.shape[0]
            w_2d = w.view(out_c, -1)                      # [out, in/groups * kh * kw]
            q, scale = rtn_quantize_per_channel(w_2d, bits=args.conv_bits)
            packed = q
            n_w8_conv += 1
            bytes_w8_conv += packed.numel() + scale.numel() * 2
            out_tensors[f"{layer_name}.weight_packed"] = packed
            out_tensors[f"{layer_name}.scale"] = scale
            cm = conv_meta[layer_name]
            manifest_layers[layer_name] = {
                "bits": args.conv_bits, "type": "Conv2d",
                "shape": list(w.shape),
                "kernel_size": cm["kernel_size"],
                "stride": cm["stride"], "padding": cm["padding"],
                "dilation": cm["dilation"], "groups": cm["groups"],
                "in_channels": cm["in_channels"], "out_channels": cm["out_channels"],
                "group_size": -1,                         # -1 = per-output-channel
                "scale_shape": list(scale.shape),
                "packed_dtype": "int8",
            }
            continue
        # 3. Else FP16 passthrough
        out_tensors[wk] = state[wk]
        fp16_passthrough.append(wk)
        bytes_fp16 += state[wk].numel() * 2

    # All non-.weight tensors (bias, norm, etc.) pass through fp16
    for k, v in state.items():
        if not k.endswith(".weight"):
            out_tensors[k] = v
            fp16_passthrough.append(k)
            bytes_fp16 += v.numel() * v.element_size()

    if not args.no_verify:
        print("Verifying pack round-trip (max abs diff per layer)...")
        max_diff_overall = 0.0
        for i, (layer_name, mani) in enumerate(manifest_layers.items()):
            diff = verify_layer(
                orig=state[f"{layer_name}.weight"],
                packed=out_tensors[f"{layer_name}.weight_packed"],
                scale=out_tensors[f"{layer_name}.scale"],
                bits=mani["bits"], group_size=mani["group_size"],
            )
            max_diff_overall = max(max_diff_overall, diff)
            if (i + 1) % 100 == 0:
                print(f"  checked {i+1}/{len(manifest_layers)}, running max abs={max_diff_overall:.2e}")
        print(f"Verify done. Max abs diff over {len(manifest_layers)} layers: {max_diff_overall:.3e}")
        if max_diff_overall > 1e-3:
            print("WARNING: large reconstruction error — investigate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_st = args.output_dir / f"{args.output_stem}.safetensors"
    out_json = args.output_dir / f"{args.output_stem}.json"

    print(f"Saving packed safetensors: {out_st.name}")
    save_file({k: v.contiguous() for k, v in out_tensors.items()}, str(out_st))

    bytes_w8 = bytes_w8_linear + bytes_w8_conv
    n_w8 = n_w8_linear + n_w8_conv
    total_bytes = bytes_w4 + bytes_w8 + bytes_fp16
    fp16_ref = sum(v.numel() * 2 for v in state.values())
    manifest = {
        "base_model": stats.get("base_model"),
        "lightning_repo": stats.get("lightning_repo"),
        "steps": stats.get("steps"),
        "group_size": group_size,
        "quantization": {
            "method": stats.get("method"),
            "linear_default_bits": stats.get("linear_bits"),
            "linear_block_bits": stats.get("linear_block_bits", {}),
            "linear_name_bits": stats.get("linear_name_bits", {}),
            "conv_bits": stats.get("conv_bits"),
            "skip_patterns": stats.get("skip_patterns", []),
        },
        "size": {
            "fp16_total_bytes": fp16_ref,
            "packed_total_bytes": total_bytes,
            "reduction": 1 - total_bytes / fp16_ref,
            "w4_bytes": bytes_w4, "w8_bytes": bytes_w8, "fp16_bytes": bytes_fp16,
            "n_w4_layers": n_w4, "n_w8_layers": n_w8,
            "n_fp16_tensors": len(fp16_passthrough),
        },
        "layers": manifest_layers,
        "fp16_passthrough": fp16_passthrough,
    }
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    actual_mb = out_st.stat().st_size / 1024 / 1024
    print(f"\nManifest: {out_json}")
    print(f"\nSummary:")
    print(f"  W4 Linear:  {n_w4}    bytes: {bytes_w4/1024/1024:7.1f} MB")
    print(f"  W8 Linear:  {n_w8_linear}    bytes: {bytes_w8_linear/1024/1024:7.1f} MB")
    print(f"  W8 Conv2d:  {n_w8_conv}    bytes: {bytes_w8_conv/1024/1024:7.1f} MB")
    print(f"  FP16 tensors (pass-through): {len(fp16_passthrough)}  bytes: {bytes_fp16/1024/1024:7.1f} MB")
    print(f"  Theoretical packed:  {total_bytes/1024/1024:7.1f} MB")
    print(f"  Actual file:         {actual_mb:7.1f} MB")
    print(f"  FP16 reference:      {fp16_ref/1024/1024:7.1f} MB")
    print(f"  Reduction vs FP16 UNet: {(1 - total_bytes/fp16_ref)*100:.1f}%")


if __name__ == "__main__":
    main()
