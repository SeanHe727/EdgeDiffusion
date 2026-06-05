#!/usr/bin/env python3
"""LoRA adapters for PackedInt4/Int8 Linear modules.

Each wrapped module computes:
    y = base.forward(x)  +  alpha/r * x @ A.T @ B.T

where A: [r, in], B: [out, r]. B is initialized to zero so the wrapped model
behaves identically to the base at training start.

The base's packed buffers stay frozen; gradients flow only through A and B.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAWrappedPackedLinear(nn.Module):
    def __init__(self, base: nn.Module, r: int = 16, alpha: float = 16.0,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_f = base.in_features
        out_f = base.out_features
        # LoRA parameters in fp32 for training stability; cast at forward.
        self.lora_A = nn.Parameter(torch.empty(r, in_f, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.compute_dtype = dtype

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base branch runs normally so grad_x propagates through the frozen
        # base (needed so upstream LoRA layers learn). We rely on the parent
        # UNet having gradient_checkpointing enabled to keep VRAM bounded.
        y_base = self.base(x)
        A = self.lora_A.to(self.compute_dtype)
        B = self.lora_B.to(self.compute_dtype)
        y_lora = F.linear(F.linear(x, A), B) * self.scaling
        return y_base + y_lora

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}"


def wrap_packed_linears_with_lora(module: nn.Module, r: int, alpha: float,
                                  dtype: torch.dtype = torch.float16) -> int:
    """Replace every PackedInt4Linear / PackedInt8Linear leaf in `module` with
    a LoRAWrappedPackedLinear. Returns the number of layers wrapped."""
    from quantization.main.packed.pack_linear import PackedInt4Linear, PackedInt8Linear
    # Collect all (parent, attr_name, child) triples first to avoid mutating
    # the module tree while iterating named_modules() (causes RecursionError).
    targets: list[tuple[nn.Module, str, nn.Module]] = []
    for parent_mod in list(module.modules()):
        if isinstance(parent_mod, (PackedInt4Linear, PackedInt8Linear, LoRAWrappedPackedLinear)):
            continue
        for attr_name, child in parent_mod.named_children():
            if isinstance(child, (PackedInt4Linear, PackedInt8Linear)):
                targets.append((parent_mod, attr_name, child))
    for parent_mod, attr_name, child in targets:
        wrapped = LoRAWrappedPackedLinear(child, r=r, alpha=alpha, dtype=dtype)
        # base is already on its device; move new lora params there (no dtype change).
        device = child.weight_packed.device
        wrapped.lora_A.data = wrapped.lora_A.data.to(device)
        wrapped.lora_B.data = wrapped.lora_B.data.to(device)
        setattr(parent_mod, attr_name, wrapped)
    return len(targets)


def collect_lora_params(module: nn.Module) -> list[torch.nn.Parameter]:
    """Return all lora_A / lora_B parameters in the module."""
    params = []
    for name, p in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)
    return params


def get_lora_state_dict(module: nn.Module) -> dict:
    out = {}
    for name, p in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            out[name] = p.detach().cpu().contiguous()
    return out


def load_lora_state_dict(module: nn.Module, state: dict) -> None:
    own = dict(module.named_parameters())
    for k, v in state.items():
        if k in own:
            own[k].data.copy_(v.to(own[k].dtype))
        else:
            print(f"WARNING: lora key {k} not found in module")
