#!/usr/bin/env python3
"""Packed Linear modules — weight stored as INT4 nibbles or INT8 on-device,
dequanted to fp16 inside forward() (then released when forward returns).

The packed weight + scale buffers stay on GPU at all times. Each forward
materializes the dequanted fp16 weight temporarily, runs F.linear, and lets
PyTorch free the temporary on return. Peak VRAM is therefore dominated by
the packed buffers, not the dequanted weights.

Both modules implement the same interface as nn.Linear (forward(x) -> y).
The `from_state` factory builds a packed module from packed-state tensors
(no copy beyond what's needed to register the buffer).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unpack_int4_nibbles(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """packed: uint8 [out, in/2]. Returns int8 [out, in] with values in [-7, 7]."""
    out_f = packed.shape[0]
    low = (packed & 0x0F).to(torch.int16) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(out_f, in_features, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = low.to(torch.int8)
    out[:, 1::2] = high.to(torch.int8)
    return out


class PackedInt4Linear(nn.Module):
    """Drop-in replacement for nn.Linear with INT4 nibble-packed weight.

    Buffers (live on device throughout):
      weight_packed: uint8 [out, in/2]
      scale:         fp16  [out, n_groups]
      bias:          fp16  [out]            (optional)

    forward(x): dequants weight to fp16 (temporary), runs F.linear, returns y.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int,
                 bias: bool = False, dtype: torch.dtype = torch.float16):
        super().__init__()
        if in_features % 2 != 0:
            raise ValueError(f"in_features {in_features} must be even for INT4 nibble pack")
        if in_features % group_size != 0:
            raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = in_features // group_size
        self.compute_dtype = dtype

        self.register_buffer("weight_packed",
                             torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scale",
                             torch.zeros(out_features, self.n_groups, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def dequant_weight(self) -> torch.Tensor:
        q = _unpack_int4_nibbles(self.weight_packed, self.in_features)
        q_g = q.view(self.out_features, self.n_groups, self.group_size).to(self.compute_dtype)
        return (q_g * self.scale.unsqueeze(2)).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequant_weight()
        return F.linear(x, w, self.bias)

    @classmethod
    def from_state(cls, weight_packed: torch.Tensor, scale: torch.Tensor,
                   group_size: int, bias: torch.Tensor | None = None,
                   dtype: torch.dtype = torch.float16) -> "PackedInt4Linear":
        out_f, half_in = weight_packed.shape
        in_f = half_in * 2
        m = cls(in_features=in_f, out_features=out_f, group_size=group_size,
                bias=bias is not None, dtype=dtype)
        m.weight_packed.copy_(weight_packed.to(torch.uint8))
        m.scale.copy_(scale.to(dtype))
        if bias is not None:
            m.bias.data.copy_(bias.to(dtype))
        return m

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, group={self.group_size}, "
                f"bias={self.bias is not None}, bits=4")


class PackedInt8Linear(nn.Module):
    """nn.Linear with INT8 weight + per-group fp16 scale."""

    def __init__(self, in_features: int, out_features: int, group_size: int,
                 bias: bool = False, dtype: torch.dtype = torch.float16):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = in_features // group_size
        self.compute_dtype = dtype

        self.register_buffer("weight_packed",
                             torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale",
                             torch.zeros(out_features, self.n_groups, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def dequant_weight(self) -> torch.Tensor:
        q_g = self.weight_packed.view(self.out_features, self.n_groups, self.group_size).to(self.compute_dtype)
        return (q_g * self.scale.unsqueeze(2)).view(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequant_weight()
        return F.linear(x, w, self.bias)

    @classmethod
    def from_state(cls, weight_packed: torch.Tensor, scale: torch.Tensor,
                   group_size: int, bias: torch.Tensor | None = None,
                   dtype: torch.dtype = torch.float16) -> "PackedInt8Linear":
        out_f, in_f = weight_packed.shape
        m = cls(in_features=in_f, out_features=out_f, group_size=group_size,
                bias=bias is not None, dtype=dtype)
        m.weight_packed.copy_(weight_packed.to(torch.int8))
        m.scale.copy_(scale.to(dtype))
        if bias is not None:
            m.bias.data.copy_(bias.to(dtype))
        return m

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, group={self.group_size}, "
                f"bias={self.bias is not None}, bits=8")


class PackedInt8Conv2d(nn.Module):
    """nn.Conv2d with INT8 weight + per-output-channel fp16 scale.

    Weight is stored as int8 [out, in_per_group * kh * kw] and reshaped back to
    [out, in_per_group, kh, kw] inside dequant_weight(). Per-output-channel
    scale is shape [out, 1].

    Stride / padding / dilation / groups are stored as attributes and passed to
    F.conv2d. Bias (if present) is fp16.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride,
                 padding, dilation, groups: int = 1, bias: bool = False,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.dilation = tuple(dilation)
        self.groups = groups
        self.compute_dtype = dtype
        in_per_group = in_channels // groups
        flat_in = in_per_group * self.kernel_size[0] * self.kernel_size[1]
        self.register_buffer("weight_packed",
                             torch.zeros(out_channels, flat_in, dtype=torch.int8))
        self.register_buffer("scale",
                             torch.zeros(out_channels, 1, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def dequant_weight(self) -> torch.Tensor:
        w_flat = self.weight_packed.to(self.compute_dtype) * self.scale       # [out, in*kh*kw]
        return w_flat.view(self.out_channels, self.in_channels // self.groups,
                           self.kernel_size[0], self.kernel_size[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.dequant_weight(), self.bias,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    @classmethod
    def from_state(cls, weight_packed: torch.Tensor, scale: torch.Tensor,
                   manifest_entry: dict, bias: torch.Tensor | None = None,
                   dtype: torch.dtype = torch.float16) -> "PackedInt8Conv2d":
        m = cls(
            in_channels=manifest_entry["in_channels"],
            out_channels=manifest_entry["out_channels"],
            kernel_size=manifest_entry["kernel_size"],
            stride=manifest_entry["stride"],
            padding=manifest_entry["padding"],
            dilation=manifest_entry["dilation"],
            groups=manifest_entry["groups"],
            bias=bias is not None, dtype=dtype,
        )
        m.weight_packed.copy_(weight_packed.to(torch.int8))
        m.scale.copy_(scale.to(dtype))
        if bias is not None:
            m.bias.data.copy_(bias.to(dtype))
        return m

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, k={self.kernel_size}, "
                f"stride={self.stride}, pad={self.padding}, groups={self.groups}, "
                f"bias={self.bias is not None}, bits=8")
