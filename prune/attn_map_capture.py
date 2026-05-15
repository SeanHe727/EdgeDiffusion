#!/usr/bin/env python3
"""
Attention Map Capture for Knowledge Distillation.

Replaces diffusers AttnProcessor with a custom CaptureAttnProcessor that
exposes head-averaged attention weight matrices during the forward pass.

Motivation: diffusers' default AttnProcessor2_0 uses F.scaled_dot_product_attention
(a fused CUDA kernel) which does NOT expose intermediate attention weights to Python.
We replace it with an equivalent manual implementation that captures them.

The captured map is head-averaged:
  [B*heads, seq_q, seq_k]  →  .view(B, heads, seq_q, seq_k).mean(dim=1)  →  [B, seq_q, seq_k]

This makes the map head-count agnostic — teacher and student may have different numbers
of attention heads after head pruning, but the averaged maps are always directly comparable.

Usage:
    registry = AttentionMapRegistry()
    registry.register(unet, [
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        "mid_block.attentions.0.transformer_blocks.0.attn2",
        ...
    ])

    # Each UNet forward pass populates registry.captured:
    #   {layer_key: Tensor[B, seq_q, seq_k]}  (head-averaged, float32, detached)

    registry.clear()          # call before each forward to drop previous captures
    registry.restore(unet)    # call when done to put back original processors
"""

import torch
import torch.nn as nn
from typing import Any


# ── Module navigation ──────────────────────────────────────────────────────────

def _get_module(model: nn.Module, dotpath: str) -> nn.Module:
    """Navigate a module tree by dot-path, supporting integer indices for ModuleLists.

    Examples:
        _get_module(unet, "mid_block.attentions.0.transformer_blocks.0.attn1")
        _get_module(unet, "down_blocks.2.attentions.1.transformer_blocks.0.attn2")
    """
    m = model
    for part in dotpath.split('.'):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


# ── Custom processor ───────────────────────────────────────────────────────────

class CaptureAttnProcessor:
    """Custom attention processor that captures head-averaged attention maps.

    Mirrors AttnProcessor (diffusers) exactly — same QKV projections, masking,
    and output computation — but also captures the softmax attention weights
    as a side effect before they are used for the context sum.

    Handles both:
      - Self-attention  (encoder_hidden_states=None):  seq_q = seq_k = H*W
      - Cross-attention (encoder_hidden_states given):  seq_q = H*W, seq_k = 77

    Captured map: [B, seq_q, seq_k], float32, detached from computation graph.
    Stored in a shared dict so the registry can collect from all layers.
    """

    def __init__(self, storage: dict, key: str) -> None:
        self.storage = storage
        self.key = key

    def __call__(
        self,
        attn,                            # diffusers Attention module instance
        hidden_states: torch.Tensor,     # [B, seq, C] or [B, C, H, W]
        encoder_hidden_states=None,      # None → self-attn; tensor → cross-attn
        attention_mask=None,
        temb=None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        # — spatial norm (used in video UNets; always None for SD-Turbo) ——————
        if getattr(attn, 'spatial_norm', None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # — handle 4-D channel-first input (some ResNet-adjacent attention) ———
        # align the input to the shape of attention layers
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]

        B, seq_q, _ = hidden_states.shape

        # — prepare attention mask ——————————————————————————————————————————————
        # seq_k differs between self-attn (= seq_q) and cross-attn (= 77)
        seq_k = seq_q if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_k, B)

        # — optional group norm on spatial features ————————————————————————————
        if getattr(attn, 'group_norm', None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # — Q / K / V projections ——————————————————————————————————————————————
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states          # self-attention
        elif getattr(attn, 'norm_cross', False):
            # Cross-attention layer norm on encoder states (present in SD 2.x)
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key_proj   = attn.to_k(encoder_hidden_states)
        value_proj = attn.to_v(encoder_hidden_states)

        # — reshape to [B*heads, seq, head_dim] ———————————————————————————————
        query     = attn.head_to_batch_dim(query)
        key_proj  = attn.head_to_batch_dim(key_proj)
        value_proj = attn.head_to_batch_dim(value_proj)

        # — attention weights: softmax(QK^T / scale) ——————————————————————————
        # attn.get_attention_scores applies scale, adds mask, and runs softmax.
        # Returns [B*heads, seq_q, seq_k] — same as AttnProcessor.
        attention_probs = attn.get_attention_scores(query, key_proj, attention_mask)

        # — capture head-averaged map ——————————————————————————————————————————
        # Reshape [B*heads, seq_q, seq_k] → [B, heads, seq_q, seq_k] → [B, seq_q, seq_k]
        n_heads = attn.heads
        attn_map = (
            attention_probs
            .view(B, n_heads, query.shape[1], key_proj.shape[1])
            .mean(dim=1)
        )
        self.storage[self.key] = attn_map.detach().float()

        # — complete attention: weighted sum of values ————————————————————————
        hidden_states = torch.bmm(attention_probs, value_proj)   # [B*H, seq_q, head_dim]
        hidden_states = attn.batch_to_head_dim(hidden_states)    # [B, seq_q, H*head_dim]
        hidden_states = attn.to_out[0](hidden_states)            # linear projection
        hidden_states = attn.to_out[1](hidden_states)            # dropout

        # — restore 4-D shape if input was channel-first ——————————————————————
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        # — residual + rescale (both disabled in standard SD-Turbo) ———————————
        if getattr(attn, 'residual_connection', False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / getattr(attn, 'rescale_output_factor', 1.0)

        return hidden_states


# ── Registry ───────────────────────────────────────────────────────────────────

class AttentionMapRegistry:
    """Manages CaptureAttnProcessor installation on a UNet.

    Typical lifecycle:
        registry = AttentionMapRegistry()
        registry.register(unet, layer_keys)     # one-time setup

        for batch in dataloader:
            registry.clear()                    # drop previous step's maps
            unet(...)                           # forward fills registry.captured
            maps = dict(registry.captured)      # {key: [B, seq_q, seq_k]}
            # ... compute loss using maps ...

        registry.restore(unet)                  # put back original processors
    """

    def __init__(self) -> None:
        self.captured: dict[str, torch.Tensor] = {}
        self._original_processors: dict[str, Any] = {}

    def register(self, unet: nn.Module, layer_keys: list) -> None:
        """Install CaptureAttnProcessor on each named attention module."""
        for key in layer_keys:
            mod = _get_module(unet, key)
            self._original_processors[key] = mod.processor
            mod.processor = CaptureAttnProcessor(self.captured, key)

    def restore(self, unet: nn.Module) -> None:
        """Restore all original processors."""
        for key, proc in self._original_processors.items():
            _get_module(unet, key).processor = proc
        self._original_processors.clear()

    def clear(self) -> None:
        """Clear captured maps — call before each forward pass."""
        self.captured.clear()
