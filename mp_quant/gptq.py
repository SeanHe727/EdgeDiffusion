#!/usr/bin/env python3
"""
GPTQ — Post-Training Quantization with Hessian-based error compensation.

Reference:
  Frantar et al., "GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers", ICLR 2023.
  https://arxiv.org/abs/2210.17323

Core idea:
  Naive RTN: `w_int = round(w_fp / scale)` — each weight quantized independently.
  GPTQ:      Quantize column by column. When column i is rounded, distribute
             the rounding error to the remaining un-quantized columns using
             the inverse of the (input) Hessian `H = X^T X`.

This implementation does **fake quantization**: the layer's weight is rounded
to the quant grid but stored as fp16/bf16. Real packed INT4/INT8 storage is
applied later in Stage 3 via torchao or custom packing — that step is
loss-free (rounding already happened here).

Algorithm overview (per-layer):
  1. Collect input activations X (shape: [N, in_features]) for the layer
     from calibration data via a forward hook.
  2. Compute Hessian: H = 2 * X^T X / N
  3. Add diagonal damping: H += λ * mean(diag(H)) * I  (numerical stability)
  4. Compute inverse via Cholesky decomposition: H_inv = cholesky_inverse(H)
  5. Process weight columns one at a time:
       - Quantize current column
       - Compute rounding error
       - Subtract (error * H_inv_row) from remaining un-quantized columns

Group-wise quantization (group_size > 0) computes a fresh scale every
`group_size` columns, matching the standard INT4 weight-only setup
(g=128 is the most common).
"""
import math
import torch
import torch.nn as nn


# —— Quantization grid ─────────────────────────────────────────────────────────

def _quant_grid(bits: int, symmetric: bool = True):
    """Return (qmin, qmax) integer range for given bit-width."""
    if symmetric:
        # e.g. int4 symmetric → [-7, 7]; int8 → [-127, 127]
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
    else:
        qmax = 2 ** bits - 1
        qmin = 0
    return qmin, qmax


def _find_scale_zero(w_block: torch.Tensor, bits: int, symmetric: bool = True):
    """Compute per-row scale (and zero point) for a [out_features, group] block.

    The scale is the smallest positive value such that round(w / scale)
    stays in [qmin, qmax] across the block.
    """
    qmin, qmax = _quant_grid(bits, symmetric)
    if symmetric:
        # |w|.max along the group dim → scale = max / qmax
        max_abs = w_block.abs().max(dim=1, keepdim=True).values
        scale = max_abs / qmax
        scale.clamp_(min=1e-8)              # avoid div-by-zero on dead rows
        zero = torch.zeros_like(scale)
    else:
        w_max = w_block.max(dim=1, keepdim=True).values
        w_min = w_block.min(dim=1, keepdim=True).values
        scale = (w_max - w_min) / (qmax - qmin)
        scale.clamp_(min=1e-8)
        zero = qmin - (w_min / scale).round()
    return scale, zero


def _quantize_row(w_col: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor,
                  bits: int, symmetric: bool = True):
    """Round a single column [out_features] using the (per-row) scale/zero.

    Returns the **fp** value of the dequantized weight (fake quantization).
    """
    qmin, qmax = _quant_grid(bits, symmetric)
    # scale, zero shapes: [out_features, 1]; w_col: [out_features]
    s = scale.squeeze(1)
    z = zero.squeeze(1)
    q = torch.round(w_col / s + z).clamp_(qmin, qmax)
    return (q - z) * s


# —— Hessian accumulator (one per layer being quantized) ──────────────────────

class HessianAccumulator:
    """Accumulates the input-side Gram matrix H = X^T X over calibration batches.

    For a layer y = x @ W^T + b with x of shape [B, in_features], the
    layer-wise MSE reconstruction loss has Hessian (w.r.t. W^T column) equal
    to 2 * E[x^T x]. We accumulate that across calibration batches.

    The accumulator is attached via a forward hook on the layer's INPUT,
    so we get x exactly as the layer sees it.
    """

    def __init__(self, in_features: int, device, dtype=torch.float32):
        self.in_features = in_features
        # H accumulates in fp32 for numerical stability; cast back at end.
        self.H = torch.zeros(in_features, in_features, device=device, dtype=dtype)
        self.n_samples = 0

    def add_batch(self, x: torch.Tensor):
        """x: input tensor of shape [..., in_features]. Will be flattened."""
        x = x.detach()
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(self.H.dtype)
        # Use running average to avoid blow-up: H ← α * H + (1-α) * X^T X / B
        # Equivalently: just sum and divide by total at the end.
        self.H += x.t() @ x
        self.n_samples += x.shape[0]

    def finalize(self) -> torch.Tensor:
        """Return the averaged Hessian H = 2/N * sum(X^T X)."""
        return 2.0 * self.H / max(1, self.n_samples)


# —— Core GPTQ algorithm ──────────────────────────────────────────────────────

@torch.no_grad()
def gptq_quantize_linear(
    layer: nn.Linear,
    hessian: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    damping: float = 0.01,
    symmetric: bool = True,
) -> dict:
    """In-place GPTQ quantization of one nn.Linear.

    Args:
        layer:      the nn.Linear to quantize (weight modified in place)
        hessian:    pre-computed input Hessian, shape [in_features, in_features]
        bits:       target bit-width (4, 8, etc.)
        group_size: number of columns per scale group. -1 = whole row.
        damping:    Cholesky regularization (% of diag mean added to diag)
        symmetric:  symmetric vs asymmetric quantization grid

    Returns dict with stats: {"layer_error", "n_groups", ...}
    """
    W = layer.weight.data.clone().float()      # [out_features, in_features]
    out_f, in_f = W.shape
    device = W.device
    H = hessian.clone().float()                # [in_features, in_features]

    # Mark dead input columns (input feature never activated → row 0 in H).
    # These columns can't be reconstructed and would break Cholesky.
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # Cholesky-based inverse for stability.
    # damping adds λ to diagonal to make H positive-definite.
    diag_mean = torch.mean(torch.diag(H))
    H[torch.arange(in_f), torch.arange(in_f)] += damping * diag_mean

    # GPTQ paper trick: compute upper-triangular Cholesky of H_inv directly.
    # L = Cholesky(H); H_inv = L^-T L^-1.  We need only H_inv's upper-tri rows.
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError as e:
        # Add more damping and retry once.
        H[torch.arange(in_f), torch.arange(in_f)] += 10 * damping * diag_mean
        L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    # Upper-triangular Cholesky of H_inv: Hinv = U^T U.
    U = torch.linalg.cholesky(H_inv, upper=True)

    # Process columns. group_size controls scale granularity:
    #   -1 → one scale per row (per_channel)
    #   N  → fresh scale every N columns (group-wise, standard INT4)
    if group_size <= 0:
        group_size = in_f

    quant_error_total = 0.0
    n_groups = 0
    scales_list = []

    # We process columns left-to-right; at each column i, update remaining
    # columns to absorb the rounding error of column i.
    for g_start in range(0, in_f, group_size):
        g_end = min(g_start + group_size, in_f)
        # Compute scale/zero on the still-unquantized weights in this group
        scale, zero = _find_scale_zero(W[:, g_start:g_end], bits, symmetric)
        scales_list.append((scale.cpu(), zero.cpu()))
        n_groups += 1

        for i in range(g_start, g_end):
            w_col = W[:, i]
            u_diag = U[i, i]                          # scalar
            # Quantize the i-th column to the quant grid
            w_q = _quantize_row(w_col, scale, zero, bits, symmetric)
            # Error introduced by quantizing column i
            err = (w_col - w_q) / u_diag              # [out_features]
            # Update remaining columns to absorb this error
            if i + 1 < in_f:
                W[:, i + 1:] -= err.unsqueeze(1) * U[i, i + 1:].unsqueeze(0)
            # Commit the quantized value back
            W[:, i] = w_q
            quant_error_total += (err * u_diag).pow(2).sum().item()

    # Write back to layer. Keep original dtype.
    layer.weight.data.copy_(W.to(layer.weight.dtype))

    return {
        "layer_error":  quant_error_total / (out_f * in_f),
        "n_groups":     n_groups,
        "in_features":  in_f,
        "out_features": out_f,
        "bits":         bits,
        "group_size":   group_size,
    }


# —— Hook-based driver: collect H + quantize ──────────────────────────────────

def collect_input_hessian(model: nn.Module, target_layer_fqn: str,
                          calib_inputs, device) -> torch.Tensor:
    """Run `model` on each calibration sample, hooking the target layer's INPUT
    to accumulate H = X^T X.

    Args:
        model:            the UNet (set to eval, fp16/bf16, on device)
        target_layer_fqn: dotted name of the nn.Linear to profile
        calib_inputs:     list of dicts with keys (latent_in, timestep, enc_hs)
                          — the same format produced by sensitivity.py
    Returns:
        H tensor of shape [in_features, in_features], on `device`, dtype fp32
    """
    # Find the layer
    target = dict(model.named_modules())[target_layer_fqn]
    if not isinstance(target, nn.Linear):
        raise ValueError(f"{target_layer_fqn} is {type(target).__name__}, not Linear")

    acc = HessianAccumulator(target.in_features, device=device)

    def hook(_mod, args, _kwargs):
        # args[0] is the input tensor (after positional/keyword resolution)
        x = args[0]
        acc.add_batch(x)

    handle = target.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for s in calib_inputs:
                _ = model(s["latent_in"], s["timestep"],
                          encoder_hidden_states=s["enc_hs"]).sample
    finally:
        handle.remove()

    return acc.finalize()
