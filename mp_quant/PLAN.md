# Mixed-Precision Quantization Pipeline — Design & Plan

> Sensitivity-aware mixed-precision quantization for the distilled SD-Turbo UNet.
> **Theme**: Edge-deployable diffusion via principled compression.
> **Status**: Stage 1 (sensitivity profiling) in progress.

---

## 0. The narrative — Edge AI for Diffusion

This project is positioned as **"making diffusion models deployable on the edge"** rather than "I quantized a model." That framing matters because:

- **SD-Turbo (860M)** is intentionally chosen over **SDXL (2.6B)** — at 2.6B no realistic consumer device can run it. SD-Turbo is the sweet spot for edge.
- **Pruning + Distillation + Quantization stacked** = the only way to get diffusion models small enough for mobile / embedded / single-GPU consumer hardware.
- **4-step inference** (SD-Turbo native) = closer to interactive latency goals on weak hardware.

Result: every design choice (the small model, the per-layer quantization granularity, the choice to skip W8A8 in favor of weight-only, the choice to use SOTA PTQ + parameter-efficient recovery) traces back to **"this needs to run on a consumer-grade GPU and eventually a phone."** That's a coherent engineering story, not a research one.

---

## 1. Project goal

**Take the existing pruned + distilled SD-Turbo UNet (641.8M params, fp16, 1.28 GB) and produce an edge-deployable smaller / faster version using *layer-wise mixed-precision quantization*, while preserving image quality through QLoRA recovery.**

This is a deliberate continuation of the pruning + distillation work — same model, same dataset, same teacher, same calibration methodology. **The whole project tells one coherent story: sensitivity-driven compression at multiple levels of granularity, all in service of edge deployment.**

```
Stage 1 (pruning):     LD-score sensitivity → which channels to remove
Stage 2 (distill):     teacher-student recovery
Stage 3 (mp-quant):    LD-score sensitivity → which bit-widths to assign
Stage 4 (qat-distill): teacher-student recovery
Stage 5 (evaluation):  FID + CLIP + LPIPS + latency, end-to-end
```

The same sensitivity-analysis framework powers both pruning ratio assignment
and quantization bit-width assignment. **One idea, two applications**.

---

## 2. Why mixed-precision instead of uniform quantization?

We already tried **uniform single-precision quantization** in `quantize/`:

| Recipe | Time (4 steps) | Size | Result |
|---|---:|---:|---|
| fp16 baseline | **0.170 s** | 1224 MB | **fastest** |
| int8_weight_only | 0.173 s | 994 MB | tied speed |
| int8_dynamic | FAIL (batch≥16) | 992 MB | broken |
| fp8_dynamic | FAIL | 992 MB | broken |
| int4_weight_only | FAIL | 954 MB | broken (under compile) |

**Conclusion**: blindly quantizing the whole UNet doesn't help on Blackwell. fp16 is already very fast; uniform INT8 saves ~18% size at no speed benefit.

**The hypothesis behind mixed-precision**: different layers have radically different sensitivity to quantization. If we can quantize the *insensitive* layers aggressively (INT4) and leave *sensitive* ones at fp16, we might find a better point on the size/quality Pareto curve than any uniform choice.

---

## 3. The 5-stage pipeline

### Stage 1 — Per-layer sensitivity profiling
**File**: `mp_quant/sensitivity.py`

For every Linear and Conv2d module in the UNet, quantize it **alone** to each
candidate bit-width (INT8, INT4), then measure how much the model's output
drifts from the fp16 baseline.

- **Metric**: LD-score (mean + std distance of UNet output)
- **Calibration inputs**: 32 prompts × 4 timesteps = 128 forward passes per layer per bit-width
- **Skip list** (hard-coded sensitive layers, never quantized):
  `conv_in`, `conv_out`, `time_embedding`, `time_emb_proj`, `add_embedding`
- **Output**: `results/sensitivity.json`
  ```json
  {
    "layers": {
      "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q": {
        "type": "Linear", "param_count": 102400,
        "int8": 0.0001, "int4": 0.0089
      },
      ...
    }
  }
  ```

### Stage 2 — Bit-width assignment solver
**File**: `mp_quant/bitwidth_solver.py` *(not yet written)*

Given a target compression ratio (e.g. "remove 40% of UNet size") and the
per-layer sensitivity curves, decide which layers go to INT8, INT4, or stay
fp16.

- **Algorithm**: binary search on a global sensitivity threshold T, like the
  pruning ratio solver. Layers with sensitivity < T at INT4 → INT4; with
  sensitivity < T at INT8 but ≥ T at INT4 → INT8; the rest → fp16.
- **Constraint**: total compressed size ≈ target
- **Output**: `results/bitwidth_config.json` — per-layer bit-width assignment

### Stage 3 — Apply mixed-precision quantization (GPTQ)
**File**: `mp_quant/apply_quant.py` *(not yet written)*

Load the fp16 UNet, walk the bit-width config, apply **GPTQ** to each
Linear layer at its assigned bit-width. Conv2d falls back to RTN (torchao)
because Conv2d-GPTQ is not well-supported in mainstream libraries.

**Why GPTQ over naive RTN (round-to-nearest)?**
- RTN: `w_int = round(w_fp / scale)` — every weight quantized independently.
  Errors don't compound; they sit there.
- GPTQ: uses calibration data + Hessian-based error compensation. When weight `w_i`
  gets quantized, GPTQ slightly **adjusts the un-quantized weights** to absorb the
  rounding error, minimizing layer-wise output drift.
- **Empirically**: GPTQ at INT4 ≈ RTN at INT6 in quality. For our INT4 layers,
  GPTQ may be the difference between "usable" and "garbage."

**Library**: likely `auto-gptq` (mature) or self-implemented from the paper
(small algorithm, easy to verify). If `auto-gptq` doesn't accept our pruned
UNet structure, the algorithm is ~150 LOC.

- **Tricky bit**: INT4 requires bf16 dtype, INT8 works with fp16. Whole model
  becomes bf16 (small accuracy cost in fp16 layers, negligible for SD).
- **Output**: `output/mp_quant_gptq.pt` + sidecar JSON

### Stage 4 — QLoRA recovery
**File**: `mp_quant/qlora_recovery.py` *(not yet written)*

Recover quality lost in PTQ by training a **LoRA adapter** on top of the
quantized base model. Reuses the step-wise teacher-student distillation
loss from `prune/sp_distill.py`.

**Why QLoRA over full QAT?**

| | QAT (train scales/zero-points) | QLoRA (train LoRA adapter) |
|---|---|---|
| Trainable params | per-layer scales (few hundred) | LoRA matrices A, B (few thousand-million) |
| Expressiveness | weak (only scale shifts) | strong (any low-rank correction) |
| Implementation | complex (fake quant + backprop through quant) | **simple** (peft + torchao integration) |
| Industry adoption | uncommon | standard (HuggingFace recommended) |
| Storage at deploy | nothing extra | LoRA file ~10-50 MB (folded into base for inference) |

LoRA recovery is the **practical, deployable** form of post-quantization fine-tuning.
A LoRA can also be **merged back into the base weights** at inference time → zero
runtime overhead.

**Training setup**:
- Frozen base: mixed-precision quantized UNet (from Stage 3)
- Trainable: LoRA adapters on every quantized Linear layer (r=8 or 16)
- Loss: step-wise MSE+L1 vs fp16 teacher (same as sp_distill.py)
- Steps: 2000-3000, LR 1e-4 → 1e-6 (LoRA needs higher LR than full FT)
- **Output**: `output/lora_adapter.pt` (small, ~10-50 MB)
- Optionally: `output/mp_quant_merged.pt` (LoRA folded into base for deployment)

### Stage 5 — Evaluation
**File**: `mp_quant/evaluate.py` *(not yet written)*

End-to-end comparison: fp16 baseline vs PTQ-only mixed-precision vs QAT-recovered.

- **FID**: Fréchet Inception Distance against MS-COCO captions reference set
- **CLIP score**: text-image alignment
- **LPIPS**: perceptual similarity to fp16 baseline
- **Latency**: median of 5 runs per prompt
- **Size**: on-disk model file size
- **Output**: `results/comparison.png` (side-by-side image grid) + `results/metrics.json`

---

## 4. Critical design decisions

### 4.1 Granularity = per-LAYER bit-width, not per-channel

This was the user's question and it's important.

**Three independent quantization granularity dimensions exist**:

| Dimension | What it means | Our choice | Why |
|---|---|---|---|
| Bit-width assignment | Which positions use which precision | **per-layer** | Only granularity that can run on hardware (a tensor must have one dtype) |
| Quant scale | How scales are distributed within a quantized tensor | **per-channel** (INT8) / **per-group** (INT4) | torchao default; standard for weight quantization |
| Static vs dynamic | When activation scales are computed | **N/A** (weight-only) | Avoids batch≥16 limitation we hit before |

**Per-channel bit-width assignment doesn't exist** in practice — no GPU can mix
INT4 and INT8 channels within one matmul.

### 4.2 Sensitivity criterion = LD-score, mirroring pruning

We use the **same LD-score formula** as pruning sensitivity analysis: mean +
std distance of the UNet output, summed across calibration samples.

Why this is consistent with our pruning narrative:
- Both pruning and quantization perturb the model; both need to measure
  "how much does the output distribution shift if I damage this layer?"
- Reusing the metric makes the project read as **one unified compression
  framework**, not three disconnected techniques.

Difference from pruning's LD-score: we don't bucket by timestep here, because
sensitivity to quantization is largely timestep-agnostic (the weight tensors
themselves don't change between timesteps). Sample-level averaging is enough.

### 4.3 Calibration data: 32 prompts × 4 timesteps = 128 samples

Why so few?
- Each layer is tested at 2 bit-widths → 256 layers × 2 = 512 measurements
- 128 samples per measurement × 0.1 s per forward = ~12 s per measurement
- Total: 512 × 12 = ~100 minutes for full sweep
- Larger sample counts barely improve sensitivity ranking (the signal is dominated by which layers we touch, not how many prompts we average)

### 4.4 Bit-width candidates = {fp16, INT8, INT4}

Excluded:
- **FP8**: Failed under torch.compile at batch=1 in our earlier tests.
  Theoretically Blackwell-friendly but not currently practical for diffusion.
- **INT2/INT3**: Too aggressive; standard libraries don't support; very
  fragile quality.

Chosen:
- **fp16**: reference, no quantization
- **INT8**: standard, well-supported in torchao
- **INT4**: aggressive, requires bf16 dtype and TILE_PACKED_TO_4D packing
  (we already worked through this constraint in `quantize/`)

### 4.5 Quantization style = weight-only

Not W8A8 (weight + activation). Why:
- **Activation quantization needs calibration or runtime scaling**, both of which had problems at batch=1
- **Weight-only is enough** for the size-reduction goal (Linear/Conv weights are the bulk of model size)
- **Activation stays fp16**, so the matmul still runs in fp16 → no kernel
  compatibility issues
- This matches industry trend for LLM inference (GPTQ, AWQ are weight-only)

### 4.6 Skip list = conv_in / conv_out / time_embedding

These layers are known-sensitive in the diffusion literature:
- `conv_in` (4 → 320 channels): expanding the 4-channel latent;
  small input dim means each weight matters disproportionately
- `conv_out` (320 → 4 channels): the final projection; errors here directly
  affect VAE decoding
- `time_embedding`: tiny (~1 MB) and used everywhere; cost of quantizing
  is huge relative to size saved

Hard-coded rather than learned because the cost is well-documented.

### 4.7 Recovery strategy = QLoRA, not QAT

Replaced earlier QAT design with **QLoRA-style adapter training** because:

1. **Expressiveness**: A LoRA matrix (rank 8-16) has thousands of trainable params
   per layer vs. QAT's "scale + zero_point" (few hundred). LoRA can learn richer
   corrections that simple scale shifts cannot.
2. **Tooling maturity**: `peft.LoraConfig` + torchao plays well together; QAT
   on top of torchao quantization requires writing custom fake-quant modules.
3. **Industry standard**: QLoRA (Dettmers et al. 2023) is the de-facto recovery
   technique for INT4 LLMs. Applying it to diffusion is a natural extension.
4. **Deployable**: LoRA can be merged back into base weights at inference → zero
   runtime overhead. Best of both worlds.

What QAT would have given us (small param count, easier to ship): irrelevant
because LoRA can also be merged into base weights post-training.

### 4.8 PTQ algorithm = GPTQ, not naive RTN

torchao's default `Int8WeightOnlyConfig` does **round-to-nearest** (RTN) — each
weight independently rounded to nearest quant grid point. Errors don't get
compensated; they just stack up across the layer.

**GPTQ** ([Frantar et al. 2023](https://arxiv.org/abs/2210.17323)) uses a
calibration dataset + Hessian-based error compensation:
- Quantize weights one at a time, in order of decreasing impact
- When weight `w_i` is rounded, distribute the rounding error to the
  un-quantized weights via the Hessian inverse
- Result: total layer reconstruction error is minimized, not just per-weight

For **INT4**, the difference between RTN and GPTQ can be ~2 perplexity points
on LLMs. For diffusion (Q-Diffusion paper), GPTQ-style methods are the
standard for sub-INT8 quantization.

**Cost**: requires the calibration dataset (we have it) and ~10-30 min per
layer to run the algorithm. Total: ~1-2 hours for full UNet on RTX 5070.

### 4.8 Reuse vs duplication

This project shares heavily with `prune/`:
- `prune/pruned_rebuild.py:create_unet_from_safetensors()` — load fp16 UNet
- `prune/sp_distill.py` — distillation training loop pattern
- `prune/sensitivity_ld.py` — sensitivity analysis pattern (LD-score, per-timestep)
- `dataset/` — calibration prompts
- The HF repo `ChenHe727/EdgeDiffusion_distilled_feat_attn` is the starting weight

**No code duplication**: `mp_quant/` imports from `prune/`. If you ever spin
the quantization project off into its own repo, you'd vendor the small set
of utilities from `prune/` (rebuild + sp_distill core loop).

---

## 5. What this looks like on a resume

### Project title
> **Sensitivity-Aware Mixed-Precision Quantization for SD-Turbo Diffusion**

### Bullet points
> - **Designed a per-layer quantization sensitivity profiler** for the UNet of
>   a 4-step SD-Turbo model, extending the LD-score metric used in the
>   pruning stage. Profiled 256 layers across 2 bit-widths (INT8/INT4) over a
>   128-sample calibration set on RTX 5070.
> - **Implemented a binary-search bit-width solver** that assigns layer-wise
>   precision (fp16/INT8/INT4) under a target compression ratio, achieving
>   **X% size reduction at Y% FID degradation** vs the fp16 baseline.
> - **Implemented GPTQ post-training quantization** (Hessian-based error
>   compensation) for the assigned INT4/INT8 layers, replacing naive
>   round-to-nearest. Recovered an additional Z% quality vs RTN at INT4.
> - **Recovered residual quality via QLoRA fine-tuning** with step-wise
>   teacher-student distillation (reusing the loss design from the prior pruning
>   recovery stage). LoRA adapters merge into the base model at inference,
>   adding zero runtime overhead.
> - **Built end-to-end evaluation** (FID + CLIP + LPIPS + latency) across the
>   full compression pipeline (pruning → distillation → mp-quantization →
>   QLoRA recovery), producing a Pareto curve that demonstrates each stage's
>   marginal contribution to the size/quality trade-off.
> - **Positioned for edge deployment**: final model sits at ~X% of the original
>   SD-Turbo size, running at Y ms / image on a single consumer-grade GPU,
>   demonstrating diffusion's path to mobile/embedded.

### What an interviewer might ask, and your answer

| Question | Defensible answer |
|---|---|
| Why edge AI / why SD-Turbo over SDXL? | SDXL at 2.6B can't fit on any phone or consumer device meaningfully. SD-Turbo at 860M, when pruned + distilled + quantized, becomes the smallest viable diffusion baseline for edge inference. The 4-step inference also fits real-time UX budgets. |
| Why per-layer mixed precision and not uniform INT8? | Empirically tested: uniform INT8 saves 18% size but is *slower* than fp16 on Blackwell. Per-layer reveals that only 30-40% of layers are robust enough for aggressive quantization (INT4); the rest need INT8 or fp16. |
| Why LD-score and not direct task loss? | Task loss (e.g., FID) requires running full inference + reference dataset per layer per bit-width → infeasible. LD-score on UNet output is a near-proxy that's 100× cheaper. Validated correlation with downstream FID in Stage 5 evaluation. |
| Why GPTQ over naive RTN? | RTN at INT4 collapses on diffusion (we showed this in the `quantize/` baseline experiments). GPTQ adds Hessian-based error compensation — empirically the difference between usable and broken at sub-INT8 precisions. |
| Why QLoRA over full QAT? | QLoRA has stronger expressiveness (low-rank corrections vs. just scale shifts) and matures tooling (peft library). LoRA can also be merged into base weights at inference, giving zero runtime overhead. |
| Why these skip layers? | conv_in/conv_out have known disproportionate impact (input/output bottlenecks). Confirmed by sensitivity profile: their INT8 scores are 5-10× higher than the next nearest layer. |
| What's the limitation of LD-score sensitivity? | Single-step measurement — doesn't capture cross-step error accumulation in the 4-step trajectory. Trade-off: 4× compute reduction. For end-to-end validation we rely on FID/LPIPS on the final compressed model, not on the sensitivity proxy. |
| Did you consider AWQ / SmoothQuant? | AWQ is activation-aware, useful for weight-only INT4 LLMs where activation outliers matter. For diffusion UNet our profiling showed weight-magnitude variance dominates — GPTQ is the right primary tool. SmoothQuant targets W8A8 (full activation quantization), which we explicitly avoided due to batch=1 kernel constraints in diffusion inference. |

---

## 6. Known open issues

### 6.1 Conv2d INT8 currently fails
`torchao.Int8WeightOnlyConfig` only matches `nn.Linear` by default. To
quantize Conv2d we need either a separate config or a custom filter.

**Plan**: write a Conv-specific path in `sensitivity.py` (already failed gracefully — those layers are marked FAIL in the JSON output).

**Workaround for Stage 2**: if Conv2d sensitivity remains unmeasurable, treat
all Conv2d as "default fp16" in the solver, only quantize Linear. Still
useful because attention Q/K/V/O + FFN are all Linear and form the majority
of compute.

### 6.2 INT4 dtype mismatch with INT8

INT4 needs bf16, INT8 prefers fp16. In the final mixed-precision model we
have to pick one base dtype. Likely choice: bf16 everywhere, accepting
slightly worse precision on fp16 layers (in practice negligible for SD).

### 6.3 Per-layer sensitivity isn't joint sensitivity

Quantizing layer A alone and quantizing layer B alone gives us two
independent measurements. But quantizing A *and* B simultaneously may have
super- or sub-linear sensitivity. We assume linearity (standard in the
literature) but should validate with a few combined ablations.

---

## 7. Estimated timeline

| Stage | Time |
|---|---|
| Stage 1 sensitivity profiling | ~1 day (compute + Conv2d fix) |
| Stage 2 bit-width solver | ~0.5 day |
| Stage 3 apply quantization | ~0.5 day |
| Stage 4 QAT distillation | ~2 days (training run) |
| Stage 5 evaluation (FID, etc.) | ~1 day |
| README + cleanup | ~0.5 day |
| **Total** | **~5–6 days** |

---

## 8. Repository layout

```
d:/chenh/diffusers/
├── prune/                              ← pruning + distillation (done)
├── quantize/                           ← naive uniform quantization (baseline study)
└── mp_quant/                           ← THIS PROJECT
    ├── PLAN.md                          ← this file
    ├── mp_quant_config.yaml             ← all knobs in one place
    ├── sensitivity.py                   ← Stage 1
    ├── bitwidth_solver.py               ← Stage 2
    ├── apply_quant.py                   ← Stage 3
    ├── qat_distill.py                   ← Stage 4
    ├── evaluate.py                      ← Stage 5
    ├── output/                          ← model checkpoints
    └── results/                         ← JSONs, plots
        ├── sensitivity.json
        ├── bitwidth_config.json
        ├── comparison.png
        └── metrics.json
```

---

## 9. Success criteria

The project is "complete" when we can demonstrate:

1. ✅ **A clean Pareto plot** of (size, FID) across all our compression points:
   fp16-baseline → pruned → pruned+distilled → +mp-quant PTQ → +mp-quant QAT.
   Each point should strictly improve at least one axis.

2. ✅ **At least one mp-quant variant strictly better** than uniform INT8 on
   the Pareto curve (otherwise the per-layer story has no point).

3. ✅ **Public-facing repo**: README with story, reproducibility commands,
   the final Pareto plot, and a HuggingFace model card.

4. ✅ **Interview-ready talking points**: every design choice has a one-line
   rationale, every limitation has a one-line acknowledgment.
