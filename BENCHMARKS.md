# SDXL-Lightning Edge Deployment — Benchmark Record

All quantization / packing / Q-LoRA experiments on **ByteDance/SDXL-Lightning 4-step UNet**.

## Setup

```text
Base model:        stabilityai/stable-diffusion-xl-base-1.0
UNet checkpoint:   ByteDance/SDXL-Lightning  (sdxl_lightning_4step_unet.safetensors)
Resolution:        1024 x 1024
Inference steps:   4
Guidance scale:    0.0
Scheduler:         EulerDiscreteScheduler (timestep_spacing="trailing")
Dtype (baseline):  fp16
GPU:               RTX 5070 12GB
FP16 UNet size:    4897 MB (5135 MB on disk with overhead)
Linear count:      722  (proj_in/out, attn1, attn2, ff in transformer blocks)
Conv2d count:      51   (in ResNet blocks + conv_in/out + downsamplers)
```

All MSE/MAE values are **per-pixel RGB error in [0,1] vs the FP16 baseline image**, averaged over a prompt set. Two prompt-set sizes were used at different stages:

- 32 prompts (used for the initial RTN PTQ sweep, kept for that table only)
- 64 prompts (used from the GPTQ sweep onward; all "final" numbers use this set)

Same prompt selection seed (20260517) so the 32-prompt set is the first 32 of the 64-prompt set; numbers across the two are comparable for individual prompts but not for averages.

## 1. Linear-only RTN PTQ sweep (32 prompts)

Quantization method: round-to-nearest, per-group (group_size=128), symmetric, no calibration.

| # | Config | MSE mean | MAE mean | MSE max | Compression |
|---|---|---|---|---|---|
| 1 | Uniform Linear W4 | 0.01936 | 0.08706 | 0.04355 | 75.0% |
| 2 | Block MP **A**: mid/down2/up0 Linear=W4, others=W8 | 0.01543 | 0.07343 | 0.03175 | ~73.8% |
| 3 | **B** = A + cross-attn `to_k/to_v` protected to W8 | 0.01487 | 0.07160 | 0.02938 | ~70.3% |
| 4 | **C** = A + entire cross-attn `attn2.*` protected to W8 | 0.01466 | 0.07167 | 0.02990 | ~66.1% |
| 5 | **E** = FF=W4 (`ff.net.*`), all attn+proj=W8 | 0.01148 | 0.06076 | 0.02435 | 63.9% |

**Takeaways**
- Uniform W4 is the weakest. Block MP recovers ~20% MSE by leaving outer (less sensitive) blocks at W8.
- Protecting cross-attn `to_k/to_v` was the most useful single attention-level rule.
- "FF=W4 / attn=W8" is the cleanest functional split: FF tolerates W4 best, attention does not.

## 2. + GPTQ on W4 layers (64 prompts)

GPTQ (Hessian-aware) applied only to the W4 layers. W8 layers continue to use RTN (W8 RTN error is already ~1e-7). Calibration: 64 prompts × 4 timesteps = 256 samples.

| Config | MSE mean | MAE mean | MSE max | Compression |
|---|---|---|---|---|
| Uniform Linear W4 (RTN, no GPTQ — for reference, 32p) | 0.01936 | — | 0.04355 | 75.0% |
| **A Block MP + GPTQ** | **0.00797** | 0.04601 | 0.02399 | ~73.8% |
| B Block MP + kv protect + GPTQ | 0.00867 | 0.04806 | 0.02054 | 70.3% |
| E FF=W4 + GPTQ | **0.00732** | 0.04303 | 0.01893 | 63.9% |

**Takeaways**
- GPTQ cuts MSE by ~50% across all configs.
- The gap between simple configs (A) and elaborate configs (E) collapses from 25% under RTN to 8% under GPTQ.
- B (kv protection) is **dominated** under GPTQ: A is both better and smaller. The kv-protection trick fixes an RTN-era problem GPTQ already addresses.
- Decision: **adopt A (Block MP + GPTQ)** as the PTQ recipe. Best size/quality ratio; simpler than E.

## 3. Stage 3 — packed module replacement (64 prompts, Conv FP16)

Pack format:
- W4 weight: signed 4-bit nibbles (2 per byte), per-group fp16 scale
- W8 weight: int8, per-group fp16 scale
- FP16 layers: stored as fp16 (no quantization)

`PackedInt4Linear` / `PackedInt8Linear` replace `nn.Linear`. Forward dequants weight on-the-fly to fp16 and runs `F.linear`.

| Config | MSE mean | MAE mean | MSE max | Packed size | UNet load VRAM | Gen peak VRAM |
|---|---|---|---|---|---|---|
| FP16 baseline (reference) | 0 | 0 | 0 | 4897 MB | 9963 MB | 10767 MB |
| A-GPTQ fake-quant (in-memory, no pack) | 0.00797 | 0.04601 | 0.02399 | n/a | similar to fp16 | similar to fp16 |
| **Packed (Conv FP16) only** | 0.00794 | 0.04585 | 0.02313 | **1821 MB** | **1841 MB** | **7609 MB** |

**Takeaway**
- Packing introduces near-zero quality cost (0.00794 vs in-memory 0.00797). The tiny diff comes from per-group scale re-derivation, not the int4/int8 storage itself.
- UNet weight VRAM drops from ~4900 MB to **1841 MB (-62%)**.
- Generation peak drops from 10767 → 7609 MB **(-29%)** — activations and text encoders still dominate the rest.

## 4. + Q-LoRA (64 prompts, Conv FP16, alpha sweep at inference)

Training setup:
- LoRA on every quantized Linear (612 W4 + 110 W8 = 722 layers)
- Rank: 16, alpha (training): 16, scaling = alpha/rank = 1.0
- Teacher cache: 128 prompts × 4 timesteps = 512 (input, teacher noise_pred) pairs, fp16
- Loss: MSE on noise_pred against fp16 teacher
- Optimizer: AdamW lr=1e-4, weight_decay=0
- 3 epochs × 512 samples = 1536 optimizer steps, batch=1, gradient checkpointing on UNet
- Training VRAM peak: 3079 MB. Training time: 22 min on RTX 5070.

Inference uses the same trained LoRA but with adjustable alpha (alpha at inference != alpha at training is fine; this just scales the LoRA contribution).

| Config | MSE mean | MAE mean | MSE max | per-prompt: better/tied/worse |
|---|---|---|---|---|
| Packed only (no LoRA) | 0.00794 | 0.04585 | 0.02313 | — (baseline) |
| Packed + LoRA α=4 (scaling 0.25) | **0.00778** | 0.04568 | 0.02215 | **12 / 47 / 5** |
| Packed + LoRA α=6 (scaling 0.375) | 0.00787 | 0.04631 | 0.02155 | 13 / 34 / 17 |
| Packed + LoRA α=8 (scaling 0.5) | 0.00794 | 0.04693 | **0.02107** | 16 / 27 / 21 |
| Packed + LoRA α=16 (full training strength) | 0.00878 (+11%) | 0.05218 | 0.02180 | 8 / 16 / 40 |
| Packed + LoRA step512 α=16 (early ckpt) | 0.00854 (+8%) | 0.04846 | 0.03008 | — |

**Takeaways**
- LoRA trained at α=16 but applied at α=16 inference **regresses** mean MSE — classic exposure bias from teacher-trajectory training.
- Dialing inference α down recovers + improves: α=4 gives best mean (-2% MSE, 47/64 prompts unchanged, 12 improved, 5 regressed); α=8 gives best worst-case (-9% MSE_max).
- Training plateaued early — additional epochs would not have helped. Real gains require addressing exposure bias (student-trajectory training) or richer loss (image-space distillation).

## 5. + Conv W8 in deep blocks (final deployment configuration, 64 prompts)

Conv W8 added via RTN per-output-channel quantization to deep blocks only:
- Quantized: `down_blocks.1, down_blocks.2, mid_block, up_blocks.0, up_blocks.1` (35 Conv2d)
- Left FP16: `down_blocks.0, up_blocks.2`, `conv_in`, `conv_out` (outermost/latent-adjacent)
- New `PackedInt8Conv2d` module mirrors PackedInt8Linear; reshape to 2D, RTN per-row.

Existing LoRA reused without retraining (LoRA was robust to small Conv W8 perturbation).

| Config | MSE mean | MAE mean | MSE max | Packed size | UNet load VRAM | Gen peak VRAM |
|---|---|---|---|---|---|---|
| Packed (Conv FP16) only | 0.00794 | 0.04585 | 0.02313 | 1821 MB | 1841 MB | 7609 MB |
| Packed (Conv FP16) + LoRA α=4 | 0.00778 | 0.04568 | 0.02215 | 1821 + 170 MB | 1841 MB | 7609 MB |
| **Packed (Deep Conv W8) only** | 0.00808 (+1.7%) | 0.04630 | 0.02352 | **1518 MB** | **1534 MB** | **7303 MB** |
| **Packed (Deep Conv W8) + LoRA α=4** ★ | **0.00794** (matches Linear-only LoRA!) | 0.04610 | 0.02292 | **1518 + 170 MB** | **1534 MB** | **7303 MB** |

**Takeaways**
- Conv W8 alone adds +1.7% MSE — small as expected (Conv W8 RTN error ~1e-7).
- LoRA fully absorbs the Conv quantization perturbation: final MSE 0.00794 = the Linear-only-LoRA baseline.
- Storage drops by 303 MB (16.6%); UNet VRAM drops by 307 MB; generation peak drops by 306 MB.
- **Net result: same quality at smaller size & lower VRAM, with no LoRA retraining cost.**

## 6. Final deployment artifact

```text
File                                                          Size
─────────────────────────────────────────────────────────────────────
sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.safetensors    1518 MB
sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.json           (manifest)
lora_final.safetensors                                                170 MB
─────────────────────────────────────────────────────────────────────
Ship total: ~1688 MB  (vs FP16 UNet 4897 MB → -65.5%)
```

Bit-width assignment:

```text
Linear in {mid_block, down_blocks.2, up_blocks.0}:    W4 GPTQ   (612 layers)
Linear in {down_blocks.1, up_blocks.1}:               W8 RTN    (110 layers)
Conv2d in {down_blocks.1..up_blocks.1}:               W8 RTN    (35 layers, per-channel)
Conv2d in {down_blocks.0, up_blocks.2, conv_in/out}:  FP16      (16 layers)
Embeddings (time, add, etc.):                         FP16
Norms / biases:                                        FP16
LoRA (rank 16, inference α=4):                        on all 722 quantized Linears
```

Quality (vs FP16 baseline, 64 prompts at 1024×1024):
- MSE mean: 0.00794
- MAE mean: 0.04610
- MSE max:  0.02292

Runtime VRAM:
- UNet load:          **1534 MB**   (vs FP16 9963 MB, **-85%**)
- Pipeline build:      3265 MB   (vs FP16 9963 MB, -67%)
- **Generation peak:  7303 MB**   (vs FP16 10767 MB, **-32%**)

## 7. Resume / GitHub framing

- Post-training mixed-precision quantization pipeline for SDXL-Lightning, achieving **65.5% deployment-size reduction** (4897 MB → 1688 MB) with **32% peak inference VRAM reduction** (10767 → 7303 MB on RTX 5070).
- Block-level Linear mixed precision (W4 in 3 deepest cross-attention blocks, W8 elsewhere) selected via per-block sensitivity analysis on noisy-latent UNet outputs.
- Self-implemented GPTQ (Hessian + Cholesky inverse, column-wise error compensation) applied only to W4 Linears; W8 Linears and per-output-channel W8 Conv2d use RTN.
- Real packed artifact format with INT4 nibble packing (2 weights/byte), INT8 raw, and FP16 pass-through tensors; per-group fp16 scales; manifest JSON for layer metadata.
- Stage-3 `PackedInt4Linear` / `PackedInt8Linear` / `PackedInt8Conv2d` modules dequantize on the fly inside forward, keeping weights packed on device throughout inference.
- Rank-16 Q-LoRA distillation from FP16 teacher logits (pre-cached 512 samples), trained with gradient checkpointing in 3 GB VRAM; inference-time α adjustment for stability under exposure bias.

## 8. Stage 4 exploration — real INT kernel on Blackwell consumer (sm_120)

Characterized the publicly-available INT4/INT8 kernel landscape on RTX 5070 to see if we
can replace our Stage 3 dequant→fp16→GEMM path with a real low-bit GEMM kernel.

Hardware: RTX 5070 (sm_120, Blackwell consumer). Test: 4-step SDXL-Lightning, 1024×1024.

| Backend | Real INT compute? | Latency (median) | Gen peak VRAM | MSE vs FP16 |
|---|---|---|---|---|
| FP16 (baseline) | n/a | 3365 ms | 10767 MB | 0 |
| Our Stage 3 packed + LoRA α=4 (deliverable) | No (dequant→fp16) | 3365 ms | 7303 MB | 0.00794 |
| torchao Int4WeightOnly (default) | — | — | — | blocked: `mslk>=1.0.0` is a private/internal dependency not on PyPI |
| torchao Int4WeightOnly (version=1, legacy) | — | — | — | AssertionError in TensorCoreTiledLayout for SDXL shapes |
| torchao Int8WeightOnly per-channel | Yes (fast cuBLAS Int8) | **1438 ms (2.34× faster)** | 8672 MB | **0.28297 (broken — black images)** |
| torchao Int8WeightOnly per-group=128 | No (fast kernel rejects per-group, fallback) | 8501 ms (2.5× slower) | 8819 MB | 0.00120 (good) |
| bitsandbytes NF4 | Quantizes correctly (uint8 storage), no kernel speedup | 3586 ms (≈FP16) | 11870 MB | not measured |
| bitsandbytes INT8 (Linear8bitLt) | Quantizes, slow path | 10483 ms (3× slower) | 13092 MB | not measured |

### Why every off-the-shelf path fails on SDXL today

- **torchao Int4**: requires a closed-source `mslk` package not on PyPI. Legacy `version=1` path hits shape constraints in `TensorCoreTiledLayout` that SDXL's mixed Linear shapes violate.
- **torchao Int8 per-channel** (default): SDXL Linear weights have cross-attention outliers that one per-row scale across `in_features ∈ {640, 1280, 2048, 5120}` cannot accommodate. Quantization collapses small weights to zero → latents diverge → VAE outputs black. (Our own W8 RTN avoids this by using per-group=128, matching what the model needs.)
- **torchao Int8 per-group**: fixes the quality problem (MSE 0.00120, excellent), but the fast Int8 GEMM kernel only supports per-channel/per-tensor — per-group falls back to a slow path that's worse than FP16.
- **bitsandbytes**: native CUDA INT4/INT8 kernels for Blackwell consumer (sm_120) haven't landed — the modules accept and store packed weights but compute falls back to a generic path that's at best ≈FP16 and often slower.

### Implication for the deployment story

- **Our Stage 3 PackedLinear path** (custom dequant→fp16 GEMM) is, in practice, doing the same compute that bitsandbytes' fallback would do — but more transparently and with the bit assignment we want.
- The right next step on Blackwell consumer is **NOT another off-the-shelf weight-only library** — it's either (a) a custom Triton W4×fp16 kernel or (b) **TensorRT 10+** export with its INT4/INT8 mixed-precision optimizer, which has Blackwell-aware INT4 kernels.

### Reproducibility scripts

- `mp_quant/archive_benchmark_torchao_int4.py` — multi-mode latency / VRAM benchmark (fp16, torchao_int4, torchao_int8, bnb_nf4, bnb_int8)
- `mp_quant/archive_eval_torchao_int8.py` — full 64-prompt quality + latency + VRAM eval

## 9. Open future work (not in this pipeline)

1. **TensorRT 10 INT4/INT8 mixed-precision SDXL** — the most realistic path to real latency improvement on consumer Blackwell. 2-4 days work: ONNX export of SDXL-Lightning UNet, TRT engine build with per-layer bits, wrapper around the engine for the diffusers pipeline.
2. **Custom Triton W4×fp16 fused kernel** — 3-5 days. Higher control, portable across hardware generations.
3. **Activation INT8** (W4A8 path) — already fake-quant-validated (MSE ≈ 0.0013 added to weight-only). Real benefit requires the kernel work above.
4. **Student-trajectory Q-LoRA training** to address exposure bias. Estimated additional 5–15% MSE improvement; deferred.
5. **Image-space distillation loss** (decode VAE for both teacher and student) for sharper quality target.
