# Resume / portfolio bullet points

Pick 3-5 from the variants below depending on slot length and the role's focus
(systems / ML infra / applied ML). Numbers are from the final deliverable on
RTX 5070 (Blackwell consumer, sm_120) at 1024×1024, 4 inference steps,
64-prompt eval vs FP16 baseline. See [BENCHMARKS.md](BENCHMARKS.md) for full results.

---

## Project header

**EdgeDiffusion — Edge-Deployable SDXL-Lightning**
Post-training compression pipeline taking the 2.6 B SDXL-Lightning 4-step UNet
to a 1.5 GB packed artifact with 7.3 GB peak inference VRAM at near-zero visual
quality loss.

---

## Long form (ML infra / systems flavor — best for ML / infra roles)

- Built a 5-stage post-training compression pipeline for SDXL-Lightning that **cut deployment artifact size from 4.9 GB to 1.5 GB (-69 %)** and **inference peak VRAM from 10.8 GB to 7.3 GB (-32 %)** on RTX 5070, with per-pixel image MSE 0.0079 vs FP16 baseline (visually indistinguishable in 47/64 prompts).
- Designed block-level mixed-precision bit assignment from per-block sensitivity analysis: **INT4 in the 3 deepest cross-attention blocks** (mid, down_blocks.2, up_blocks.0; 612 of 722 Linears) and **INT8 elsewhere**; verified empirically that this beats elaborate per-layer-type schemes once GPTQ is applied (gap from 25 % to 8 %).
- **Self-implemented GPTQ** (Frantar et al. 2023, ~260 LOC): forward-hook input-Hessian collection, Cholesky-inverted damping, column-by-column quantization with rounding-error compensation. Chunked Hessian accumulation keeps GPU memory bounded on 12 GB hardware (3 chunks × 4 GB budget for 612 W4 Linears).
- Wrote a real packed safetensors format with **INT4 nibble packing (2 weights/byte, offset +8), INT8 raw, per-group fp16 scales, JSON manifest with per-layer kernel hyperparameters**, plus `PackedInt4Linear` / `PackedInt8Linear` / `PackedInt8Conv2d` nn.Module subclasses that keep weights packed on device and dequantize inside forward(). UNet weight VRAM drops from 4.9 GB to 1.5 GB on load.
- Q-LoRA distillation: rank-16 LoRA on all 722 quantized Linears trained against pre-cached fp16 teacher noise predictions (512 samples) in 22 min on RTX 5070 with gradient checkpointing (training VRAM peak 3 GB). Solved an exposure-bias regression at training-strength α (α=16 made simple prompts worse) by introducing **inference-time α scaling** — α=4 keeps simple prompts unchanged while still improving complex worst-case prompts.
- Stage-4 kernel evaluation: characterized the off-the-shelf INT4/INT8 GEMM landscape on Blackwell consumer (sm_120) — **torchao Int8 default per-channel gave 2.34× speedup but broke SDXL quality (MSE 0.28, black images) due to cross-attention outliers; per-group fix loses the fast kernel; torchao Int4 blocked by closed-source dependency; bitsandbytes kernels fall back on sm_120**. Documented the gap and the real next step (TensorRT 10 or custom Triton W4×fp16 kernel).

## Medium form (3-4 bullets — generic technical role)

- Built a 5-stage post-training compression pipeline (block-level mixed-precision quantization → GPTQ → INT4 nibble packing → on-device dequant module replacement → Q-LoRA recovery) that **cut SDXL-Lightning UNet ship size 65 % (4.9 GB → 1.5 GB + 170 MB LoRA) and inference peak VRAM 32 % (10.8 GB → 7.3 GB)** on RTX 5070 with near-zero visual quality loss.
- Self-implemented GPTQ (Hessian + Cholesky + column-wise error compensation, ~260 LOC) and used per-block sensitivity analysis to assign INT4 to the deepest cross-attention blocks (612 layers) and INT8 elsewhere (110 layers); GPTQ reduced W4 image MSE by ~50 % vs naïve round-to-nearest.
- Designed a real packed safetensors format (INT4 nibbles + INT8 + FP16 mixed, per-group fp16 scales, JSON manifest) and `PackedInt4Linear` / `PackedInt8Linear` / `PackedInt8Conv2d` modules; the dequant-on-forward path is mathematically equivalent to in-memory fake-quant generation (verified: same images bit-for-bit modulo nondeterminism) and reduces UNet load VRAM from 4.9 GB to 1.5 GB.
- Rank-16 Q-LoRA distilled from cached fp16 teacher noise predictions; trained in 22 min on RTX 5070 with gradient checkpointing at 3 GB peak. Discovered training-strength α regresses simple prompts (exposure bias) and used inference-time α scaling (α=4 vs trained α=16) to keep 47/64 prompts unchanged while improving worst-case prompts.

## Short form (1-2 bullets — résumé summary line)

- Built end-to-end PTQ pipeline (block-level mixed precision + self-implemented GPTQ + INT4 nibble packing + Q-LoRA) cutting **SDXL-Lightning UNet from 4.9 GB to 1.5 GB (-69 %) and inference peak VRAM from 10.8 GB to 7.3 GB (-32 %)** on RTX 5070 at near-zero visual quality loss vs FP16 baseline.
- Reimplemented GPTQ from the 2023 paper (Hessian + Cholesky + column-wise error compensation, 260 LOC) and integrated it as the W4 quantizer in a custom mixed-precision PTQ pipeline for a 2.6 B-parameter diffusion model on a 12 GB consumer GPU.

## One-liner (résumé skills/projects column)

- EdgeDiffusion: 5-stage PTQ pipeline (mixed-precision + GPTQ + INT4 packing + Q-LoRA) cutting SDXL-Lightning to 1.5 GB / 7.3 GB peak VRAM at near-zero quality loss. [github.com/SeanHe727/...]

---

## Stage-by-stage one-liners (for an oral interview deep-dive)

| Stage | One line |
|---|---|
| Sensitivity analysis | Per-block UNet output drift after jointly quantizing all Linears in that block to INT4/INT8; uses 1 − cosine and relative L2 over 256 calibration samples. |
| Mixed-precision assignment | Block-MP "A": INT4 in mid_block + down_blocks.2 + up_blocks.0 (the three 1280-channel attention-heavy blocks, 612 Linears); INT8 in down_blocks.1 + up_blocks.1 (110 Linears); FP16 elsewhere. |
| GPTQ | Self-implemented Hessian-aware PTQ on W4 Linears only; W8 uses RTN (already at noise floor); chunked Hessian collection (3 chunks × 4 GB) on 12 GB GPU; image MSE 0.0154 (RTN) → 0.0080 (GPTQ). |
| Pack | Signed INT4 nibble packing (2 per byte, offset +8) + per-group fp16 scales for Linear W4, raw int8 + per-channel fp16 scale for Linear W8 and Conv2d W8, FP16 pass-through for the rest; manifest JSON has per-layer bits/shape/kernel-hyperparams. |
| PackedLinear / PackedConv2d | nn.Module subclasses that dequantize the packed weight inside forward() and run F.linear/F.conv2d; lossless math vs in-memory fake quant; reduces UNet load VRAM by 85 %. |
| Q-LoRA | Rank-16 LoRA on all quantized Linears, distilled against fp16 teacher noise predictions cached for 512 samples; gradient checkpointing keeps training VRAM at 3 GB. Inference α=4 (vs trained α=16) recovers from exposure bias. |
| Stage 4 exploration | Characterized why off-the-shelf INT4/INT8 GEMM kernels (torchao, bitsandbytes) don't yet deliver real speedup on Blackwell consumer (sm_120) for SDXL specifically; identified TensorRT 10 / custom Triton as the realistic next step. |

---

## Talking points / FAQ for interview

**"Why mixed-precision instead of uniform INT4?"** Uniform W4 on every Linear gives image MSE 0.0194 — usable but with visible degradation. Block-level mixed precision (deepest 3 blocks at W4, others at W8) drops to 0.0154 (-21 %) at almost no extra storage cost because the deepest 3 blocks hold ~95 % of the Linear parameters anyway.

**"Why GPTQ over round-to-nearest?"** RTN ignores the input distribution and quantizes each weight in isolation. GPTQ uses the input-side Hessian to absorb the rounding error of column i into the not-yet-quantized columns, minimizing the layer-level reconstruction error E\[‖Wx − Ŵx‖²\]. On our W4 Linears, GPTQ cut layer error from ~1e-4 (RTN) to ~1e-5 and final image MSE from 0.0154 to 0.0080 (-48 %). For W8 we skipped GPTQ because RTN error is already at the fp16 noise floor (~1e-7) — no headroom.

**"Why nibble-packed INT4 with per-group scales instead of asymmetric per-channel?"** Symmetric per-group (group_size=128) matches how GPTQ internally chose its scales, so dequant is lossless. Per-channel (one scale per output channel covering the entire input axis) is what torchao's default Int8 path uses and it visibly broke on SDXL — cross-attention outliers in `attn2.to_k/to_v` (in_features=2048) collapsed the per-row scale and made small weights round to zero, killing the latents.

**"Why doesn't Stage 3 give a latency speedup?"** PackedLinear's forward dequantizes the weight to fp16 every call and runs the standard F.linear, so compute time is the same as FP16. The win is purely the weight VRAM footprint. Real latency requires an INT4 × fp16 (or INT8) fused GEMM that operates on the packed weight directly — that's the Stage 4 / TensorRT work.

**"Why is the LoRA helping the worst cases but training α makes mean MSE worse?"** Classic exposure bias of teacher-trajectory distillation: training samples are on the FP16 teacher's denoising trajectory, but inference samples come from the student's own previous-step output. At full training strength (α=16, scaling=1.0) the LoRA over-corrects on the off-distribution inputs of inference; reducing inference α to 4 (scaling=0.25) keeps the helpful corrections on hard cases while not hurting easy cases.

**"Why didn't you use torchao's Int4WeightOnly?"** Torchao 0.17's default INT4 backend requires `mslk>=1.0.0`, an internal kernel library that has only a 0.0.0 stub on PyPI. The legacy `version=1` path hits a `TensorCoreTiledLayout` shape assertion on SDXL Linear shapes. Until that's resolved publicly, our `PackedInt4Linear` (dequant→fp16 GEMM) does the same compute that bitsandbytes' fallback would do — but transparently and with the bit assignment we want.

**"What's the realistic next step?"** TensorRT 10 export with mixed INT4/INT8 quantization. Existing NVIDIA TRT-SDXL recipes apply; we'd plug our bit assignment + packed weights into the TRT calibrator and let it generate a Blackwell-optimized engine. Expected 2-3× latency improvement, additional VRAM savings from activation INT8 in the engine. Estimated 2-4 days.
