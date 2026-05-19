# EdgeDiffusion — SDXL-Lightning for Edge Deployment

A reproducible 5-stage post-training pipeline that takes **SDXL-Lightning 4-step UNet** (2.6 B params, 4.9 GB fp16) to an **edge-deployable artifact at 1.5 GB packed + 170 MB LoRA (-65 % ship size)** with **inference peak VRAM 7.3 GB (-32 %)** at near-zero visual quality loss on consumer GPU.

> **TL;DR**: Block-level mixed-precision quantization (W4 in 3 deepest cross-attention blocks, W8 elsewhere) + self-implemented GPTQ + INT4-nibble packed safetensors + on-device `PackedInt4Linear`/`PackedInt8Linear`/`PackedInt8Conv2d` module replacement + rank-16 Q-LoRA distillation. All five stages are implemented, tested end-to-end, and reproducible from this repo.

The repository previously contained a parallel SD-Turbo (860 M) pruning + distillation pipeline; that work has been archived. The SDXL-Lightning pipeline here is the current focus and lives on a larger, more capable base model.

---

## Headline numbers (RTX 5070, 12 GB, sm_120 Blackwell consumer)

| Metric | FP16 baseline | This pipeline | Change |
|---|---:|---:|---:|
| **UNet ship size** | 4897 MB | **1518 MB packed + 170 MB LoRA** | **-65.5 %** |
| **UNet load VRAM (after settling)** | 9963 MB | 1534 MB | **-85 %** |
| **Inference peak VRAM** (1024×1024, 4 steps) | 10767 MB | **7303 MB** | **-32 %** |
| Per-pixel MSE vs FP16 (64 prompts, RGB ∈ [0,1]) | 0 | 0.00794 | hard to see in side-by-side |
| Worst-case prompt MSE | 0 | 0.02292 | structurally faithful |
| Per-image latency (4 step) | 3365 ms | 3365 ms | unchanged (no fused INT kernel yet, see Stage 4 below) |

Full per-stage results in [BENCHMARKS.md](BENCHMARKS.md).

## Final deployment artifact

```text
models/real_quant/sdxl_lightning_weight/
  sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.safetensors   1518 MB
  sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.json          (manifest: per-layer bits/shape/etc.)
qlora_rank16_e3_lr1e4/
  lora_final.safetensors                                                170 MB

Bit assignment (recipe = "Block MP A + Deep Conv W8 + Q-LoRA"):
  Linear in {mid_block, down_blocks.2, up_blocks.0}:       W4  GPTQ   (612 layers)
  Linear in {down_blocks.1, up_blocks.1}:                  W8  RTN    (110 layers)
  Conv2d in {down_blocks.1..up_blocks.1}:                  W8  RTN    (35 layers, per-channel)
  Conv2d in {down_blocks.0, up_blocks.2, conv_in/out}:     FP16       (16 layers, latent-adjacent)
  Embeddings (time/add) + norms + biases:                  FP16
  LoRA r=16 on every quantized Linear, inference α=4:      42.6 M trainable
```

## Pipeline (7 stages, end-to-end implemented)

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. Block-level sensitivity profiling                                           │
│    For each transformer block (mid, down_blocks.1, down_blocks.2,              │
│    up_blocks.0, up_blocks.1) measure UNet output drift after quantizing all    │
│    its Linears jointly to INT4/INT8. Decides where W4 is "cheap".              │
│    → tools/profile_sdxl_lightning_block_sensitivity.py                         │
│    → tools/profile_sdxl_lightning_sensitivity.py  (per-layer variant)          │
│                                                                                │
│ 2. Mixed-precision bit assignment                                              │
│    Sensitivity says: the deepest 3 blocks (1280 ch, attention-heavy) are       │
│    least sensitive to W4 in absolute MSE terms but hold the most parameters.  │
│    → W4 there, W8 in shallower transformer blocks, FP16 in outer ResNet blocks │
│                                                                                │
│ 3. GPTQ for W4 Linears (W8 stays at RTN)                                       │
│    Self-implemented Hessian-aware PTQ: collect input-side X^T X via forward    │
│    hooks on 64 prompts × 4 timesteps = 256 calibration samples; Cholesky-      │
│    inverted Hessian; column-by-column quantize with rounding error compensated │
│    into remaining columns. W4 layer_error drops from ~1e-4 (RTN) to ~1e-5      │
│    (GPTQ) and image MSE from 0.0154 → 0.0080.                                  │
│    → mp_quant/gptq.py + tools/run_sdxl_lightning_dataset_fakequant.py          │
│                                                                                │
│ 4. Real packed storage                                                         │
│    INT4 weights packed as signed nibbles (2 per byte, offset +8) + per-group   │
│    fp16 scales; INT8 weights raw int8 + per-channel fp16 scales; FP16 layers   │
│    pass through. Manifest JSON has per-layer bits, shape, kernel hyperparams.  │
│    → mp_quant/pack_sdxl_lightning_weight.py                                    │
│                                                                                │
│ 5. PackedInt4Linear / PackedInt8Linear / PackedInt8Conv2d module replacement   │
│    Custom nn.Module subclasses keep weights packed on device throughout        │
│    inference, dequantizing inside forward(). Conv2d variant respects stride/   │
│    padding/dilation/groups from manifest. Lossless math vs in-memory fake-     │
│    quant; lower weight VRAM in practice.                                       │
│    → mp_quant/packed_linear.py + mp_quant/build_packed_unet.py                 │
│                                                                                │
│ 6. Q-LoRA recovery                                                             │
│    Rank-16 LoRA on all 722 quantized Linears, trained against pre-cached       │
│    FP16 teacher noise predictions (128 prompts × 4 timesteps = 512 samples).   │
│    Gradient checkpointing on UNet keeps training peak VRAM at 3 GB.            │
│    Inference-time α scaling exploits the trained LoRA without exposure-bias    │
│    regression (training α=16 → inference α=4 = -2 % MSE, 47/64 prompts         │
│    unchanged, only 5 regressed).                                               │
│    → mp_quant/qlora_cache_teacher.py + qlora_train.py + qlora_lora.py          │
│                                                                                │
│ 7. Stage 4 exploration — real INT kernel (documented in BENCHMARKS §8)         │
│    Characterized the off-the-shelf INT4/INT8 GEMM landscape on Blackwell       │
│    consumer (sm_120). Findings: torchao Int8 fast kernel only supports per-    │
│    channel which collapses SDXL outliers; torchao per-group fixes quality      │
│    but loses the fast path; torchao Int4 blocked by `mslk>=1.0.0` (private);   │
│    bitsandbytes INT4/INT8 quantize but kernel falls back on sm_120. Custom     │
│    Triton kernel or TensorRT 10 are the realistic next steps.                  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Why these design choices

- **Block-level mixed precision (not per-layer)** — structurally cleaner for module replacement and future kernel work; verified empirically to give ~95 % of the quality benefit of per-layer schemes at far less complexity.
- **GPTQ only for W4** — W8 RTN error is already at the fp16 noise floor (~1e-7). GPTQ on Conv would need ~530 MB Hessian per 1280-channel layer; not worth it.
- **Conv W8 RTN, per-channel** — deep blocks have small spatial activations; Conv W8 RTN was tested in the earlier handoff and added ~no MSE; saves another 303 MB on disk.
- **Block-MP A wins over more elaborate configs once GPTQ is applied** — under RTN, "FF=W4 / attn=W8 (E)" beat A by 25 % MSE; under GPTQ, the gap collapses to 8 % while A keeps a much smaller W8 footprint (104 MB vs 419 MB). The "kv-protected" config B is **dominated** by A under GPTQ.
- **Inference-time LoRA α (not just training α)** — training α=16 (scaling 1.0) regresses simple prompts at inference (classic exposure-bias problem of teacher-trajectory training). α=4 (scaling 0.25) keeps simple prompts unchanged while still improving worst-case complex prompts.

## Quick start

```bash
# 1) Generate fake-quant SDXL-Lightning UNet (block MP + GPTQ on W4)
python tools/run_sdxl_lightning_dataset_fakequant.py \
  --repo-root . --output-dir gen_test_output/quant \
  --prompt-count 64 --calib-prompt-count 64 \
  --method gptq --gptq-bits 4 --gptq-chunk-mem-gb 4.0 \
  --linear-bits 8 \
  --linear-block-bits "mid_block=4,down_blocks.2=4,up_blocks.0=4" \
  --conv-bits none --save-unet

# 2) Pack to INT4 nibble + INT8 + FP16 mixed safetensors
python mp_quant/pack_sdxl_lightning_weight.py \
  --fakequant-dir gen_test_output/quant \
  --output-dir models/real_quant/sdxl_lightning_weight \
  --conv-include "down_blocks.1,down_blocks.2,mid_block,up_blocks.0,up_blocks.1" \
  --conv-bits 8

# 3) (Optional) Q-LoRA recovery
python mp_quant/qlora_cache_teacher.py \
  --repo-root . --output-dir qlora_teacher_cache --prompt-count 128
python mp_quant/qlora_train.py \
  --packed-safetensors models/real_quant/sdxl_lightning_weight/*.safetensors \
  --packed-manifest    models/real_quant/sdxl_lightning_weight/*.json \
  --teacher-cache qlora_teacher_cache --repo-root . \
  --output-dir qlora_run --rank 16 --epochs 3 --lr 1e-4

# 4) Inference (packed UNet + LoRA, alpha=4) — 64-prompt eval vs FP16 baseline
python mp_quant/eval_packed_with_lora.py \
  --packed-safetensors models/real_quant/sdxl_lightning_weight/*.safetensors \
  --packed-manifest    models/real_quant/sdxl_lightning_weight/*.json \
  --lora qlora_run/lora_final.safetensors --lora-rank 16 --lora-alpha 4 \
  --baseline-dir gen_test_output/quant \
  --repo-root . --output-dir eval_output --prompt-count 64
```

## Repository layout

```text
mp_quant/                              SDXL-Lightning pipeline (this README's focus)
  gptq.py                                Self-implemented GPTQ (Hessian + Cholesky + column scan)
  pack_sdxl_lightning_weight.py          Packer: fake-quant -> INT4/INT8/FP16 safetensors + manifest
  packed_linear.py                       PackedInt4Linear / PackedInt8Linear / PackedInt8Conv2d
  build_packed_unet.py                   Loads packed -> assembles UNet with Packed modules
  qlora_cache_teacher.py                 Caches fp16 teacher noise_pred trajectories
  qlora_lora.py                          LoRAWrappedPackedLinear adapter
  qlora_train.py                         Q-LoRA training loop with gradient checkpointing
  eval_packed_with_lora.py               End-to-end eval (packed + LoRA vs FP16)
  benchmark_torchao_int4.py              Stage-4 exploration: torchao + BNB latency/VRAM bench
  eval_torchao_int8.py                   Stage-4 exploration: torchao Int8 quality + latency eval
  test_unpack_sdxl_lightning_weight.py   Standalone unpack-and-generate sanity tester
  make_contact_sheet.py                  3-column FP16 / Packed / LoRA visual comparison
  make_step_sweep_sheet.py               6-column step-count sweep contact sheet

tools/                                 SDXL-Lightning runner + profilers
  run_sdxl_lightning_dataset_fakequant.py        Main runner: GPTQ + RTN + image eval + --save-unet
  profile_sdxl_lightning_block_sensitivity.py    Per-block Linear sensitivity profiler
  profile_sdxl_lightning_sensitivity.py          Per-layer sensitivity profiler (legacy granularity)

models/real_quant/sdxl_lightning_weight/   Final packed artifact + manifest
qlora_rank16_e3_lr1e4/                     Trained LoRA weights
qlora_teacher_cache_128p_1024/             Cached teacher trajectories

BENCHMARKS.md                              All experimental results across every stage (RTN sweep, GPTQ, packing, LoRA α sweep, Conv W8, torchao/BNB Stage 4)
RESUME_BULLETS.md                          Pre-written resume bullets at multiple lengths + interview FAQ
REAL_QUANT_LOCAL_HANDOFF.md                Earlier handoff doc (pre-Stage 3, kept for context)

dataset/                               Prompt set used for calibration + eval (~7 MB)
logs/                                  Captured run logs from each pipeline stage
eval_summaries/                        Preserved per-config quant_stats.json / eval_summary.json
                                       from exploration runs whose PNG outputs were cleaned up
mp_quant/results/                      SDXL-Lightning per-block + per-layer sensitivity JSON

# Reference eval outputs (the 4 deployment-journey nodes + FP16 baseline)
gen_test_output/sdxl_lightning_gptq_blockmp_dataset64_1024/    Node 2: A-GPTQ fake-quant (Linear only, no pack)
gen_test_output/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8/  FP16 baseline images + quant_stats.json
qlora_eval_packed_only_64p_1024/                               Node 1: packed Stage 3 (Conv FP16)
qlora_eval_deepconvw8_packed_only/                             Node 3: packed Deep Conv W8 (no LoRA)
qlora_eval_deepconvw8_lora_a4/                                 Node 4: FINAL packed + Q-LoRA α=4
qlora_eval_contact_sheet_alpha8.png                            12-prompt visual comparison (FP16 / Packed / +LoRA)
```

## Stage 4 — what works and what doesn't on Blackwell consumer (sm_120)

We characterized the public INT4/INT8 GEMM landscape on the 5070. Full numbers in [BENCHMARKS §8](BENCHMARKS.md#8-stage-4-exploration--real-int-kernel-on-blackwell-consumer-sm_120). Headline:

- **torchao Int8 per-channel** (default): 2.34× faster than FP16, but **breaks SDXL quality entirely** (MSE 0.28, black images) — SDXL Linear has cross-attention weight outliers that one per-row scale across in_features ∈ {640, 1280, 2048, 5120} cannot accommodate.
- **torchao Int8 per-group=128**: fixes quality (MSE 0.0012) but the fast Int8 kernel rejects per-group and falls back to a slow path (2.5× slower than FP16).
- **torchao Int4**: blocked — depends on `mslk>=1.0.0`, a private/internal package not on PyPI.
- **bitsandbytes NF4 / INT8**: quantizes correctly (uint8 storage verified) but the CUDA INT kernels for sm_120 haven't fully landed yet — compute falls back to dequant→fp16 paths.

**Implication**: our Stage 3 `PackedLinear` is, today, **the practical equivalent of what an off-the-shelf INT4 library would do on this hardware** (dequant→fp16→GEMM), with the bit assignment we want and no closed-source dependency. The next real speed step requires either (a) a custom Triton W4×fp16 fused kernel or (b) **TensorRT 10** export with its Blackwell-aware INT kernels — that's the planned v2.

## Limitations & open work

| Item | Why deferred / planned for v2 |
|---|---|
| Real INT4 × INT8 fused GEMM kernel | Requires custom Triton (3-5 days) or TensorRT 10 export (2-4 days). Without it, Stage 3 gives only the VRAM win, not the latency win. |
| Activation INT8 (W4A8) | Fake-quant validated (MSE +0.0013 on top of weight-only). Real benefit needs the kernel work above. |
| Student-trajectory Q-LoRA | Would address the exposure-bias regression observed at training-strength α. Estimated 5-15 % additional MSE improvement. |
| Image-space distillation loss | Decode VAE for both teacher and student during training. Sharper quality target but ~3× training cost. |
| 8-step Lightning variant | Whole pipeline currently locked to the 4-step distilled checkpoint. 8-step would require re-running everything. |

## Hardware tested

```text
GPU:           NVIDIA RTX 5070 12 GB (sm_120, Blackwell consumer)
CPU/RAM:       Windows 11 + 64 GB RAM
PyTorch:       2.12.0.dev (CUDA 12.8)
diffusers:     latest
torchao:       0.17.0
bitsandbytes:  0.49.2
```

VRAM numbers above are for this exact GPU at 1024×1024, 4 inference steps. The pipeline targets **7.3 GB peak** — fits RTX 4060+ / 4070+ / 5070+ desktops and Jetson Orin 16 GB+ comfortably; tight on 8 GB consumer GPUs; not a fit for mobile-class devices yet (would need activation quant + resolution drop + text-encoder CPU offload).

## Acknowledgments

- **SDXL-Lightning** ([Lin et al. 2024](https://arxiv.org/abs/2402.13929)) — the 4-step distilled base model.
- **GPTQ** ([Frantar et al. 2023](https://arxiv.org/abs/2210.17323)) — Hessian-based PTQ algorithm reimplemented here.
- **QLoRA** ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) — parameter-efficient quantized fine-tuning, adapted from LLMs to diffusion.
- **torchao**, **bitsandbytes** — INT-quantization reference implementations used in the Stage 4 baseline study.

Maintainer: Chen He ([`SeanHe727`](https://github.com/SeanHe727) on GitHub, [`ChenHe727`](https://huggingface.co/ChenHe727) on Hugging Face).
