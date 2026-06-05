# EdgeDiffusion — Full Detail Work Log (Agent-Readable)

This document is the canonical, agent-readable detail dump for the EdgeDiffusion
project. It covers every phase, every script, every experimental result (positive
and negative), the failure modes encountered, and the reproduction procedure for
each artifact. README.md is the brief human-facing companion; this file is for
agents continuing or auditing the work.

Contents:

1. [Repository layout (current)](#1-repository-layout-current)
2. [Phase 1 — Fake-quant + GPTQ on PyTorch](#2-phase-1--fake-quant--gptq-on-pytorch)
3. [Phase 2 — Real packed deployment + Q-LoRA](#3-phase-2--real-packed-deployment--q-lora)
4. [Phase 3 — ONNX export + TensorRT engine](#4-phase-3--onnx-export--tensorrt-engine)
5. [Phase 3 EC2 detail — Sprints 1, 2, 2b, 2c, 2d, 3](#5-phase-3-ec2-detail--sprints-1-2-2b-2c-2d-3)
6. [Negative results catalog](#6-negative-results-catalog)
7. [Hardware and environment notes](#7-hardware-and-environment-notes)
8. [Reproduction recipes per phase](#8-reproduction-recipes-per-phase)
9. [Handoff notes for the next agent](#9-handoff-notes-for-the-next-agent)

---

## 1. Repository layout (current)

```text
EdgeDiffusion/
├── README.md                              brief human-facing
├── BENCHMARKS.md                          all benchmark numbers, source of truth
├── EC2_results.md                         (this file)
│
├── quantization/main/
│   ├── fakequant/
│   │   ├── gptq.py                        self-implemented GPTQ
│   │   └── run_fakequant.py               Phase 1 main runner
│   ├── packed/
│   │   ├── pack_weight.py                 fake-quant dir → packed safetensors
│   │   ├── pack_linear.py                 PackedInt4/Int8 Linear + Conv2d modules
│   │   ├── build_packed_unet.py           packed safetensors → UNet
│   │   ├── qlora_cache_teacher.py         FP16 teacher noise pred cache builder
│   │   ├── qlora_wrapper.py               LoRA wrapper for PackedLinear
│   │   ├── qlora_train.py                 Q-LoRA training loop
│   │   ├── eval_packed_with_qlora.py      Phase 2 end-to-end eval
│   │   ├── qlora_eval_deepconvw8_lora_a4/  3 sample PNGs (Phase 2 row of showcase)
│   │   ├── qlora_rank16_e3_lr1e4/         LoRA weights + summary
│   │   └── qlora_teacher_cache_128p_1024/ teacher cache (~512 samples)
│   └── TRT/
│       ├── qlora_merge.py                 packed + LoRA → merged FP16 UNet
│       ├── qlora_merge_eval.py            merged FP16 eval
│       ├── export_onnx.py                 PyTorch FP16 UNet → fp32 ONNX
│       ├── export_onnx_clear.py           dead-initializer cleanup
│       ├── export_onnx_verification.py    ONNX numerical validation vs PyTorch
│       ├── qdq_marking.py                 insert Q/DQ from TRT calib cache
│       ├── build_trt_engine.py            TRT engine build (implicit calib + --qdq)
│       └── trt_unet_wrapper.py            TRT engine → diffusers UNet adapter
│
├── quantization/archived/                 7 archive_*.py files: torchao/BNB survey,
│                                          percentile calibration, TRT diagnostics,
│                                          contact-sheet helpers, unpack sanity test
│
├── evals/
│   ├── onnx_eval.py                       ORT end-to-end eval
│   ├── trt_eval.py                        TRT engine end-to-end eval
│   └── eval_output/
│       ├── showcase_4row.jpg              4-row × 6-col comparison sheet
│       ├── reduction_quality_chart.png    Phase 2 size/VRAM reduction
│       ├── size_vram_chart.png            Phase 2 checkpoints chart
│       ├── images/
│       │   └── sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8/
│       │       └── fp16_*.png             64 FP16 baseline images (CANONICAL)
│       ├── records/                       14 eval_summary.json (BENCHMARKS source)
│       └── trt_b1c_64p/                   Phase 3 deliverable benchmark notes
│
├── utils/
│   ├── profile_sdxl_lightning_block_sensitivity.py  Phase 1 block sensitivity
│   ├── download_coco_subset.py            COCO prompt set
│   ├── evaluate_fid.py                    FID metric (unused in shipping)
│   ├── gen_custom_3backends.py            EC2: 3-backend custom prompt gen
│   ├── gen_custom_phase2.py               local: Phase 2 row for custom prompts
│   ├── showcase.py                        4-row × 6-col image builder
│   └── make_readme_assets.py              Phase 2 charts (size_vram, reduction_quality)
│
├── models/
│   ├── real_quant/sdxl_lightning_weight/
│   │   ├── sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.safetensors  (1.5 GB)
│   │   └── sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.json         (manifest)
│   └── onnx/
│       ├── unet_fp32.onnx                                  (graph, 3.8 MB)
│       ├── unet_fp32.onnx_data                             (9.6 GB external weights)
│       ├── sdxl_lightning_unet_fp16_merged.safetensors     (4.8 GB; pre-ONNX merge)
│       └── unet_fp32_export_summary.json
│
├── dataset/*.txt                          COCO prompt subset (text only)
└── logs/                                  run logs from Phase 1-2
```

Repo branches:

- `main` — public clean branch
- `Sean` — private dev / archive with TRT engine references, EC2 session records,
  intermediate ONNX variants, and tracked sample PNGs needed by `showcase.py`

---

## 2. Phase 1 — Fake-quant + GPTQ on PyTorch

### 2.1 Block sensitivity profiling

Script: `utils/profile_sdxl_lightning_block_sensitivity.py`.

For each transformer block (`mid_block`, `down_blocks.1`, `down_blocks.2`,
`up_blocks.0`, `up_blocks.1`) measure UNet output drift after jointly quantizing
all Linears in that block to INT4 or INT8.

Finding: the three deepest blocks (`mid_block`, `down_blocks.2`, `up_blocks.0`)
hold ≈ 95 % of UNet parameters but are *least* sensitive to W4 in absolute MSE
terms. This drove the block-MP bit assignment used in Phase 2 and Phase 3.

### 2.2 Self-implemented GPTQ

Script: `quantization/main/fakequant/gptq.py` (~260 LOC).

Pipeline:

1. Forward-hook the target Linear; accumulate input-side `X^T X` (Hessian)
   across calibration data (64 prompts × 4 timesteps = 256 samples).
2. Add diagonal damping to ensure positive-definiteness.
3. Cholesky-invert; obtain inverse Hessian `H^{-1}`.
4. Column-by-column quantization with rounding error compensated into the
   remaining columns via `H^{-1}` (standard GPTQ update rule).

Empirical: W4 layer error drops 1e-4 (RTN) → 1e-5 (GPTQ); image MSE
0.0154 → 0.0080 (in-memory fake-quant).

### 2.3 Block-MP bit assignment ablation

Configs tested (see `evals/eval_output/records/` for raw JSONs):

| Config | Recipe | RTN MSE | GPTQ MSE |
| --- | --- | ---: | ---: |
| A (chosen) | W4 in deep blocks, W8 in shallower transformer blocks | 0.0154 | 0.0080 |
| B | A + protect attention KV (kv stays FP16) | 0.0136 | 0.0086 |
| C | A + protect attention all (Q/K/V/out stay FP16) | 0.0142 | n/a |
| E | FFN W4 / attention W8 | 0.0115 | 0.0074 |
| Linear W4 only (Conv FP16) | All Linears W4 | 0.0181 | n/a |
| Linear W4/W8 (block-MP) | W4 deep, W8 shallow Linear-only | 0.0157 | n/a |

Under GPTQ, A and E differ by only 8 % MSE while A keeps a much smaller W8
footprint (104 MB vs 419 MB), so A wins on size+quality jointly. Under RTN
alone, E beats A by 25 % MSE, but GPTQ closes that gap.

### 2.4 Findings — what to quantize and what NOT to

- **Linears (Q/K/V/out + FFN)**: safely W4 in deep blocks, W8 in shallower.
- **Conv2d**: W8 RTN in deep blocks (small spatial activations); FP16 in
  outermost blocks adjacent to the latent (`down_blocks.0`, `up_blocks.2`,
  `conv_in`, `conv_out`).
- **Embeddings, norms, biases**: keep FP16.
- **Attention BMM** (`Q @ K^T`, `attn @ V`): never quantized in Phase 2 — both
  operands are activations, not weights.

### 2.5 Phase 1 deliverable

Phase 1 has no shipping artifact; it produces a fake-quant UNet held in memory
plus the calibration sample → scale tables that feed Phase 2's packer.

---

## 3. Phase 2 — Real packed deployment + Q-LoRA

### 3.1 Packed safetensors format

Script: `quantization/main/packed/pack_weight.py`.

Layout inside the `.safetensors`:

- **INT4 weights** → packed as signed nibbles, 2 weights per byte, with +8
  offset; per-group fp16 scales (group size 128).
- **INT8 weights** → raw int8; per-channel fp16 scales.
- **FP16 weights** → stored directly (norms, biases, latent-adjacent conv,
  embeddings).

Manifest JSON records per-layer bits, shape, kernel hyperparameters (stride,
padding, dilation, groups for Conv), and the original module name.

Final deliverable: `models/real_quant/sdxl_lightning_weight/sdxl_lightning_4step_blockmp_a_gptq_deepconvw8_packed.{safetensors,json}` — 1.5 GB total.

### 3.2 PackedLinear / PackedConv2d modules

Script: `quantization/main/packed/pack_linear.py`.

Three custom `nn.Module` subclasses:

- `PackedInt4Linear` — holds nibble buffer + per-group scale; `forward()`
  unpacks to fp16 on the fly and calls `F.linear`.
- `PackedInt8Linear` — holds int8 buffer + per-channel scale; same on-the-fly
  dequant pattern.
- `PackedInt8Conv2d` — wraps `F.conv2d` with the same int8-buffer pattern;
  respects stride/padding/dilation/groups from manifest.

The packed buffers stay in their compressed form throughout inference — only
the dequantized fp16 tile lives transiently inside `forward()`.

### 3.3 Build packed UNet from packed safetensors

Script: `quantization/main/packed/build_packed_unet.py`. Loads the packed
state-dict + manifest, instantiates a SDXL UNet, replaces every `nn.Linear` /
`nn.Conv2d` named in the manifest with the corresponding `Packed*` module.

### 3.4 Q-LoRA recovery

Cache teacher noise predictions: `quantization/main/packed/qlora_cache_teacher.py`.
128 prompts × 4 timesteps = 512 samples. Cache location:
`quantization/main/packed/qlora_teacher_cache_128p_1024/`.

LoRA wrapper: `quantization/main/packed/qlora_wrapper.py`. Wraps every
`PackedInt4Linear` / `PackedInt8Linear` with rank-r LoRA. The packed weights
stay frozen; only the LoRA `A` and `B` matrices are trainable.

Training: `quantization/main/packed/qlora_train.py`. Rank 16, 3 epochs,
lr 1e-4. Trains against pre-cached teacher noise predictions (MSE loss on
ε prediction). Gradient checkpointing on the UNet keeps training peak VRAM at
3 GB.

LoRA artifacts: `quantization/main/packed/qlora_rank16_e3_lr1e4/`:

- `lora_final.safetensors` (170 MB)
- `lora_step512.safetensors` (170 MB; earlier checkpoint, kept for repro)
- `qlora_train_summary.json`

### 3.5 Exposure-bias mitigation: inference α=4

Training α = 16 minimizes ε MSE against the teacher trajectory but regresses
on simple prompts at inference (classic exposure-bias artifact). Inference α =
4 (0.25× LoRA scaling) keeps simple prompts unchanged while still improving
worst-case complex ones. α sweep in BENCHMARKS §6 LoRA α table.

### 3.6 Phase 2 evaluation

Script: `quantization/main/packed/eval_packed_with_qlora.py`.

| Metric | FP16 baseline | Phase 2 deliverable | Change |
| --- | ---: | ---: | ---: |
| Ship size | 4897 MB | 1518 + 170 MB | -65.5 % |
| UNet load VRAM | 9963 MB | 1534 MB | -85 % |
| Inference peak VRAM | 10767 MB | 7303 MB | -32 % |
| MSE vs FP16 (64 prompts) | 0 | **0.00794** | (mean) |
| MSE worst case | 0 | 0.02292 | |
| Latency (RTX 5070, 4 step) | 3365 ms | 3365 ms | unchanged |

Latency is unchanged because `PackedLinear.forward()` dequantizes to fp16 then
calls cuBLAS — same compute path as FP16. Phase 2 wins on size + VRAM, not
latency. Phase 3 addresses latency.

### 3.7 Phase 2 deliverable summary

The PyTorch deliverable consists of:

1. `models/real_quant/sdxl_lightning_weight/*.safetensors` + `*.json`
2. `quantization/main/packed/qlora_rank16_e3_lr1e4/lora_final.safetensors`
3. Inference: load packed UNet via `build_packed_unet.py`, attach LoRA via
   `qlora_wrapper.py` (α = 4), run normal diffusers pipeline.

---

## 4. Phase 3 — ONNX export + TensorRT engine

### 4.1 Merge LoRA into FP16 UNet (qlora_merge)

Script: `quantization/main/TRT/qlora_merge.py`.

Combines: SDXL-Lightning fp16 UNet base + `lora_final.safetensors` (α=4
scaling) → a single merged FP16 UNet state-dict. Output:
`models/onnx/sdxl_lightning_unet_fp16_merged.safetensors` (4.8 GB).

`qlora_merge_eval.py` verifies the merged FP16 model reproduces the Phase 2
quality (MSE within 0.001 of `packed + LoRA α=4`).

### 4.2 ONNX export (PyTorch → fp32 ONNX)

Script: `quantization/main/TRT/export_onnx.py`. Uses the legacy
`torch.onnx.export` path (`dynamo=False`), opset 17, external-data format.

Workarounds:

- `time_proj` was exported in fp32 to avoid fp16 overflow on certain
  timestep encodings.
- Initial export wrote ~1491 sharded `.onnx_data` files; `export_onnx_clear.py`
  consolidates these into a single `unet_fp32.onnx_data` (9.6 GB).

`export_onnx_verification.py` runs 4 calibration samples through both PyTorch
fp16 and the exported fp32 ONNX (via ORT CPU EP) and reports per-pixel MSE.
Validated **MSE 0.00573 vs FP16 PyTorch baseline (4 prompts, RTX 5070)** —
within numerical-noise tolerance.

Output artifacts in `models/onnx/`:

- `unet_fp32.onnx` (3.8 MB graph)
- `unet_fp32.onnx_data` (9.6 GB external weights)
- `unet_fp32_export_summary.json`

### 4.3 TRT entropy calibration → calib.cache

Script: `quantization/main/TRT/build_trt_engine.py` (without `--qdq`).

Builds a TRT INT8 engine from `unet_fp32.onnx` using
`IInt8EntropyCalibrator2` over 64 teacher-cache samples. The engine itself is
the Sprint 2 implicit engine (see §5.3). The byproduct is
`unet_int8_fp16.calib.cache` — 745 KB, 9926 tensor scales — which is the
foundation for the Phase 3 explicit-QDQ path.

### 4.4 Q/DQ insertion (the recipe that wins)

Script: `quantization/main/TRT/qdq_marking.py`.

Flag matrix:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--op-types` | `MatMul,Gemm,Conv` | Which op types to consider |
| `--matmul-mode` | `all` | `linear-only` skips two-activation BMM; `bmm-only` does only BMM |
| `--no-output-qdq` | off | Skip Q/DQ on op outputs (standard QDQ pattern) |
| `--quantize-weights` | off | Insert Q/DQ on weight initializers (per-tensor max/127) |
| `--no-activation-qdq` | off | Skip activation Q/DQ (weight-only mode for ablation) |
| `--include-substrings` | "" | Filter nodes by name substring |
| `--exclude-substrings` | "" | Skip nodes by name substring |
| `--activation-scale-multiplier` | 1.0 | Multiply every activation scale (widening sweep) |

**The winning recipe (B1-c)**:

```text
--op-types Conv,MatMul --matmul-mode linear-only --no-output-qdq --quantize-weights
```

That is: quantize Conv + linear-style MatMul (skip attention BMM and Gemm),
input-only Q/DQ on activations, *plus* per-tensor weight Q/DQ on the small set
of weights that the script picks up as strict initializers (102 of them).

### 4.5 TRT engine build (B1-c)

Script: `quantization/main/TRT/build_trt_engine.py --qdq`.

Adds `config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED` so
`EngineInspector` can return per-layer precision after build (without it the
inspector only returns layer names).

B1-c engine build metrics (logged by TRT):

| Metric | Value |
| --- | ---: |
| Engine file size | 2.66 GB |
| Weights memory | 2.62 GB |
| Activation memory | 178 MB |
| Build time | 25.6 min |
| TRT CPU peak | 1380 MiB |
| TRT GPU peak | 2502 MiB |

The activation memory of 178 MB ≈ implicit engine's 172 MB confirms TRT is
fusing the Q/DQ nodes into INT8 MatMul kernels (Sprint 2b failed variants had
474 MB, which was the smoking gun for missed fusion). The weights memory of
2.62 GB ≈ 54 % of FP16 baseline weights confirms majority INT8.

### 4.6 TRT engine evaluation

Script: `evals/trt_eval.py`.

Final 64-prompt benchmark vs FP16 baseline:

| Metric | Value | Gate | Status |
| --- | ---: | ---: | --- |
| MSE mean | **0.0369** | ≤ 0.02 | exceeds gate but in same ballpark as Phase 2 (0.008) and far better than failed variants (0.166) |
| MSE max | 0.0898 | — | hard prompts (e.g. portrait close-ups, busy interiors) |
| Latency | **0.98 s / image** | — | steady-state on A10G |
| Peak VRAM (full pipeline) | 19.6 GB | — | text encoders + VAE on PyTorch + TRT UNet |

---

## 5. Phase 3 EC2 detail — Sprints 1, 2, 2b, 2c, 2d, 3

### 5.1 EC2 environment

- Instance: AWS EC2 g5.2xlarge.
- GPU: NVIDIA A10G 24 GB.
- vCPU: 8 (AWS account capped at 8 vCPU — cannot upgrade to g5.4xlarge).
- RAM: 32 GB.
- Storage: persistent EBS volume `vol-080fb2f650d1f01de` mounted at
  `/opt/ebs`.
- PyTorch env: `/opt/pytorch/bin/activate`.
- Extra libs (TensorRT 10, ORT 1.24): `/opt/ebs/py_extra` and
  `/opt/ebs/py_extra/tensorrt_libs`.
- Required env exports for any TRT/ORT script:

```bash
source /opt/pytorch/bin/activate
export TMPDIR=/opt/dlami/nvme/.tmp
export HF_HOME=/opt/dlami/nvme/.hf_cache
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/opt/ebs/py_extra:.
export LD_LIBRARY_PATH=/opt/ebs/py_extra/tensorrt_libs:$LD_LIBRARY_PATH
```

ORT CUDA EP failed on this instance (`libcublasLt.so.12: cannot open shared object`)
because only CUDA 13 was present and ORT 1.24 needs CUDA 12. All ORT
diagnostic runs fall back to CPU EP — slow (~100 s / image) but mathematically
correct. TRT itself is unaffected.

### 5.2 Sprint 1 — NVIDIA ModelOpt INT8 QDQ → SKIPPED

ModelOpt 0.44 preprocess of the 9.6 GB ONNX OOM-killed; virtual memory grew
to ~78 GB. Even 32 GB RAM + 32 GB EBS swap was insufficient. Path abandoned in
favour of TRT-native calibration.

### 5.3 Sprint 2 — TRT implicit calibration engine

Built `unet_int8_fp16.engine` from `unet_fp32.onnx` using
`IInt8EntropyCalibrator2` with 64 teacher cache samples, `FP16 | INT8` flags,
12 GB workspace, static shapes.

Result: MSE **0.01215** on 4 prompts (later confirmed 0.01354 / 0.03201 on 64
prompts), latency 1.23 s / image (steady-state 1.08 s on 64-prompt mean),
engine **4.6 GB**, weights memory 4.82 GB.

Interpretation: the weights memory ≈ raw FP16 UNet's 4.9 GB means TRT only
quantized a handful of MatMuls. ~30–40 %+ of MatMul weights stayed FP16
because TRT emitted "Missing scale and zero-point" warnings for
LayerNorm-adjacent tensors and fell back. **This engine looks fine on paper
but is effectively "FP16 with a touch of INT8", not a real INT8 deliverable.**

### 5.4 Sprint 2b — Explicit QDQ ONNX, three failed variants

Attempt to build a smaller engine by inserting Q/DQ from the Sprint 2 calib
cache directly into the ONNX graph.

| Variant | Recipe | Engine size | MSE | Status |
| --- | --- | ---: | ---: | --- |
| v1 | per-tensor weight QDQ + activation QDQ + `PREFER_PRECISION_CONSTRAINTS` | 2.59 GB | 0.187 | output binding bug + bad quality |
| v2 | per-tensor weight QDQ + activation QDQ (no `PREFER_PRECISION_CONSTRAINTS`) | 2.59 GB | 0.166 | output binding fixed, still bad |
| v3 | activation-only QDQ (no weight QDQ) | 2.95 GB | 0.166 | still bad |

Bugs found and fixed:

1. **Output binding bug** — the QDQ insertion originally also added Q/DQ on
   the final graph output `noise_pred`, renaming it to a synthetic DQ name.
   The wrapper had `noise_pred` hardcoded → TRT couldn't find the output
   buffer → returned garbage. Fixed by skipping graph-level outputs in the
   QDQ pass and discovering the output name dynamically via
   `engine.get_tensor_name()`.
2. **Per-tensor weight scale collapse** — `add_embedding.linear_1.weight`
   has max 1.39, p99 0.08, ratio 17×. Per-tensor scale `= max(|W|) / 127`
   maps 99 % of values to ±1 in int8. Removing weight QDQ entirely (v3)
   did not fix the MSE, implying the quality problem was elsewhere.

The diagnostic that broke the case open: explicit-QDQ engine had
**activation memory 474 MB** vs implicit's 172 MB. That's TRT executing the
Q/DQ pairs as real precision-loss operations, not fusing them into INT8
kernels.

### 5.5 Sprint 2c — ORT-only ablation isolating the QDQ insertion bug

Used ORT (CPU EP) to evaluate intermediate QDQ ONNXs directly, separating
QDQ-graph bugs from TRT fusion bugs. 4 prompts per variant.

| Variant | Q/DQ pairs | MSE | Δ vs fp32 | Per-op damage |
| --- | ---: | ---: | ---: | ---: |
| FP32 ONNX (no QDQ, reference) | 0 | 0.006 | — | — |
| Conv only (51 ops) | 51 | 0.011 | +0.005 | 0.0001 |
| Conv + Gemm (51 + 21 ops) | 72 | 0.165 | +0.160 | Gemm: 0.0073 ⚠ |
| MatMul only (all 1002, mixed) | 1002 | 0.051 | +0.045 | — |
| **B1: Conv + Linear-style MatMul** (skip BMM + Gemm) | 773 | 0.028 | +0.022 | 0.000028 |
| B2: BMM only (280 attention BMM) | 280 | 0.044 | +0.038 | 0.00014 |
| B1-a: B1 weight-only QDQ (102 strict initializers) | 102 | 0.0078 | +0.002 | — |
| B1-b: B1 activation-only QDQ (current B1) | 1495 | 0.0284 | +0.022 | — |
| **B1-c: B1 activation + weight QDQ (chosen recipe)** | 1597 | 0.029 | +0.023 | identical to B1-b |

Findings:

- Gemm in `time_embedding` / `add_embedding` paths is catastrophic (73× the
  per-op damage of Conv). These outputs are broadcast-added to every spatial
  activation, so one bad scale poisons the whole image.
- Attention BMM (two-activation MatMul, never quantized in Phase 2) is the
  second-largest contributor.
- Linear-style MatMul + Conv is the safe set.
- Per-tensor weight Q/DQ on the 102 strict initializers is essentially
  harmless on its own (MSE 0.008) — the 17× outlier concern from Sprint 2b is
  not really triggered for *these particular* MatMul-weight tensors. (The
  17× outlier weight, `add_embedding.linear_1.weight`, lives in the Gemm
  path which we now skip entirely.)
- Activation QDQ contributes essentially all of the +0.022 delta. Per-channel
  weight QDQ is NOT the lever for further quality improvement — the lever is
  per-tensor adaptive activation scales, which ONNX QDQ spec does not support
  (activation scales must be per-tensor static).

### 5.6 Sprint 2d — B1-c engine build (the winner)

Re-built with the B1-c QDQ ONNX, `--qdq` flag, `ProfilingVerbosity.DETAILED`.

| Metric | Sprint 2 implicit | Sprint 2b v3 (broken) | **Sprint 2d B1-c** |
| --- | ---: | ---: | ---: |
| Engine size | 4.6 GB | 2.95 GB | **2.66 GB** |
| Weights memory | 4.82 GB | 2.73 GB | **2.62 GB** |
| Activation memory | 172 MB | 474 MB | **178 MB** |
| MSE (4-prompt sample) | 0.012 | 0.166 | **0.027** |
| MSE (64-prompt benchmark) | 0.014 | n/a | **0.037** |
| Latency | 1.23 s | 1.22 s | **0.98 s (steady)** |
| Build time | ~30 min | ~30 min | 25.6 min |

The 178 MB activation memory at near-implicit levels is the decisive signal
that TRT is now fusing Q/DQ into INT8 kernels (Sprint 2b v3 was the
counter-example with 474 MB). The 2.62 GB weights memory, vs implicit's
4.82 GB, confirms majority INT8 weight kernels.

### 5.7 Sprint 3 — Quality-tuning attempts (negative results)

Three approaches tried to push MSE from 0.027 down to the 0.02 quality gate.
None succeeded.

#### 5.7.1 Block-skip probing (P1)

Only quantize the 3 deepest blocks (`mid_block`, `down_blocks.2`,
`up_blocks.0`), skip the shallower `down_blocks.1` + `up_blocks.1`. Result on
4 prompts: MSE 0.026 (vs B1-c's 0.028). The Phase 2 PyTorch "shallow is more
sensitive" intuition does not transfer to INT8 — quantization damage is
fairly uniform across blocks here. Skipping 166 nodes (17 %) only saved
0.003 MSE (10 %). Not worth a fresh TRT engine build.

#### 5.7.2 Activation scale multiplier sweep

Adds `--activation-scale-multiplier` to `qdq_marking.py` and sweeps:

| Multiplier | 2-prompt MSE mean | 4-prompt MSE mean | 4-prompt MSE max |
| ---: | ---: | ---: | ---: |
| 0.8 | 0.028 | n/a | n/a |
| 1.0 (baseline) | 0.0186 | 0.0294 | 0.0438 |
| 1.1 | 0.0173 | n/a | n/a |
| 1.2 | 0.0148 | n/a | n/a |
| **1.4** | n/a | **0.0279** | **0.0572** |
| 1.5 | 0.0137 | 0.0295 | 0.0624 |
| 1.6 | n/a | 0.0311 | 0.0677 |
| 1.8 | 0.021 | n/a | n/a |
| 2.0 | 0.033 | n/a | n/a |
| 2.5 | 0.024 | n/a | n/a |
| 3.0 | 0.018 | n/a | n/a |

2-prompt sweep suggested 1.5× was optimal. 4-prompt sweep revealed the truth:
widening scales helps *easy* prompts but hurts *hard* ones (MSE max grows
monotonically with multiplier). No global scale operating point works
across all prompts. The 2-prompt subset was happenstance easier.

#### 5.7.3 Percentile (p99.9) activation calibration

Tool: `quantization/archived/archive_percentile_calibrate.py`.

Captures activation distributions by chunked ORT-CPU forward over the fp32
ONNX (1219 tensors recalibrated using 4 calibration samples, ~25 min
wall-time). Writes a TRT-compatible calib cache where every per-tensor scale
is `p99.9 / 127` instead of TRT's KL-optimal value.

Result: MSE **0.052** — *worse* than the original entropy calib (0.029). The
p99.9 scales are wider than entropy's KL-optimal scales (more outlier
headroom, more in-range quantization noise) and exhibit the same
hard-prompt regression as the 1.5–2× scale multiplier.

Decision: stick with the original entropy-calibrated 1.0× multiplier as the
B1-c operating point.

---

## 6. Negative results catalog

| Approach | Outcome | Lesson |
| --- | --- | --- |
| ORT `quantize_static` | Killed by BFCArena fragmentation in ORT 1.23, OOM even on 32 GB RAM | Switching to a tool with a different memory architecture is faster than tuning the fragmenting one |
| NVIDIA ModelOpt 0.44 | Preprocess wants ~78 GB virtual memory; 32 GB RAM + 32 GB swap insufficient | Bleeding-edge quantization libraries don't budget for SDXL-sized graphs on consumer/cloud single-node hardware |
| TRT implicit calibration | Builds a real engine but mostly FP16; LayerNorm-adjacent weights skipped | Implicit calibration in TRT is fine for well-behaved CNNs, weak for transformer attention — explicit QDQ is the right path but needs both weight AND activation Q/DQ to trigger fused INT8 kernels |
| Explicit QDQ without weight QDQ | TRT executes Q/DQ as real ops (activation memory 474 MB vs implicit's 172 MB) → MSE 0.166 | TRT 10 needs both sides of the Q/DQ pair for INT8 fusion |
| Quantizing Gemm in embedding paths | One per-tensor scale collapses 99 % of weight values to ±1, broadcast-poisons every pixel | Embedding-broadcast operators are special; skip them |
| Quantizing attention BMM | Two activations multiplied; never touched in Phase 2 — adds 0.038 MSE in Phase 3 | Phase 2 wisdom transfers: BMM stays FP16 |
| Block-skip probing on shallow blocks | Only -0.003 MSE | Phase 2's "shallow is more sensitive" rule does not carry to INT8 |
| Global activation scale multiplier sweep | Easy prompts improve, hard prompts regress | Per-tensor symmetric INT8 cannot adapt per-prompt; one global multiplier always trades one set of prompts off against another |
| Percentile (p99.9) activation calibration | MSE 0.052, worse than entropy | Wider scales preserve outliers but accumulate more in-range noise on hard prompts |
| torchao Int8WeightOnly per-channel | Black images, MSE 0.28 | SDXL attention has weight outliers across `in_features ∈ {640,1280,2048,5120}` that one per-row scale cannot accommodate |
| torchao Int8WeightOnly per-group=128 | Quality fine, but the fast cuBLAS path rejects per-group → falls back to a 2.5× slower kernel | torchao's fast Int8 path is per-channel-only by design |
| torchao Int4 | Blocked: depends on `mslk>=1.0.0`, a private/internal package not on PyPI | Off-the-shelf INT4 isn't available on this hardware combo yet |
| bitsandbytes NF4 / INT8 | Quantizes correctly (uint8 storage verified) but compute falls back to dequant→fp16 on sm_120 | Native CUDA INT kernels for Blackwell consumer haven't fully landed |

---

## 7. Hardware and environment notes

### 7.1 Local (Windows 11)

- GPU: NVIDIA RTX 5070 12 GB, sm_120 Blackwell.
- CPU: Windows 11, 64 GB RAM.
- Python env with CUDA: `D:\chenh\Anaconda\envs\sd_native\python.exe`
  (torch 2.12.0.dev + CUDA 12.8).
- Phase 2 packed UNet + Q-LoRA train + Phase 2 eval all run here.
- Phase 3 ONNX export runs here.
- Phase 3 TRT engine build does NOT run here (TRT 10 + sm_120 stack would need
  separate install; we use EC2 A10G instead).

### 7.2 EC2 (Ubuntu, AWS g5.2xlarge)

- GPU: NVIDIA A10G 24 GB.
- vCPU 8, RAM 32 GB, AWS account capped at 8 vCPU.
- Persistent EBS at `/opt/ebs` — survives instance stop/start.
- SSH: `ssh -i <key.pem> ubuntu@<ip>` (public IP rotates on stop/start).
- ORT CUDA EP unavailable (CUDA 12 vs CUDA 13 mismatch); ORT runs go through
  CPU EP (~100 s / image).
- TRT 10 works fine; engine build takes ~25 min for B1-c.

### 7.3 Targets

- Phase 2 packed UNet peak VRAM 7.3 GB fits RTX 4060+ / 4070+ / 5070+ and
  Jetson Orin 16 GB.
- Phase 3 TRT engine targets A10G-class data-center GPUs (and Blackwell
  consumer with TRT 10) with native INT8 kernels.

---

## 8. Reproduction recipes per phase

### 8.1 Phase 1 (Fake-quant + sensitivity)

```bash
# Block sensitivity (decides which blocks tolerate W4)
python utils/profile_sdxl_lightning_block_sensitivity.py

# Fake-quant runner (saves UNet + per-prompt PNG + eval_summary)
python quantization/main/fakequant/run_fakequant.py \
  --repo-root . --output-dir evals/eval_output/images/fakequant \
  --method gptq --gptq-bits 4 --linear-bits 8 \
  --linear-block-bits "mid_block=4,down_blocks.2=4,up_blocks.0=4" \
  --conv-bits 8 --save-unet
```

### 8.2 Phase 2 (Real packed + Q-LoRA)

```bash
# Pack the fake-quant UNet into INT4-nibble + INT8 + FP16 safetensors
python quantization/main/packed/pack_weight.py \
  --fakequant-dir evals/eval_output/images/fakequant \
  --output-dir   models/real_quant/sdxl_lightning_weight \
  --conv-include "down_blocks.1,down_blocks.2,mid_block,up_blocks.0,up_blocks.1" \
  --conv-bits 8

# Cache FP16 teacher noise predictions
python quantization/main/packed/qlora_cache_teacher.py \
  --repo-root . \
  --output-dir quantization/main/packed/qlora_teacher_cache_128p_1024

# Q-LoRA training
python quantization/main/packed/qlora_train.py \
  --packed-safetensors models/real_quant/sdxl_lightning_weight/*.safetensors \
  --packed-manifest    models/real_quant/sdxl_lightning_weight/*.json \
  --teacher-cache quantization/main/packed/qlora_teacher_cache_128p_1024 \
  --output-dir quantization/main/packed/qlora_rank16_e3_lr1e4 \
  --rank 16 --epochs 3 --lr 1e-4

# Phase 2 eval at α=4
python quantization/main/packed/eval_packed_with_qlora.py \
  --packed-safetensors models/real_quant/sdxl_lightning_weight/*.safetensors \
  --packed-manifest    models/real_quant/sdxl_lightning_weight/*.json \
  --lora quantization/main/packed/qlora_rank16_e3_lr1e4/lora_final.safetensors \
  --lora-alpha 4 --repo-root . \
  --baseline-dir evals/eval_output/images/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8 \
  --output-dir   evals/eval_output/images/phase2_lora_a4_64p
```

### 8.3 Phase 3 (ONNX + TRT engine)

On local (RTX 5070) for ONNX prep:

```bash
# Merge LoRA into FP16 UNet
python quantization/main/TRT/qlora_merge.py
python quantization/main/TRT/qlora_merge_eval.py     # sanity-check merged FP16

# Export to fp32 ONNX, clean up, verify
python quantization/main/TRT/export_onnx.py
python quantization/main/TRT/export_onnx_clear.py
python quantization/main/TRT/export_onnx_verification.py
```

On EC2 (A10G) for TRT engine build:

```bash
# Required env (see §5.1)
source /opt/pytorch/bin/activate
export TMPDIR=/opt/dlami/nvme/.tmp HF_HUB_DISABLE_XET=1
export PYTHONPATH=/opt/ebs/py_extra:.
export LD_LIBRARY_PATH=/opt/ebs/py_extra/tensorrt_libs:$LD_LIBRARY_PATH

# Sprint 2 implicit (also produces calib.cache)
python quantization/main/TRT/build_trt_engine.py \
  --input-onnx  models/onnx/unet_fp32.onnx \
  --engine-path models/trt/unet_int8_fp16.engine

# B1-c recipe: insert Q/DQ
python quantization/main/TRT/qdq_marking.py \
  --input-onnx  models/onnx/unet_fp32.onnx \
  --output-onnx models/onnx/unet_int8_qdq_b1c_both.onnx \
  --calib-cache models/trt/unet_int8_fp16.calib.cache \
  --op-types Conv,MatMul --matmul-mode linear-only \
  --no-output-qdq --quantize-weights

# Build the deliverable TRT engine
python quantization/main/TRT/build_trt_engine.py \
  --input-onnx  models/onnx/unet_int8_qdq_b1c_both.onnx \
  --engine-path models/trt/unet_int8_b1c.engine \
  --qdq --workspace-gb 12

# Evaluate on 64 prompts
python evals/trt_eval.py \
  --engine models/trt/unet_int8_b1c.engine \
  --baseline-dir evals/eval_output/images/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8 \
  --repo-root . \
  --output-dir evals/eval_output/trt_b1c_64p --prompt-count 64
```

### 8.4 Showcase image

The 4-row × 6-col comparison sheet `evals/eval_output/showcase_4row.jpg` uses
3 custom prompts (idx 65/66/67) plus 3 indices from the COCO eval set
(03/04/60). Sources:

```bash
# EC2: FP16 baseline, Sprint 2 implicit TRT, Phase 3 B1-c TRT on custom prompts
python utils/gen_custom_3backends.py \
  --implicit-engine models/trt/unet_int8_fp16.engine \
  --b1c-engine     models/trt/unet_int8_b1c.engine

# Local: Phase 2 packed+LoRA row on the same custom prompts
python utils/gen_custom_phase2.py \
  --packed-safetensors models/real_quant/sdxl_lightning_weight/*.safetensors \
  --packed-manifest    models/real_quant/sdxl_lightning_weight/*.json \
  --lora quantization/main/packed/qlora_rank16_e3_lr1e4/lora_final.safetensors

# Stitch into the 4-row sheet
python utils/showcase.py
```

---

## 9. Handoff notes for the next agent

### 9.1 Where to start

1. Read this file end to end.
2. Skim `BENCHMARKS.md` for the numbers; raw JSONs live in
   `evals/eval_output/records/`.
3. Inspect `evals/eval_output/showcase_4row.jpg` for the qualitative story.

### 9.2 Known-good artifacts

- Phase 2: `models/real_quant/sdxl_lightning_weight/...packed.{safetensors,json}` (1.5 GB)
  + `quantization/main/packed/qlora_rank16_e3_lr1e4/lora_final.safetensors` (170 MB)
  → MSE 0.008 vs FP16 on 64 prompts at α=4.
- Phase 3 input: `models/onnx/unet_fp32.onnx` + `unet_fp32.onnx_data` (9.6 GB)
  → MSE 0.0057 vs PyTorch FP16 (4 prompts).
- Phase 3 deliverable: TRT engine `unet_int8_b1c.engine` (2.66 GB) on the
  EBS volume — MSE 0.0369 on 64 prompts, 0.98 s/image on A10G.

### 9.3 Recommended next experiments

1. **Per-layer sensitivity probe inside MatMul** — the 0.022 MSE budget of
   B1's activation Q/DQ is roughly uniform across blocks but probably *not*
   uniform across individual layers. Identifying the worst ~10 % of layers
   and excluding them from quantization is the most promising next lever.
2. **Cross-backend ONNX deployment demo** — OpenVINO / CoreML / RKNN. We
   already have a validated fp32 ONNX; running it on a second backend is
   small-effort, large-signal.
3. **Student-trajectory Q-LoRA** — replaces the inference α=4 compromise.
   Estimated 5–15 % MSE improvement; requires retraining LoRA on student-side
   trajectories.

### 9.4 Things to avoid

- Don't push tarballs / logs / model weights (>100 MB) to the public branch.
- Don't rewrite git history.
- Don't push without explicit user confirmation.
- Don't trust 2-prompt ablation results without a 4+ prompt confirmation
  (Sprint 3's multiplier sweep was a cautionary tale).
- Don't quantize Gemm or attention BMM in any future variant — both were
  shown experimentally to be catastrophic and trivial-to-avoid.

### 9.5 EC2 access

- SSH key (path may differ): `D:\chenh\chenh_key.pem`.
- Public IP rotates on stop/start — ask user for the current IP.
- EBS volume `vol-080fb2f650d1f01de` at `/opt/ebs` survives instance
  stop/start; tarballs, ONNX files, TRT engines all live there.
- AWS account capped at 8 vCPU (g5.2xlarge max).
- `Stop` instance to save cost; `Terminate` would lose EBS. Default action is
  always `Stop`.
