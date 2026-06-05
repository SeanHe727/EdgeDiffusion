# EdgeDiffusion — SDXL-Lightning for Edge Deployment

A reproducible 3-phase post-training pipeline that takes the **SDXL-Lightning 4-step UNet** (2.6 B params, 4897 MB fp16) to an edge-deployable artifact, end-to-end from PyTorch research to a real INT8 TensorRT engine. Block-level mixed precision (W4 in the 3 deepest cross-attention blocks, W8 elsewhere) + self-implemented GPTQ + INT4-nibble packed safetensors + custom `PackedInt4Linear` / `PackedInt8Linear` / `PackedInt8Conv2d` modules + rank-16 Q-LoRA recovery, then ONNX export and TensorRT engine build with explicit Q/DQ — yielding a deployable INT8 engine on consumer / cloud GPUs.

## Headline results

| Stage | Artifact | Ship size | UNet VRAM (load / peak)\* | MSE vs FP16 | Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 baseline | SDXL-Lightning 4-step | 4897 MB | 9963 / 10767 MB | 0 | 3365 ms (5070) |
| **Phase 2 — PyTorch Packed + Q-LoRA** | `*.safetensors` + LoRA | **1518 + 170 MB** | **1534 / 7303 MB** | **0.0079** | 3365 ms (5070) |
| **Phase 3 — TensorRT INT8 engine** | `unet_int8_b1c.engine` | **2724 MB** | 19660 MB (full pipeline, A10G) | 0.037 | **980 ms (A10G)** |

\*Phase 2 measured on RTX 5070 12 GB; Phase 3 on AWS g5.2xlarge A10G 24 GB. Pipeline-level peak VRAM in Phase 3 is high because text-encoders + VAE stay FP16 on PyTorch alongside the TRT-backed UNet.

Full per-stage results and ablation studies in [BENCHMARKS.md](BENCHMARKS.md). Phase 3 EC2 work log in [EC2_results.md](EC2_results.md).

## Visual comparison

![4-row showcase](evals/eval_output/showcase_4row.jpg)

Rows: FP16 baseline / Sprint 2 TRT implicit (4710 MB, mostly-FP16-with-some-INT8) / Phase 2 Packed + Q-LoRA (PyTorch deliverable) / Phase 3 B1-c TRT INT8 (deployment deliverable).

## Three-phase workflow

```text
                ┌──────────────────────────────────────────────────────────────────┐
   Phase 1      │ Fake-quant + sensitivity profiling on PyTorch SDXL-Lightning     │
   (research)   │   - Block-level sensitivity → bit assignment (W4 / W8 / FP16)    │
                │   - Self-implemented GPTQ for W4 Linears (Hessian + Cholesky +   │
                │     column-wise error compensation)                              │
                │   - In-memory fake-quant; all ablations live here                │
                └────────────────────────────┬─────────────────────────────────────┘
                                             │ best bit assignment + GPTQ scales
                                             ▼
                ┌──────────────────────────────────────────────────────────────────┐
   Phase 2      │ Real packed deployment on PyTorch (consumer GPU)                 │
   (PyTorch     │   - INT4-nibble + INT8 + FP16 mixed safetensors + JSON manifest  │
    deliverable)│   - PackedInt4Linear / PackedInt8Linear / PackedInt8Conv2d       │
                │   - Q-LoRA rank-16 recovery trained against pre-cached teacher   │
                │     noise predictions (frozen packed weights)                    │
                │   - Inference-time α=4 scaling to mitigate exposure bias         │
                └────────────────────────────┬─────────────────────────────────────┘
                                             │ packed safetensors + LoRA
                                             ▼
                ┌──────────────────────────────────────────────────────────────────┐
   Phase 3      │ ONNX export + TensorRT INT8 engine (cross-backend deployment)    │
   (cloud       │   - Merge LoRA into FP16 UNet → fp32 ONNX export                 │
    deliverable)│   - TRT entropy calibration → calib.cache (per-tensor scales)    │
                │   - Insert explicit Q/DQ on Conv + linear-style MatMul (skip     │
                │     Gemm in embedding paths and attention BMM)                   │
                │   - Build TensorRT engine; real INT8 kernels fuse the Q/DQ       │
                └──────────────────────────────────────────────────────────────────┘
```

Each phase consumes the previous phase's artifact. Phase 1 is research-only (no shipping artifact). Phase 2's `*.safetensors` + LoRA is the PyTorch deliverable. Phase 3's `*.engine` is the cross-backend deployment deliverable.

## Repository layout

```text
quantization/
  main/
    fakequant/       Phase 1 — sensitivity profiling + GPTQ + fake-quant runner
    packed/          Phase 2 — packed storage + custom modules + Q-LoRA
    TRT/             Phase 3 — ONNX export + Q/DQ insertion + TRT engine build
  archived/          Exploration / negative results (torchao + bitsandbytes survey,
                     percentile calibration, contact-sheet helpers)

utils/               Common helpers (sensitivity profiling, COCO subset, showcase
                     builder, custom-prompt generators)

evals/
  onnx_eval.py       End-to-end ORT eval (Phase 3 ONNX validation)
  trt_eval.py        End-to-end TRT engine eval (Phase 3 deliverable)
  eval_output/
    showcase_4row.jpg
    {size_vram,reduction_quality}_chart.png
    images/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8/   FP16 baseline (64 PNG)
    records/         Per-experiment eval_summary.json (BENCHMARKS source of truth)
    trt_b1c_64p/     Phase 3 deliverable benchmark notes

models/
  real_quant/sdxl_lightning_weight/   Phase 2 packed UNet + manifest
  onnx/                               fp32 ONNX (Phase 3 input) + merged FP16

dataset/             COCO prompt subset (.txt only)
BENCHMARKS.md        Full numbers per experiment
EC2_results.md       Agent-readable detailed work log
```

## Environment setup

```bash
# 1) Create a fresh Python 3.10–3.12 environment.
# 2) Install a CUDA-enabled PyTorch matching your machine. Phase 1/2 was
#    developed with CUDA 12.8 on RTX 5070 (Blackwell sm_120):
pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.12.0.dev20260408+cu128
# 3) Install the rest of the Python deps:
pip install -r requirements.txt
```

TensorRT 10 (Phase 3 engine build) is installed separately on an A10G-class
EC2 instance; see [EC2_results.md §5.1](EC2_results.md) for the exact env
exports.

## Quick start

### Phase 2 — Reproduce the PyTorch deliverable

```bash
# 1) Fake-quant SDXL-Lightning (block-MP + GPTQ on W4 Linears)
python quantization/main/fakequant/run_fakequant.py \
  --repo-root . --output-dir evals/eval_output/images/quant \
  --method gptq --gptq-bits 4 --linear-bits 8 \
  --linear-block-bits "mid_block=4,down_blocks.2=4,up_blocks.0=4" \
  --conv-bits 8 --save-unet

# 2) Pack to INT4-nibble + INT8 + FP16 safetensors
python quantization/main/packed/pack_weight.py \
  --fakequant-dir evals/eval_output/images/quant \
  --output-dir   models/real_quant/sdxl_lightning_weight \
  --conv-include "down_blocks.1,down_blocks.2,mid_block,up_blocks.0,up_blocks.1" \
  --conv-bits 8

# 3) Q-LoRA recovery
python quantization/main/packed/qlora_cache_teacher.py \
  --repo-root . --output-dir quantization/main/packed/qlora_teacher_cache_128p_1024
python quantization/main/packed/qlora_train.py \
  --packed-safetensors models/real_quant/sdxl_lightning_weight/*.safetensors \
  --teacher-cache quantization/main/packed/qlora_teacher_cache_128p_1024 \
  --output-dir quantization/main/packed/qlora_rank16_e3_lr1e4 --rank 16

# 4) Evaluate vs FP16 baseline (α=4 at inference)
python quantization/main/packed/eval_packed_with_qlora.py \
  --packed-safetensors models/real_quant/sdxl_lightning_weight/*.safetensors \
  --lora quantization/main/packed/qlora_rank16_e3_lr1e4/lora_final.safetensors \
  --lora-alpha 4 --repo-root .
```

### Phase 3 — Reproduce the TensorRT INT8 engine

```bash
# 1) Merge LoRA into FP16 UNet, export fp32 ONNX
python quantization/main/TRT/qlora_merge.py
python quantization/main/TRT/export_onnx.py
python quantization/main/TRT/export_onnx_clear.py
python quantization/main/TRT/export_onnx_verification.py

# 2) Run TRT entropy calibration to produce calib.cache
python quantization/main/TRT/build_trt_engine.py \
  --input-onnx models/onnx/unet_fp32.onnx \
  --engine-path models/trt/unet_int8_fp16.engine

# 3) Insert explicit Q/DQ on Conv + linear-MatMul (B1-c recipe)
python quantization/main/TRT/qdq_marking.py \
  --input-onnx models/onnx/unet_fp32.onnx \
  --output-onnx models/onnx/unet_int8_qdq_b1c_both.onnx \
  --calib-cache models/trt/unet_int8_fp16.calib.cache \
  --op-types Conv,MatMul --matmul-mode linear-only \
  --no-output-qdq --quantize-weights

# 4) Build the deliverable engine
python quantization/main/TRT/build_trt_engine.py \
  --input-onnx models/onnx/unet_int8_qdq_b1c_both.onnx \
  --engine-path models/trt/unet_int8_b1c.engine --qdq

# 5) Evaluate end-to-end
python evals/trt_eval.py \
  --engine models/trt/unet_int8_b1c.engine \
  --baseline-dir evals/eval_output/images/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8 \
  --output-dir evals/eval_output/trt_b1c_64p --prompt-count 64
```

## Hardware tested

| Role | Hardware | Phase |
| --- | --- | --- |
| PyTorch development + Phase 2 eval | RTX 5070 12 GB (sm_120 Blackwell), Windows 11, 64 GB RAM, PyTorch 2.12 + CUDA 12.8 | Phase 1, 2 |
| ONNX export | RTX 5070 12 GB | Phase 3 prep |
| TRT engine build + Phase 3 eval | AWS EC2 g5.2xlarge, A10G 24 GB, 8 vCPU, 32 GB RAM, TRT 10 | Phase 3 |

Phase 2 packed UNet targets **7303 MB inference peak VRAM** — fits RTX 4060+ / 4070+ / 5070+ desktops and Jetson Orin 16 GB. Phase 3 TRT engine targets A10G-class data-center GPUs with native INT8 kernels.

## Acknowledgments

- **SDXL-Lightning** ([Lin et al. 2024](https://arxiv.org/abs/2402.13929)) — 4-step distilled base model.
- **GPTQ** ([Frantar et al. 2023](https://arxiv.org/abs/2210.17323)) — Hessian-based PTQ algorithm reimplemented here.
- **QLoRA** ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) — parameter-efficient quantized fine-tuning, adapted from LLMs to diffusion.
- **TensorRT 10** — INT8 graph optimizer and execution engine.

Maintainer: Chen He ([`SeanHe727`](https://github.com/SeanHe727) on GitHub, [`ChenHe727`](https://huggingface.co/ChenHe727) on Hugging Face).
