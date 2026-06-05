# Phase 3 — B1-c TRT INT8 engine — 64-prompt benchmark

Engine: `models/trt/unet_int8_b1c.engine` (2.66 GB)
Source ONNX: `unet_int8_qdq_b1c_both.onnx` — fp32 ONNX with explicit Q/DQ on
input activations + per-tensor weight QDQ for Conv + linear-style MatMul;
attention BMM and embedding Gemm both skipped.
Hardware: AWS EC2 g5.2xlarge, NVIDIA A10G 24 GB.
Pipeline: SDXL-Lightning 4-step at 1024 × 1024, TRT engine for UNet only;
text encoders + VAE remain FP16 on PyTorch.

## Results (64 prompts vs FP16 baseline)

| Metric | Value |
| --- | ---: |
| MSE mean | **0.0369** |
| MSE max | 0.0898 |
| Latency (steady-state) | **0.98 s / image** |
| Build time | 25.6 min |
| Engine size | 2.66 GB |
| Engine weights memory | 2.62 GB |
| Engine activation memory | 178 MB |

## Reference points

| Engine | Size | Weights mem | Activation mem | MSE | Latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sprint 2 implicit (entropy calib) | 4.6 GB | 4.82 GB | 172 MB | 0.012 | 1.23 s |
| Sprint 2b explicit v3 (broken) | 2.95 GB | 2.73 GB | 474 MB | 0.166 | 1.22 s |
| **B1-c (delivered)** | **2.66 GB** | **2.62 GB** | **178 MB** | **0.037** | **0.98 s** |

## Files

- `trt_e2e_0{1..6}*.png` — first 6 of 64 generated images for visual inspection.
- `benchmark_summary.txt` — TRT eval script summary tail.
- Full 64 PNGs remain on EC2 EBS at `gen_test_output/eval_trt_b1c_64p/`.
