# EdgeDiffusion 本地 Real Quant 交接说明

这份文档给本地机器上的 agent 使用。当前目标是把 SDXL-Lightning 的 fake quant 结果推进到 real packed weight，并逐步走向真正 low-bit inference。

## 当前结论

我们已经从 pruning 路线转向 quantization 路线。当前最稳定、有价值的配置是：

```text
Base model: SDXL-Lightning 4-step
Resolution: 1024x1024
Guidance: 0.0

Weight:
  Linear W4
  Conv W8
  保持 FP16: conv_in / conv_out / time_embedding / time_emb_proj / add_embedding / bias / norm / 非 Linear-Conv

Activation:
  所有未 skip 的 Linear/Conv activation A8 都可以作为候选
```

已测得的理论/实验指标：

```text
UNet FP16 weight size:        5.135 GB
Packed weight estimate:       1.530 GB
Weight reduction:             70.2%

Quantized modules:            771
Linear W4:                    722 modules, 2207.7M params
Conv W8:                      49 modules, 333.2M params
FP16 kept:                    26.5M params

Activation quantizable traffic, 1024x1024 4-step:
  FP16 Linear/Conv activation traffic: 11.459 GB
  A8 all Linear/Conv traffic:          5.730 GB
  Activation traffic reduction:        50.0%
```

质量结果：

```text
Weight fake quant, Linear W4 + Conv W8, 32 prompts:
  MSE = 0.013139
  MAE = 0.068885
  肉眼质量可接受，minor degradation

Activation A8 all Linear/Conv vs weight-only baseline, 16 prompts:
  MSE = 0.001305
  MAE = 0.015476
  肉眼基本稳定
```

## 为什么换到本地

云实例的问题：

```text
GPU: A10G 23GB
RAM: 15GB
Root disk: 19GB, 容易满
NVMe ephemeral: stop/start 会丢数据
```

我们在云上运行 unpack/dequant 生成测试时，实例连接断开。判断主要原因不是 root disk，而是 RAM 峰值过高。之前的测试脚本同时持有：

```text
1. packed weight 文件:           1.53GB
2. dequant 后完整 fp16 state:    5.13GB
3. real-dequant pipeline
4. fp16 baseline pipeline
5. SDXL text encoders + VAE
6. 1024x1024 generation buffers
```

15GB RAM 没有 swap，很容易把实例打到不可连接。用户手动 stop/start 后 ephemeral NVMe 数据丢失。

本地机器：

```text
GPU: RTX 5070
RAM: 64GB
```

本地更适合做 real quant / unpack / module replacement，因为瓶颈主要是 CPU RAM 和稳定存储，而不是单纯 GPU。

## 已完成但云上丢失的脚本

云实例重启后，这些临时新增脚本可能已经丢失，需要在本地重建或从聊天记录恢复：

```text
mp_quant/pack_sdxl_lightning_weight.py
mp_quant/test_unpack_sdxl_lightning_weight.py
```

其中 `pack_sdxl_lightning_weight.py` 做的事：

```text
1. 加载 ByteDance/SDXL-Lightning 的 4-step UNet
2. Linear weight 按 group_size=128 做 symmetric W4
3. W4 packed 到 uint8 nibbles
4. Conv2d weight 按 group_size=128 做 symmetric W8
5. scale 保存为 FP16
6. skip/bias/norm 保存为 FP16
7. 输出 packed safetensors + JSON manifest
```

目标输出：

```text
models/real_quant/sdxl_lightning_weight/
  sdxl_lightning_4step_linearw4_convw8_packed.safetensors
  sdxl_lightning_4step_linearw4_convw8_packed.json
```

预期结果：

```text
actual packed file: 1.530 GB
reduction:          70.2%
verify max abs:     0.0
```

## 本地建议执行顺序

### 1. 环境检查

先确认本地 PyTorch/CUDA 对 RTX 5070 可用：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
PY
```

如果 SDXL-Lightning baseline 能正常生成，再继续 real quant。

### 2. 重新跑 baseline smoke

建议先跑 1-4 张 SDXL-Lightning 4-step baseline，确认本地环境没问题：

```bash
PYTHONPATH=. HF_HOME=.hf_cache HF_HUB_DISABLE_XET=1 TMPDIR=.tmp \
STEP=4 PROMPT_LIMIT=4 HEIGHT=1024 WIDTH=1024 GUIDANCE=0.0 DTYPE=float16 \
OUTPUT_DIR=gen_test_output/sdxl_lightning_local_baseline_smoke \
python prune/gen_sdxl_lightning_smoke.py
```

如果脚本路径不同，找已有的 SDXL-Lightning smoke 脚本；关键参数是 4-step、1024、CFG=0。

### 3. 重新生成 real packed weight

目标是只做 weight-only real packed artifact，不做推理 kernel。

推荐命令：

```bash
PYTHONPATH=. HF_HOME=.hf_cache HF_HUB_DISABLE_XET=1 TMPDIR=.tmp \
STEP=4 LINEAR_BITS=4 CONV_BITS=8 GROUP_SIZE=128 \
OUTPUT_DIR=models/real_quant/sdxl_lightning_weight \
python mp_quant/pack_sdxl_lightning_weight.py
```

预期输出：

```text
fp16 estimate:   5.135 GB
actual file:     1.530 GB
reduction:       70.2%
verify max abs:  0.0
```

### 4. 低内存 unpack/dequant 生成测试

不要再使用“同时加载 baseline pipeline + dequant pipeline”的方式。

安全做法：

```text
进程 A:
  只加载 fp16 baseline
  生成 baseline 图片
  退出进程

进程 B:
  只加载 packed artifact
  streaming dequant 到 UNet
  生成 real-dequant 图片
  退出进程

进程 C:
  只读取磁盘图片
  计算 MSE/MAE/contact sheet
```

最开始只跑：

```text
PROMPT_LIMIT=1
HEIGHT=768 或 512
```

确认不会 OOM 后再跑：

```text
PROMPT_LIMIT=8
HEIGHT=1024
WIDTH=1024
```

## 后续真正难点

目前 packed 文件只是 storage artifact。它可以真实减少磁盘大小，但还不能直接带来推理显存/速度收益。

下一步难点在：

```text
1. module replacement
   把 nn.Linear / nn.Conv2d 替换成 PackedInt4Linear / PackedInt8Conv2d

2. low-bit forward
   简单版: forward 时临时 dequant 到 fp16，然后 F.linear/F.conv2d
   真正版: 使用 int4/int8 kernel 直接计算

3. kernel / backend
   需要 torchao / Triton / CUDA / TensorRT 等支持

4. activation A8
   需要 dynamic scale 或 calibration scale，并接入 Linear/Conv/attention processor
```

推荐技术路线：

```text
Stage 1:
  Real packed storage
  已经验证过，需在本地重跑并保存

Stage 2:
  Low-memory unpack/dequant loader
  只验证 artifact 正确性，不追求省显存

Stage 3:
  Packed module replacement, temporary dequant forward
  目标是验证 diffusers module 替换路径

Stage 4:
  真正 low-bit kernel
  目标是真实 peak VRAM / latency improvement

Stage 5:
  Activation A8 real quant
  当前 fake quant 已验证 all Linear/Conv A8 可行
```

## 简历/GitHub 表述

当前最适合包装成：

```text
Post-training mixed-precision quantization pipeline for SDXL-Lightning edge deployment.
```

可以写：

```text
- Built a post-training mixed-precision quantization pipeline for SDXL-Lightning, achieving 70.2% UNet weight storage reduction with Linear W4 + Conv W8.
- Profiled activation quantization sensitivity and found all Linear/Conv activations can be quantized to A8 with minimal visual degradation, reducing quantizable activation traffic by 50%.
- Implemented a real packed weight artifact format using INT4 nibble packing, INT8 Conv storage, FP16 scales, and JSON manifest metadata.
```

暂时不要夸大为：

```text
real low-bit inference speedup
```

除非后面完成 module replacement 和 kernel benchmark。

## 注意事项

不要把大文件写到 root disk。统一设置：

```bash
export HF_HOME=$PWD/.hf_cache
export TMPDIR=$PWD/.tmp
export XDG_CACHE_HOME=$PWD/.cache_nvm
export TORCH_HOME=$PWD/.cache_nvm/torch
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
```

不要在 15GB RAM 的云实例上同时加载两套 SDXL pipeline。即使在本地 64GB，也建议分进程做 baseline / dequant / compare，避免峰值过高。


---

## 2026-05-17 Update: SDXL-Lightning Fake Quant and Block-Level Mixed Precision

This section records the latest local SDXL-Lightning quantization experiments after the RKNN SD1.5 route was deprioritized. The key difference is that this path is controlled on the PyTorch side: we decide which weights are fake-quantized and can later pack them into a real low-bit artifact. This avoids the black-box RKNN conversion-time PTQ problem.

### Environment

```text
Repo: D:\chenh\EdgeDiffusion\EdgeDiffusion
Runner repo: D:\chenh\model-compression-toolkit
Python: D:\chenh\Anaconda\envs\sd_native\python.exe
GPU: NVIDIA GeForce RTX 5070
Torch: CUDA build, CUDA available
Model: ByteDance/SDXL-Lightning 4-step UNet + SDXL base pipeline
Resolution for visual eval: 1024x1024
Steps: 4
Guidance: 0.0
Prompt set: 32 diverse prompts selected from dataset/*.txt
```

### New helper scripts

The new scripts are currently stored in `D:\chenh\model-compression-toolkit\tools`:

```text
tools/run_sdxl_lightning_dataset_fakequant.py
  Runs FP16 vs fake-quant image generation on selected dataset prompts.
  Supports Linear/Conv fake quant, Conv block filtering, and Linear block bit overrides.

tools/profile_sdxl_lightning_block_sensitivity.py
  Measures block-level Linear sensitivity for SDXL-Lightning.
  Each test quantizes all Linear layers inside one block together, then compares UNet output drift.

tools/profile_sdxl_lightning_sensitivity.py
  Earlier per-layer profiler. It produced partial results, but we decided not to use per-layer sensitivity for final decisions because it is too fine-grained for practical GPU/NPU deployment.
```

### Prompt selection

For the 32-prompt fake-quant tests, prompts were selected from `dataset/*.txt` with category coverage:

```text
people / animals / food / vehicle / sports / indoor / street / nature / objects
```

The selected prompt list is saved in each output directory as:

```text
selected_prompts.json
```

### Baseline visual smoke

A 1024x1024 SDXL-Lightning baseline smoke test was run first and produced normal images.

```text
Output:
D:\chenh\EdgeDiffusion\EdgeDiffusion\gen_test_output\sdxl_lightning_local_baseline_512
```

A 512 fake-quant smoke test also produced coherent images, confirming that PyTorch-side fake quant does not cause the RKNN-style snow/solid-color failure.

### Full fake quant: Linear W4 + Conv W8

Configuration:

```text
Linear: W4 fake quant
Conv2d: W8 fake quant
Skipped / FP16: conv_in, conv_out, time_embedding, time_emb_proj, add_embedding
Group size: 128
Resolution: 1024x1024
Prompts: 32
Steps: 4
```

Output:

```text
D:\chenh\EdgeDiffusion\EdgeDiffusion\gen_test_output\sdxl_lightning_fakequant_w4c8_dataset32_1024
```

Result:

```text
Quantized modules: 771
Linear layers: 722
Conv2d layers: 49
Quantized weights: 2540.95M
Targeted-weight theoretical reduction: 71.7%

Image MSE mean: 0.019322
Image MAE mean: 0.086825
Image MSE max:  0.043671
Image MAE max:  0.161166
```

Visual observation: images are coherent; there is no catastrophic noise, solid color, or numerical explosion. There are visible detail/semantic shifts compared with FP16, but the model remains functional.

### Deep Conv-only test

We tested whether quantizing only the Conv layers near the mid block helps quality.

Configuration:

```text
Linear: W4 fake quant
Conv2d W8 only in:
  down_blocks.1
  down_blocks.2
  up_blocks.0
  up_blocks.1
Other Conv2d: FP16
```

Output:

```text
D:\chenh\EdgeDiffusion\EdgeDiffusion\gen_test_output\sdxl_lightning_fakequant_w4c8_deepconv_dataset32_1024
```

Result:

```text
Quantized modules: 753
Linear layers: 722
Conv2d layers: 31
Quantized weights: 2467.43M

Image MSE mean: 0.019439
Image MAE mean: 0.087346
Image MSE max:  0.043521
Image MAE max:  0.160995
```

Conclusion: reducing Conv quantization did not meaningfully improve quality. The main quality loss is likely from aggressive Linear W4, not Conv W8.

### Linear-only W4 test

Configuration:

```text
Linear: W4 fake quant
Conv2d: FP16
Skipped / FP16: conv_in, conv_out, time_embedding, time_emb_proj, add_embedding
```

Output:

```text
D:\chenh\EdgeDiffusion\EdgeDiffusion\gen_test_output\sdxl_lightning_fakequant_linearw4_dataset32_1024
```

Result:

```text
Linear W4 layers: 722
Linear W4 params: 2207.74M

Image MSE mean: 0.019355
Image MAE mean: 0.087058
Image MSE max:  0.043545
Image MAE max:  0.160951
```

Conclusion: this is almost the same quality level as Linear W4 + Conv W8, again suggesting Linear W4 is the dominant source of error.

### Block-level Linear sensitivity

We stopped using per-layer sensitivity as the final decision basis because per-layer mixed precision is too fine-grained and may be inefficient or difficult to implement on GPU/NPU. Instead, we measured block-level sensitivity.

Method:

```text
For each block, quantize all Linear layers in that block together.
Compare UNet output against FP16 baseline on fixed calibration samples.
Metrics: 1 - cosine similarity, relative L2.
Resolution: 512x512
Prompts: 32
Steps: 4
```

Output:

```text
D:\chenh\EdgeDiffusion\EdgeDiffusion\mp_quant\results\sdxl_lightning_block_sensitivity_32p_512.json
```

Results:

```text
block          params     INT8 1-cos     INT8 rel_l2   INT4 1-cos     INT4 rel_l2
down_blocks.1   41.62M   1.38e-05       3.94e-03      2.48e-03       5.76e-02
down_blocks.2  701.24M   3.26e-06       1.70e-03      3.50e-04       2.33e-02
mid_block      350.62M   2.13e-06       1.20e-03      2.55e-04       1.92e-02
up_blocks.0   1051.85M   3.68e-06       1.95e-03      6.73e-04       3.18e-02
up_blocks.1     62.42M   6.41e-06       2.51e-03      1.55e-03       4.27e-02
```

Interpretation:

```text
INT8 is stable for all Linear blocks.
Best INT4 candidates: mid_block, down_blocks.2.
Acceptable INT4 candidate: up_blocks.0.
More sensitive blocks: up_blocks.1, down_blocks.1.
```

Note on block numbering:

```text
Only blocks containing Linear layers appear in this Linear sensitivity table.
Outer down/up blocks can be mostly Conv/ResNet and therefore may not appear here.
The names come directly from Diffusers module names, e.g. down_blocks.2 or up_blocks.0.
```

### Current best fake-quant configuration: block mixed precision Linear W4/W8

Based on block sensitivity, we tested a structured mixed-precision Linear scheme:

```text
mid_block Linear:      W4
down_blocks.2 Linear:  W4
up_blocks.0 Linear:    W4
Other Linear:          W8
Conv2d:                FP16
Critical modules:      FP16
  time_embedding
  time_emb_proj
  add_embedding
  conv_in
  conv_out
```

Output:

```text
D:\chenh\EdgeDiffusion\EdgeDiffusion\gen_test_output\sdxl_lightning_fakequant_blockmp_linear_w4w8_dataset32_1024
```

Result:

```text
Linear W4: 612 layers, 2103.71M params
Linear W8: 110 layers, 104.04M params
Conv2d: FP16

Image MSE mean: 0.015425
Image MAE mean: 0.073426
Image MSE max:  0.031745
Image MAE max:  0.128846
```

Comparison against uniform Linear W4:

```text
Uniform Linear W4:
  MSE mean: 0.019355
  MAE mean: 0.087058
  MSE max:  0.043545
  MAE max:  0.160951

Block MP Linear W4/W8:
  MSE mean: 0.015425
  MAE mean: 0.073426
  MSE max:  0.031745
  MAE max:  0.128846
```

Conclusion: block-level mixed precision improves quality clearly while keeping most Linear parameters in W4. This is currently the most promising fake-quant scheme.

### Implementation implications

Per-layer mixed precision is not recommended as the final deployment format because it may cause fragmented kernels, frequent format switching, and complicated scheduling.

Block-level mixed precision is more practical:

```text
W4 blocks:
  mid_block
  down_blocks.2
  up_blocks.0

W8 blocks:
  down_blocks.1
  up_blocks.1
  other non-critical Linear layers

FP16:
  critical embeddings / in-out / skipped modules
  Conv2d for the current best Linear-only test
```

Potential impact:

```text
Pros:
  Better quality than uniform W4.
  Still compresses most Linear weights aggressively.
  Easier to explain and maintain than per-layer assignment.

Cons:
  Real implementation must support at least W4 Linear and W8 Linear formats/kernels.
  FP16/W8/W4 block transitions may still introduce some scheduling overhead.
  If only one low-bit kernel format is available, we may need to simplify to all Linear W8 or selected-block W4 only.
```

### Suggested next steps

```text
1. Generate a real packed artifact for the current block MP Linear W4/W8 scheme.
2. Keep Conv2d FP16 initially, since Conv W8 did not seem to be the main quality issue.
3. Implement / test module replacement for block-level packed Linear.
4. Compare:
   - FP16 baseline
   - uniform Linear W4 fake quant
   - block MP Linear W4/W8 fake quant
   - real packed/dequant block MP Linear W4/W8
5. If real packed/dequant matches fake quant numerically, then optimize kernels.
6. Later test adding Conv W8 back if storage pressure requires it.
```
