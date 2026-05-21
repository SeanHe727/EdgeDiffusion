# EC2 Handoff — Sprint 1 (ONNX INT8) + Sprint 2 (TensorRT)

You are the agent on the EC2 instance. The user is moving here from a local Windows + RTX 5070 12 GB box because:
- `onnxruntime.quantization.quantize_static` hits `bad_allocation` at LayerNorm during calibration on Windows even with 64 GB RAM
- Windows console GBK encoding fails on ORT/PyTorch unicode prints (✅ emoji)
- C: drive is 100 % full → temp file paths break
- TensorRT 10 INT8 kernel coverage on Blackwell consumer (sm_120) is unclear

EC2 instance: **g5.2xlarge** (A10G 24 GB VRAM, 8 vCPU, 32 GB RAM, ~450 GB NVMe ephemeral, Linux). Datacenter A10G is sm_86 (Ampere) with mature ORT/TRT INT8 support.

---

## 0. What the user wants out of you

Two deliverables, in order:

### Sprint 1 deliverable — INT8 QDQ ONNX
A single portable ONNX file usable on NVIDIA GPU / Intel CPU / Apple / RKNN, with INT8 weights compressed properly. Target:
- **Size**: ~2-3 GB (vs 9.8 GB fp32 source = ≥70 % reduction)
- **Quality**: MSE vs the 64-prompt FP16 baseline ≤ 0.02 (target ≈ 0.01; reference fp32 ONNX MSE is 0.0057)
- **Functional**: loads + runs on ORT CUDA EP without runtime errors

### Sprint 2 deliverable — TensorRT engine
A TRT engine built from the fp32 ONNX with INT8 (and optionally INT4) mixed-precision kernels. Target:
- **Size**: ~1.5-2.5 GB engine file
- **Latency**: noticeably faster than FP16 baseline (anything ≥ 1.3× is a win)
- **Quality**: MSE ≤ 0.02 vs FP16 baseline

You can do Sprint 1 first then Sprint 2. They share the same fp32 ONNX input. Do not redo the PyTorch quantization / LoRA training — that's done.

---

## 1. What's in the upload bundle

The user is uploading a tarball. Expected contents (paths relative to repo root):

```text
models/onnx/
  unet_fp32.onnx                    2.8 MB graph
  unet_fp32.onnx_data               9.8 GB external weights
  unet_fp32_export_summary.json     export metadata

qlora_teacher_cache_128p_1024/      ~285 MB
  index.json                         calib index
  prompt0000_step0.safetensors       512 calibration samples
  ... (× 512 files)

gen_test_output/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8/
  fp16_*.png                         64 FP16 baseline images (for MSE comparison)
  selected_prompts.json              64 prompt definitions w/ seeds

mp_quant/                           all SDXL pipeline source
  gptq.py
  packed_linear.py
  build_packed_unet.py
  qlora_lora.py
  trt_prepare_fp16_unet.py
  trt_export_onnx.py
  verify_onnx_vs_pytorch.py
  eval_onnx_e2e.py
  onnx_quantize_static_qdq.py
  onnx_cleanup_dead_initializers.py
  __init__.py

tools/
  (sensitivity profilers — not needed for this task)

BENCHMARKS.md                       complete history of results
EC2_HANDOFF.md                      THIS FILE
```

If `models/real_quant/sdxl_lightning_weight/...packed.safetensors` is also uploaded you don't need it for either sprint; ignore.

---

## 2. Environment setup

The instance is fresh. Set up Python 3.10+ + CUDA 12 + the relevant libraries.

### 2.1 Conda / venv

```bash
# Using miniconda (common on DLAMI)
conda create -n sd python=3.10 -y
conda activate sd

# Or venv if conda unavailable
python3.10 -m venv ~/sd && source ~/sd/bin/activate
```

### 2.2 Install deps

```bash
# PyTorch with CUDA 12.x (A10G is sm_86, fully supported by recent torch)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Diffusion + ONNX ecosystem
pip install diffusers transformers safetensors accelerate huggingface_hub
pip install onnx onnxscript onnxruntime-gpu
pip install numpy pillow

# For Sprint 2 (install eagerly; we'll use it later)
pip install tensorrt
pip install onnx-graphsurgeon polygraphy

# Sanity check
python -c "
import torch, onnx, onnxruntime as ort, tensorrt
print(f'torch:        {torch.__version__}')
print(f'CUDA cap:     sm_{int(torch.cuda.get_device_capability()[0])}{int(torch.cuda.get_device_capability()[1])}')
print(f'onnx:         {onnx.__version__}')
print(f'ort:          {ort.__version__}  EPs={ort.get_available_providers()}')
print(f'tensorrt:     {tensorrt.__version__}')
"
```

Expected:
```
torch: 2.x.x
CUDA cap: sm_86  (A10G)
onnx: 1.x
ort: 1.x  EPs=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
tensorrt: 10.x
```

### 2.3 Disk + env vars

Put all caches on the NVMe (assume mounted at `/opt/dlami/nvme/` per AWS DLAMI; adjust if different):

```bash
mkdir -p /opt/dlami/nvme/edge/{.hf_cache,.tmp}

export EDGE=/opt/dlami/nvme/edge
export HF_HOME=$EDGE/.hf_cache
export HF_HUB_DISABLE_XET=1
export TMPDIR=$EDGE/.tmp
export PYTHONIOENCODING=utf-8

cd ~/EdgeDiffusion       # or wherever you extracted the tarball
```

Confirm the upload landed:
```bash
ls -la models/onnx/unet_fp32.onnx_data       # should be ~9.8 GB
ls qlora_teacher_cache_128p_1024/ | wc -l    # should be ~513
ls gen_test_output/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8/fp16_*.png | wc -l   # should be 64
```

---

## 3. Sprint 1 — Static QDQ INT8 ONNX

### 3.1 Run the quantizer

The script is `mp_quant/onnx_quantize_static_qdq.py`. The local failure was `bad_allocation` at LayerNorm during calibration with Percentile + 128 samples + Windows. On Linux + 32 GB RAM + 24 GB A10G this should fit. The key is CUDA-side calibration.

Run as-is with the recommended config:

```bash
PYTHONPATH=. python mp_quant/onnx_quantize_static_qdq.py \
  --input-onnx  models/onnx/unet_fp32.onnx \
  --output-onnx models/onnx/unet_int8_qdq.onnx \
  --teacher-cache qlora_teacher_cache_128p_1024 \
  --n-calib 128 \
  --calib-method percentile \
  --activation-type uint8 \
  --skip-preprocess \
  2>&1 | tee logs/phase_c_static_qdq.log
```

Inside, the script already passes
`extra_options={"CalibProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"]}`
so calibration forward should run on A10G GPU. If it doesn't (older ORT version), edit the extra_options key (sometimes named `CalibrationProviders` in different versions) and retry.

Expected runtime: 15-30 min. Expected output (raw, before cleanup):
```
unet_int8_qdq.onnx        ~9 MB graph
unet_int8_qdq.onnx.data   ~7-9 GB  (still has dead fp32 initializers)
```

If `bad_allocation` still happens:
1. Drop to `--n-calib 64`
2. Drop to `--calib-method minmax` (uses less memory than percentile)
3. Add `--no-conv` (skip Conv quantization — Conv is what triggers LayerNorm-adjacent OOM on 12 GB GPU; on 24 GB this should not be necessary but is the escape hatch)
4. Verify VRAM with `nvidia-smi` during the run

If you change the recipe, document what worked in `logs/phase_c_static_qdq.log`.

### 3.2 Cleanup dead fp32 initializers

ORT leaves the original fp32 weight initializers in the file even after they're replaced by int8 + DequantizeLinear chains. The cleanup script removes them.

```bash
PYTHONPATH=. python mp_quant/onnx_cleanup_dead_initializers.py \
  --input-onnx  models/onnx/unet_int8_qdq.onnx \
  --output-onnx models/onnx/unet_int8_qdq_clean.onnx \
  2>&1 | tee logs/phase_c_cleanup.log
```

Expected:
```
Reduction:  75-85 %
Output size: ~1.5 - 2.5 GB total (unet_int8_qdq_clean.onnx + .onnx_data)
```

If reduction is only ~10 % then either the quantizer didn't actually convert weights (rare) or some unused-initializer detection is wrong — share the cleanup log and ask the user before continuing.

### 3.3 End-to-end verification

```bash
PYTHONPATH=. python mp_quant/eval_onnx_e2e.py \
  --onnx models/onnx/unet_int8_qdq_clean.onnx \
  --baseline-dir gen_test_output/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8 \
  --repo-root . \
  --output-dir gen_test_output/eval_int8_qdq_clean_4p \
  --prompt-count 4 \
  --ep cuda \
  2>&1 | tee logs/phase_c_int8_e2e.log
```

This downloads SDXL base (text encoders + VAE) from HuggingFace on first run (~10 GB to `$HF_HOME`). Then generates 4 images using the INT8 UNet via ORTUNetWrapper.

**Gate**: MSE mean vs FP16 baseline must be ≤ 0.02 (ideally ≤ 0.01). The fp32 ONNX reference (already validated locally) hits 0.00573 on the same 4 prompts. If MSE > 0.05, the quantization broke the model — check log, inspect one image visually (it should look like an actual scene, not structured noise).

If quality is too low:
- Switch to **MinMax + QUInt8 + 64 samples + --no-conv** (the safest config; expect smaller compression because Conv is preserved fp32)
- Or try **--calib-method entropy** (different statistical method, sometimes better than Percentile for diffusion)
- Document each attempt in `logs/`

If you load the ONNX and ORT throws `transformer_memcpy node_provider != nullptr` or `Could not find an implementation for ConvInteger`: that means a wrong op type got quantized. The script already excludes Conv from the QOperator path, so this shouldn't happen with static QDQ — but if it does, re-run with `--no-conv`.

### 3.4 Sprint 1 wrap-up

When the cleaned INT8 QDQ ONNX passes the quality gate, write a short summary to `logs/SPRINT1_RESULT.md` with:
```
Final ONNX:         path + size
Compression ratio:  X % vs fp32
Quality (4 prompts): MSE mean = ..., MSE max = ...
Latency / image (CUDA EP):  ... s   (raw; not directly comparable to baseline since ORTUNetWrapper has CPU↔GPU copy overhead)
Quantizer config used:  calib_method=..., activation_type=..., n_calib=..., op_types=..., nodes_excluded=...
```

Move on to Sprint 2.

---

## 4. Sprint 2 — TensorRT engine build

Same source ONNX (`unet_fp32.onnx`), different target backend. TRT 10 has more mature INT8 (and Blackwell datacenter has INT4 — A10G is Ampere so INT8 is the realistic target; INT4 may fall back to FP16).

There is **no existing TRT script** in the repo. You need to write it. Sketch below.

### 4.1 Build calibration data for TRT

TRT's `IInt8EntropyCalibrator2` expects to feed batches of input tensors. Reuse the 256 cached samples in `qlora_teacher_cache_128p_1024/`. Write a calibrator class that:
- Reads N samples (start with 64)
- Returns each as the 5-input feed dict matching the ONNX input names (`sample, timestep, encoder_hidden_states, text_embeds, time_ids`)
- Caches a calibration cache file under `models/trt/` so subsequent builds can skip recalibration

### 4.2 Build engine

```bash
mkdir -p models/trt

trtexec \
  --onnx=models/onnx/unet_fp32.onnx \
  --saveEngine=models/trt/unet_int8.engine \
  --int8 --fp16 \
  --calib=models/trt/calib.cache \
  --workspace=16384 \
  --memPoolSize=workspace:16384 \
  --useCudaGraph \
  --shapes=sample:1x4x128x128,timestep:1,encoder_hidden_states:1x77x2048,text_embeds:1x1280,time_ids:1x6 \
  2>&1 | tee logs/phase_d_trt_build.log
```

Note: `trtexec --int8` without a calibration file will use TRT's default entropy calibrator with synthetic data — bad quality. Use the calibrator class via the Python TensorRT API to produce a real calib cache first.

A more controlled script:

```python
# mp_quant/build_trt_engine.py  (you write this)
import tensorrt as trt
from pathlib import Path

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("models/onnx/unet_fp32.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)   # 16 GB
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = MyTeacherCacheCalibrator(...)   # your class

# (Optional) Force precision per layer here if you want to mimic the
# block-MP A recipe — see BENCHMARKS.md §5

engine = builder.build_serialized_network(network, config)
Path("models/trt/unet_int8.engine").write_bytes(engine)
```

Calibrator skeleton (CRITICAL for quality):
```python
class TeacherCacheCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_dir: Path, n: int, batch_size: int = 1):
        super().__init__()
        self.files = sorted(cache_dir.glob("prompt*_step*.safetensors"))[:n]
        self.idx = 0
        self.bs = batch_size
        self.cache_file = Path("models/trt/calib.cache")
        # Allocate device buffers for inputs ...

    def get_batch_size(self): return self.bs

    def get_batch(self, names):
        if self.idx >= len(self.files): return None
        sample = load_file(str(self.files[self.idx]))
        self.idx += 1
        # Copy each input into the right device pointer, return list of ptrs
        # Map names → tensor: 'sample' → latent_in, 'timestep' → timestep,
        # 'encoder_hidden_states' → prompt_embeds, 'text_embeds' → pooled_embeds,
        # 'time_ids' → time_ids. Cast types to fp32 / int64 as needed.
        return [int(self.dev_buffers[n].data_ptr()) for n in names]

    def read_calibration_cache(self):
        if self.cache_file.exists(): return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(cache)
```

Expected build time: 30-90 min. Watch `nvidia-smi` — TRT builder uses lots of VRAM.

### 4.3 TRT engine wrapper + e2e

Write `mp_quant/trt_unet_wrapper.py` that loads the engine, exposes a `forward(sample, timestep, encoder_hidden_states, added_cond_kwargs, return_dict=True, **)` matching `UNet2DConditionModel.forward()`. Also stub `.config` and `.add_embedding.linear_1.in_features = 2816` (see `mp_quant/eval_onnx_e2e.py` for the ORT analogue — copy that pattern).

Then run `mp_quant/eval_onnx_e2e.py`-style eval with `pipe.unet = TRTUNet(...)`:

```bash
PYTHONPATH=. python mp_quant/eval_trt_e2e.py \
  --engine models/trt/unet_int8.engine \
  --baseline-dir gen_test_output/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8 \
  --repo-root . \
  --output-dir gen_test_output/eval_trt_int8 \
  --prompt-count 4 \
  2>&1 | tee logs/phase_d_trt_e2e.log
```

Measure latency this time — TRT should give a real number (no ORTUNetWrapper overhead). Compare to FP16 PyTorch baseline of 3365 ms/image (RTX 5070; on A10G the baseline is different — generate one FP16 PyTorch image on A10G first as the local reference).

### 4.4 Sprint 2 wrap-up

Write `logs/SPRINT2_RESULT.md` with:
```
Engine path + size:
Calibration: N samples, method
FP16 PyTorch baseline (A10G):  ... ms/image
TRT INT8+FP16 mixed (A10G):     ... ms/image    (speedup: X×)
Quality vs FP16 baseline (4 prompts):  MSE mean = ..., MSE max = ...
```

---

## 5. Local baseline numbers (for your sanity checks)

These are from the local RTX 5070 runs (`BENCHMARKS.md §3-§5`). Use them only as ballpark — A10G is a different GPU, baselines will differ:

| Config | MSE mean (64p, 1024x1024, 4 step) | UNet load VRAM | Gen peak VRAM |
|---|---|---|---|
| FP16 PyTorch (5070) | 0 (it is the reference) | 9.9 GB | 10.8 GB |
| Packed W4/W8 + Q-LoRA (5070) | 0.00794 | 1.5 GB | 7.3 GB |
| fp32 ONNX (5070 via ORT CUDA EP) | 0.00573 (4 prompts) | not relevant | not relevant |

Your A10G numbers will differ — don't panic if VRAM is bigger (24 GB available) or latency is different.

---

## 6. Pitfalls collected from local runs (avoid re-discovering)

- **ConvInteger has no kernel** in ORT for SDXL. → Don't use `quantize_dynamic` with Conv in op_types. Use `quantize_static + QDQ` if you want Conv quantized.
- **`onnx.checker.check_model` complains** if external data is sharded into many files. Cleanup script (`onnx_cleanup_dead_initializers.py`) consolidates into one `.onnx_data` file.
- **ORT static_quantize doesn't auto-delete fp32 weights** that got replaced by int8 + DQ chain. Always run cleanup pass after quantize_static.
- **PyTorch 2.12 dynamo exporter** breaks SDXL with mixed fp16/fp32 Concat errors. Use legacy exporter (`dynamo=False`). Already encoded in `trt_export_onnx.py` (you don't need to re-export — the fp32 ONNX in the bundle is good).
- **diffusers SDXL pipeline pokes at `unet.add_embedding.linear_1.in_features`** — see how `eval_onnx_e2e.py:ORTUNetWrapper` stubs it. Replicate this in the TRT wrapper.
- **`load_file` from safetensors** uses `device="cpu"` by default; pass `device="cuda"` if you want to skip a copy.

---

## 7. How to report back

When you finish (or get stuck), have ready:

1. `logs/SPRINT1_RESULT.md` and/or `logs/SPRINT2_RESULT.md`
2. Final ONNX file (`unet_int8_qdq_clean.onnx` + `.onnx_data`) ready to download
3. Final TRT engine file (`unet_int8.engine`) ready to download
4. 4 generated images per sprint for visual eyeballing
5. Any blocker / unexpected behavior with the relevant log file

Push the deliverables to a public S3 bucket or just leave them on the instance with paths in the summary — the user can `scp` them down.

---

Good luck. The fp32 ONNX is solid (locally validated MSE 0.00573 vs FP16 baseline) — that's your stable starting point for both sprints.
