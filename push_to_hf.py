#!/usr/bin/env python3
"""
One-shot script: build the final diffusers pipeline with the mp_quant UNet
and push it to ChenHe727/EdgeDiffusion on HuggingFace.

Outputs after push:
  - Full pipeline (text_encoder, vae, scheduler, tokenizer, our quantized UNet)
    → users can do `StableDiffusionPipeline.from_pretrained("ChenHe727/EdgeDiffusion")`
  - lora_adapter.pt (separately, for advanced users who want QLoRA recovery)
  - README.md (model card with usage examples)
"""
import os
import sys
import torch
import json
from huggingface_hub import HfApi, login

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOKEN = open("hf_token").read().strip()
REPO_ID = "ChenHe727/EdgeDiffusion"

PTQ_WEIGHTS = "mp_quant/output/distill_final_mp_r37.safetensors"
PTQ_CONFIG  = "mp_quant/output/distill_final_mp_r37.config.json"
LORA_PATH   = "mp_quant/output/lora_adapter.pt"
BASE_MODEL  = "stabilityai/sd-turbo"


def main():
    login(token=TOKEN, add_to_git_credential=False)
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    # 1. Build pipeline with mp_quant UNet
    print("Building pipeline ...")
    from diffusers import StableDiffusionPipeline
    from prune.pruned_rebuild import create_unet_from_safetensors

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    print(f"Loading mp_quant UNet from {PTQ_WEIGHTS} ...")
    unet = create_unet_from_safetensors(PTQ_WEIGHTS, PTQ_CONFIG)
    unet = unet.to(dtype=torch.bfloat16)
    pipe.unet = unet
    print(f"  UNet params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

    # 2. Push full pipeline
    print(f"\nPushing pipeline to {REPO_ID} ...")
    pipe.push_to_hub(REPO_ID, token=TOKEN, commit_message="Upload mp_quant UNet + SD-Turbo components")

    # 3. Upload LoRA adapter separately
    print(f"\nUploading LoRA adapter ...")
    api.upload_file(
        path_or_fileobj=LORA_PATH,
        path_in_repo="lora_adapter.pt",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload QLoRA recovery adapter (Stage 4 output, optional)",
    )

    # 4. Upload mp_quant sidecar config (records bit-width assignment)
    print(f"Uploading mp_quant sidecar config ...")
    api.upload_file(
        path_or_fileobj=PTQ_CONFIG,
        path_in_repo="mp_quant_metadata.json",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload per-layer bit-width assignment metadata",
    )

    # 5. Upload model card
    print(f"Uploading model card ...")
    card = build_model_card()
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload model card",
    )

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


def build_model_card():
    """Generate a Hugging Face model card."""
    return """---
license: openrail++
library_name: diffusers
pipeline_tag: text-to-image
base_model: stabilityai/sd-turbo
tags:
  - diffusion
  - text-to-image
  - sd-turbo
  - quantization
  - pruning
  - distillation
  - edge-ai
  - mixed-precision
---

# EdgeDiffuse

Edge-deployable SD-Turbo via multi-stage compression: structural pruning → distillation → sensitivity-aware mixed-precision quantization (GPTQ) → QLoRA recovery.

**Code & paper-style writeup**: [github.com/SeanHe727/EdgeDiffusion](https://github.com/SeanHe727/EdgeDiffusion)

---

## What's in this repo

| File / dir | What it is |
|---|---|
| `unet/` | Mixed-precision quantized UNet (GPTQ-applied). 152 of 192 Linear layers quantized to INT4 (45) / INT8 (107); the rest stay fp16. Fake-quantized: values rounded to int grid, stored as bf16. |
| `text_encoder/`, `vae/`, `tokenizer/`, `scheduler/`, `model_index.json` | Standard `stabilityai/sd-turbo` components, unmodified |
| `lora_adapter.pt` | (Optional) QLoRA recovery adapter trained on top of the quantized UNet. Improves LPIPS by ~8 % when applied. See "Advanced: QLoRA recovery" below. |
| `mp_quant_metadata.json` | Per-layer bit-width assignment + GPTQ hyper-parameters for full reproducibility |

---

## Quick start

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "ChenHe727/EdgeDiffusion",
    torch_dtype=torch.bfloat16,        # required: INT4 layers use bf16 dtype
)
pipe = pipe.to("cuda")

image = pipe(
    "a photo of a tabby cat sitting on a wooden chair, sharp focus",
    num_inference_steps=2,             # 2-step is the sweet spot for SD-Turbo derivatives
    guidance_scale=0.0,                # SD-Turbo doesn't use CFG
).images[0]

image.save("output.png")
```

### Why 2 inference steps?

SD-Turbo is fundamentally trained with **adversarial diffusion distillation** for 1-step generation. Empirically, 2 steps gives the best quality/speed trade-off for our compressed model: 28 % faster than 4 steps with marginally better LPIPS.

---

## Results

Benchmark on RTX 5070 (Blackwell), 512 × 512, 2-step inference:

| Variant | Params | Latency | VRAM | LPIPS vs original SD-Turbo | LPIPS vs fp16 baseline |
|---|---:|---:|---:|---:|---:|
| stabilityai/sd-turbo (original) | 860 M | 0.146 s | 3.05 GB | 0 | 0.278 |
| fp16 baseline (pruned + distilled) | 642 M | 0.142 s | 2.64 GB | 0.278 | 0 |
| **this repo (mp_quant PTQ)** | 642 M | 0.145 s | 2.64 GB | 0.277 | 0.062 |
| with LoRA adapter loaded | 642 M + 9 MB | 0.171 s | 2.65 GB | 0.278 | **0.057** |

**Key takeaway**: mixed-precision quantization adds essentially **zero perceptual cost** on top of the pruned + distilled baseline (LPIPS 0.062 vs fp16). The dominant quality cost in the pipeline is the pruning stage; quantization is "free".

---

## Advanced: QLoRA recovery adapter

The included `lora_adapter.pt` was trained for 500 steps with step-wise teacher-student distillation to recover residual PTQ quality loss. It reduces the LPIPS gap from 0.062 to 0.057 (~8 % improvement).

```python
import torch
from peft import LoraConfig, get_peft_model
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
import json

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "ChenHe727/EdgeDiffusion", torch_dtype=torch.bfloat16,
).to("cuda")

# Discover which layers were quantized (LoRA targets these)
meta_path = hf_hub_download("ChenHe727/EdgeDiffusion", "mp_quant_metadata.json")
with open(meta_path) as f:
    meta = json.load(f)
target_fqns = [fqn for fqn, bit in meta["quantization"]["assignment"].items() if bit != "fp16"]

# Re-attach LoRA structure and load adapter weights
lora_state = torch.load(hf_hub_download("ChenHe727/EdgeDiffusion", "lora_adapter.pt"),
                        weights_only=False, map_location="cuda")
sample_key = next(k for k in lora_state if "lora_A" in k)
rank = lora_state[sample_key].shape[0]

pipe.unet = get_peft_model(pipe.unet, LoraConfig(
    r=rank, lora_alpha=rank * 2, target_modules=target_fqns,
    lora_dropout=0.0, bias="none",
))
own = pipe.unet.state_dict()
for k, v in lora_state.items():
    if k in own:
        own[k].copy_(v.to(own[k].device, dtype=own[k].dtype))
pipe.unet.eval()

# Generate as usual
image = pipe("a cat", num_inference_steps=2, guidance_scale=0.0).images[0]
```

---

## Pipeline overview

The model in this repo is the output of a three-stage compression pipeline applied to `stabilityai/sd-turbo`:

```
stabilityai/sd-turbo (860 M)
    ↓ structural pruning + step-wise distillation
ChenHe727/EdgeDiffusion_distilled_feat_attn (642 M, fp16)
    ↓ sensitivity-aware mixed-precision GPTQ (this repo's UNet)
    ↓ QLoRA recovery training (this repo's lora_adapter.pt)
ChenHe727/EdgeDiffusion (this repo)
```

Full design rationale, ablations, and reproducibility instructions: see the [GitHub repo](https://github.com/SeanHe727/EdgeDiffusion).

---

## Limitations

- **Conv2d layers are not quantized in v1** — only `nn.Linear` (attention projections, FFN). Conv2d holds ~70 % of UNet parameters; full quantization is planned for v2.
- **Fake-quant storage**: weights are rounded to INT4/INT8 grids but stored as bf16 (2 bytes/value). Real packed INT4/INT8 storage would shrink the file from 1.22 GB to ~900 MB but requires a separate packing step.
- **LPIPS vs original SD-Turbo ≈ 0.28** mostly comes from the upstream pruning + distillation stage. The quantization stage itself adds only 0.005-0.062.
- **2-step inference is the recommended default.** 1-step works (faster) but quality drops noticeably; 4-step is slower and not better.

---

## Acknowledgments

- **LD-Pruner** ([Castells et al. 2024](https://arxiv.org/abs/2404.11936)) — sensitivity metric
- **GPTQ** ([Frantar et al. 2023](https://arxiv.org/abs/2210.17323)) — Hessian-based PTQ (re-implemented from the paper in this repo)
- **QLoRA** ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) — parameter-efficient recovery
- **SD-Turbo** ([Sauer et al. 2023](https://stability.ai/research/adversarial-diffusion-distillation)) — base model
"""


if __name__ == "__main__":
    main()
