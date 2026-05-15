#!/usr/bin/env python3
"""
Evaluate FID for a Hugging Face diffusion model.

Requirements (install before running):
  pip install torch torchvision diffusers transformers accelerate safetensors pillow pytorch-fid

Usage example:
  python tools/evaluate_fid.py \
    --model ChenHe727/EdgeDiffusion_distilled_final \
    --dataset celeba --num_images 500 --batch_size 4

Notes:
  - Default dataset is CelebA (downloaded via torchvision).
  - The script will save sampled real images and generated images to `output_dir` and compute FID using pytorch-fid.
  - If your model requires special conditioning (edge maps), pass appropriate prompts or adapt generation logic.
"""

import os
import argparse
import random
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from PIL import Image

from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
import importlib

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except Exception:
    calculate_fid_given_paths = None


def download_and_prepare_celeba(root, split='train', num_images=500, image_size=256, out_dir=None):
    ds_root = Path(root) / 'celeba'
    ds = CelebA(root=str(ds_root), split=split, download=True)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(ds)))
    random.shuffle(indices)
    selected = indices[:num_images]
    for i, idx in enumerate(selected):
        img, _ = ds[idx]
        img = transform(img)
        img.save(out_dir / f'{i:05d}.png')
    return out_dir


def generate_images_with_pipeline(model_id, device, out_dir, num_images=500, batch_size=4, prompt="a photo", seed=42, height=512, width=512, num_inference_steps=30, safetensors_path=None, safetensors_cfg=None, base_model=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    # If safetensors provided, rebuild UNet from safetensors + config
    unet = None
    if safetensors_path:
        # Use local path or download from HF repo if repo id provided
        st_path = safetensors_path
        cfg_path = safetensors_cfg
        if safetensors_path.startswith("hf://"):
            # format: hf://<repo_id>/<filename>
            parts = safetensors_path[len("hf://"):].split('/', 1)
            repo_id = parts[0]
            filename = parts[1]
            st_path = hf_hub_download(repo_id, filename)
            if cfg_path and cfg_path.startswith("hf://"):
                parts = cfg_path[len("hf://"):].split('/', 1)
                cfg_path = hf_hub_download(parts[0], parts[1])

        # import local pruned_rebuild
        pr = importlib.import_module('prune.pruned_rebuild')
        unet = pr.create_unet_from_safetensors(st_path, cfg_path)

    if unet is not None:
        sp_core = importlib.import_module('prune.sp_core')
        base_model = base_model or 'models/sd-turbo'
        base_model = sp_core.resolve_base_model_source(base_model)
        pipe = StableDiffusionPipeline.from_pretrained(base_model, unet=unet, torch_dtype=dtype)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    gen = torch.Generator(device=device).manual_seed(seed)

    n_batches = (num_images + batch_size - 1) // batch_size
    idx = 0
    for b in range(n_batches):
        this_bs = min(batch_size, num_images - idx)
        prompts = [prompt] * this_bs
        with torch.no_grad():
            out = pipe(prompts=prompts, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=7.5, generator=gen)
        images = out.images
        for im in images:
            im.save(out_dir / f'{idx:05d}.png')
            idx += 1
    return out_dir


def compute_fid(real_dir, gen_dir, batch_size=32, device='cuda'):
    if calculate_fid_given_paths is None:
        raise RuntimeError('pytorch-fid not installed. pip install pytorch-fid')
    paths = [str(real_dir), str(gen_dir)]
    fid_value = calculate_fid_given_paths(paths, batch_size, device, 2048)
    return fid_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Hugging Face model id, e.g. ChenHe727/EdgeDiffusion_distilled_final')
    parser.add_argument('--dataset', default='celeba', choices=['celeba'], help='Dataset to download/use')
    parser.add_argument('--data_root', default='./data', help='Location to store downloaded dataset')
    parser.add_argument('--output_dir', default='./fid_output', help='Where to store real/gen images and results')
    parser.add_argument('--num_images', type=int, default=500, help='Number of images to sample / generate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prompt', default='a high quality photograph of a person', help='Prompt for generation')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=30, help='Number of diffusion steps for sampling')
    parser.add_argument('--safetensors_path', default=None, help='Local path or hf://repo/file.safetensors to pruned safetensors')
    parser.add_argument('--safetensors_cfg', default=None, help='Local path or hf://repo/file.config.json for safetensors')
    parser.add_argument('--base_model', default='models/sd-turbo', help='Base model path/id for tokenizer/vae/etc (auto-downloads SD-Turbo into models/ if missing)')
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    real_dir = out_root / 'real'
    gen_dir = out_root / 'gen'
    out_root.mkdir(parents=True, exist_ok=True)

    print('Downloading dataset...')
    if args.dataset == 'celeba':
        download_and_prepare_celeba(args.data_root, split='train', num_images=args.num_images, image_size=min(args.height, args.width), out_dir=real_dir)
    else:
        raise NotImplementedError()

    print('Loading model and generating images...')
    if args.dataset == 'coco':
        # expect data_root/images and data_root/prompts
        real_dir = Path(args.data_root) / 'images'
        prompts_dir = Path(args.data_root) / 'prompts'
        prompts = []
        for p in sorted(prompts_dir.glob('*.txt')):
            with open(p, 'r', encoding='utf-8') as f:
                prompts.append(f.read().strip())
        if len(prompts) >= args.num_images:
            prompts = prompts[:args.num_images]
        else:
            while len(prompts) < args.num_images:
                prompts += prompts[:(args.num_images - len(prompts))]

        # build pipeline
        dtype = torch.float16 if args.device.startswith('cuda') else torch.float32
        if args.safetensors_path:
            pr = importlib.import_module('prune.pruned_rebuild')
            st_path = args.safetensors_path
            cfg_path = args.safetensors_cfg
            if st_path.startswith('hf://'):
                parts = st_path[len('hf://'):].split('/', 1)
                st_path = hf_hub_download(parts[0], parts[1])
            if cfg_path and cfg_path.startswith('hf://'):
                parts = cfg_path[len('hf://'):].split('/', 1)
                cfg_path = hf_hub_download(parts[0], parts[1])
            unet = pr.create_unet_from_safetensors(st_path, cfg_path)
            sp_core = importlib.import_module('prune.sp_core')
            base_model_source = sp_core.resolve_base_model_source(args.base_model)
            pipe = StableDiffusionPipeline.from_pretrained(base_model_source, unet=unet, torch_dtype=dtype).to(args.device)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
        pipe.enable_attention_slicing()
        gen = torch.Generator(device=args.device).manual_seed(42)
        idx = 0
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i+args.batch_size]
            with torch.no_grad():
                out = pipe(prompts=batch_prompts, height=args.height, width=args.width, num_inference_steps=args.num_inference_steps, guidance_scale=7.5, generator=gen)
            for im in out.images:
                im.save(gen_dir / f'{idx:05d}.png')
                idx += 1
    else:
        generate_images_with_pipeline(
            args.model,
            args.device,
            gen_dir,
            num_images=args.num_images,
            batch_size=args.batch_size,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=42,
            safetensors_path=args.safetensors_path,
            safetensors_cfg=args.safetensors_cfg,
            base_model=args.base_model,
        )

    print('Computing FID...')
    fid_value = compute_fid(real_dir, gen_dir, batch_size=32, device=args.device)
    print(f'FID: {fid_value:.4f}')


if __name__ == '__main__':
    main()
