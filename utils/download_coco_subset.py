#!/usr/bin/env python3
"""
Download a subset of COCO 2017 captions and images and save prompts.

This script downloads the COCO captions annotations zip, extracts
`captions_train2017.json`, then downloads the first `N` unique images
and saves one caption per image to a `.txt` file next to the image.

Usage:
  python tools/download_coco_subset.py --out_dir ./data/coco2017 --num 1000
"""

import os
import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ANNOT_ZIP_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
IMAGE_BASE = 'http://images.cocodataset.org/train2017/'


def download_annotations(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / 'annotations_trainval2017.zip'
    if not zip_path.exists():
        print('Downloading annotations zip...')
        urlretrieve(ANNOT_ZIP_URL, str(zip_path))
    else:
        print('Annotations zip already exists')
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extract('annotations/captions_train2017.json', path=str(dest_dir))
    return dest_dir / 'annotations' / 'captions_train2017.json'


def download_images_and_captions(annotations_json: Path, out_dir: Path, num_images: int = 1000):
    with open(annotations_json, 'r', encoding='utf-8') as f:
        ann = json.load(f)

    images = {im['id']: im for im in ann['images']}
    captions_by_image = {}
    for c in ann['annotations']:
        img_id = c['image_id']
        if img_id not in captions_by_image:
            captions_by_image[img_id] = c['caption']

    out_images = out_dir / 'images'
    out_images.mkdir(parents=True, exist_ok=True)
    out_prompts = out_dir / 'prompts'
    out_prompts.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, meta in images.items():
        if img_id not in captions_by_image:
            continue
        fname = meta['file_name']
        img_url = IMAGE_BASE + fname
        dest = out_images / fname
        if not dest.exists():
            try:
                urlretrieve(img_url, str(dest))
            except Exception as e:
                print(f'Failed to download {img_url}: {e}')
                continue
        # save prompt
        prompt = captions_by_image[img_id]
        with open(out_prompts / (Path(fname).stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write(prompt)
        count += 1
        if count >= num_images:
            break
        if count % 100 == 0:
            print(f'Downloaded {count}/{num_images}')

    print(f'Download complete: {count} images saved to {out_images}')
    return out_images, out_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./data/coco2017', help='Output base directory')
    parser.add_argument('--num', type=int, default=1000, help='Number of images to download')
    args = parser.parse_args()

    out = Path(args.out_dir)
    ann_json = download_annotations(out)
    download_images_and_captions(ann_json, out, num_images=args.num)


if __name__ == '__main__':
    main()
