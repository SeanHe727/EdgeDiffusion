"""Build README assets: benchmark charts + 3-column visual comparison.

Pulls images from the `sean` branch via `git show <ref>:<path>`, so it works
on `main` without checking those files into history.
"""
from __future__ import annotations
import io, subprocess
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------- benchmark data (from BENCHMARKS.md) ----------
stages = [
    "FP16\nbaseline",
    "Packed\n(Linear only,\nConv FP16)",
    "Packed\n(+ Conv W8)",
    "Packed\n+ Q-LoRA  ★",
]
ship_mb       = [4897, 1821,        1518,        1518 + 170]
load_vram_mb  = [9963, 1841,        1534,        1534]
peak_vram_mb  = [10767, 7609,       7303,        7303]
mse           = [0.0,  0.00794,     0.00808,     0.00794]

# ---------- chart 1: size + VRAM grouped bars ----------
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(stages)); w = 0.27
b1 = ax.bar(x - w, ship_mb,      w, label="Ship size (MB)",     color="#4C78A8")
b2 = ax.bar(x,     load_vram_mb, w, label="UNet load VRAM (MB)", color="#F58518")
b3 = ax.bar(x + w, peak_vram_mb, w, label="Inference peak VRAM (MB)", color="#54A24B")
for bars in (b1, b2, b3):
    for r in bars:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 120, f"{int(h)}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("MB"); ax.set_title("SDXL-Lightning UNet — size & VRAM across checkpoints (RTX 5070, 1024², 4-step)")
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(peak_vram_mb) * 1.18)
plt.tight_layout(); plt.savefig(ASSETS / "size_vram_chart.png", dpi=110); plt.close()
print("wrote", ASSETS / "size_vram_chart.png")

# ---------- chart 2: reduction % + quality (MSE) ----------
fig, ax1 = plt.subplots(figsize=(10, 4.2))
size_red = [(ship_mb[0]-v)/ship_mb[0]*100 for v in ship_mb]
peak_red = [(peak_vram_mb[0]-v)/peak_vram_mb[0]*100 for v in peak_vram_mb]
ax1.plot(stages, size_red, "o-", color="#4C78A8", label="Ship size reduction %", linewidth=2, markersize=8)
ax1.plot(stages, peak_red, "s-", color="#54A24B", label="Peak VRAM reduction %", linewidth=2, markersize=8)
ax1.set_ylabel("Reduction vs FP16 (%)"); ax1.set_ylim(-5, 80); ax1.grid(alpha=0.3)
for i, (s, p) in enumerate(zip(size_red, peak_red)):
    ax1.annotate(f"{s:.0f}%", (i, s), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#4C78A8")
    ax1.annotate(f"{p:.0f}%", (i, p), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color="#54A24B")
ax2 = ax1.twinx()
ax2.plot(stages, mse, "^--", color="#E45756", label="Per-pixel MSE vs FP16", linewidth=2, markersize=8)
ax2.set_ylabel("MSE (lower = closer to FP16)"); ax2.set_ylim(0, 0.012)
for i, m in enumerate(mse):
    ax2.annotate(f"{m:.4f}", (i, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#E45756")
ax1.set_title("Compression vs quality across checkpoints")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="lower right")
plt.tight_layout(); plt.savefig(ASSETS / "reduction_quality_chart.png", dpi=110); plt.close()
print("wrote", ASSETS / "reduction_quality_chart.png")

# ---------- chart 3: 3-column visual comparison ----------
SEAN = "sean"
fp16_dir = "gen_test_output/sdxl_lightning_gptq_blockmp_ffn4_attn8_conv8"
packed_dir = "qlora_eval_deepconvw8_packed_only"
lora_dir = "qlora_eval_deepconvw8_lora_a4"

def show(ref: str, path: str) -> Image.Image:
    raw = subprocess.check_output(["git", "show", f"{ref}:{path}"])
    return Image.open(io.BytesIO(raw)).convert("RGB")

def first_n_paths(ref: str, dir_: str, prefix: str, n: int):
    out = subprocess.check_output(["git", "ls-tree", "--name-only", ref, f"{dir_}/"]).decode().splitlines()
    return [p for p in sorted(out) if Path(p).name.startswith(prefix)][:n]

N = 6
fp16_paths   = first_n_paths(SEAN, fp16_dir,   "fp16_",        N)
packed_paths = first_n_paths(SEAN, packed_dir, "packed_only_", N)
lora_paths   = first_n_paths(SEAN, lora_dir,   "packed_lora_", N)
assert len(fp16_paths) == len(packed_paths) == len(lora_paths) == N, (len(fp16_paths), len(packed_paths), len(lora_paths))

THUMB = 320
def thumb(im): im = im.copy(); im.thumbnail((THUMB, THUMB), Image.LANCZOS); return im

rows = [
    ("FP16 baseline (4.9 GB)",              [thumb(show(SEAN, p)) for p in fp16_paths]),
    ("Packed (1.5 GB, no LoRA)",            [thumb(show(SEAN, p)) for p in packed_paths]),
    ("Packed + Q-LoRA (1.5 GB + 170 MB)",   [thumb(show(SEAN, p)) for p in lora_paths]),
]
PAD = 8; LABEL_W = 180
W = LABEL_W + PAD + (THUMB + PAD) * N
H = PAD + (THUMB + PAD) * len(rows)
sheet = Image.new("RGB", (W, H), "white")
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except OSError:
    font = ImageFont.load_default()
for ri, (label, imgs) in enumerate(rows):
    y = PAD + ri * (THUMB + PAD)
    # wrap label onto multiple lines at parens
    parts = label.replace(" (", "\n(").split("\n")
    for li, line in enumerate(parts):
        draw.text((8, y + THUMB // 2 - 14 + li * 18), line, fill="black", font=font)
    for ci, im in enumerate(imgs):
        x = LABEL_W + PAD + ci * (THUMB + PAD)
        sheet.paste(im, (x, y))
sheet.save(ASSETS / "comparison_3col.jpg", quality=85, optimize=True)
print("wrote", ASSETS / "comparison_3col.jpg", f"({(ASSETS / 'comparison_3col.jpg').stat().st_size/1024:.0f} KB)")
