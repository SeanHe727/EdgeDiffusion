#!/usr/bin/env bash
# setup_instance.sh — One-time instance setup for AWS DLAMI with NVMe at /opt/dlami/nvme/
#
# Run once after cloning the repo on a fresh instance:
#   bash setup_instance.sh
#
# What this does:
#   1. Creates required directories on NVMe (fast storage)
#   2. Copies dataset/*.txt prompts (git-tracked) from repo into NVMe
#   3. Creates symlinks: models/ .hf_cache .tmp dataset -> NVMe
#   4. Downloads stabilityai/sd-turbo to models/sd-turbo/ if missing

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVME="/opt/dlami/nvme"
PYTORCH_ENV="/opt/pytorch/bin/activate"

echo "=== Instance Setup for model-compression-toolkit ==="
echo "Repo:    $REPO_DIR"
echo "NVMe:    $NVME"
echo ""

# ── 1. Create NVMe directories ──────────────────────────────────────────
echo "[1/4] Creating NVMe directories..."
mkdir -p "$NVME/models"
mkdir -p "$NVME/hf_cache"
mkdir -p "$NVME/tmp"
mkdir -p "$NVME/dataset"

# ── 2. Copy .txt prompts BEFORE creating the symlink ────────────────────
# At this point dataset/ is still the real git-tracked directory.
# After step 3, it becomes a symlink and the originals are no longer reachable.
echo ""
echo "[2/4] Copying prompt .txt files to NVMe..."
if [ -d "$REPO_DIR/dataset" ] && [ ! -L "$REPO_DIR/dataset" ]; then
    TXT_COUNT=$(find "$REPO_DIR/dataset" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)
    if [ "$TXT_COUNT" -gt 0 ]; then
        cp "$REPO_DIR/dataset/"*.txt "$NVME/dataset/"
        echo "  Copied $TXT_COUNT prompt files -> $NVME/dataset/"
    else
        echo "  No .txt files found in repo dataset/ — nothing to copy"
    fi
else
    echo "  dataset/ is already a symlink or missing — skipping copy"
    EXISTING=$(find "$NVME/dataset" -maxdepth 1 -name "*.txt" 2>/dev/null | wc -l)
    echo "  NVMe dataset has $EXISTING prompt files"
fi

# ── 3. Create symlinks in repo root ─────────────────────────────────────
echo ""
echo "[3/4] Creating symlinks in $REPO_DIR ..."
cd "$REPO_DIR"

ln -sfn "$NVME/models"   models
echo "  models    -> $NVME/models"

ln -sfn "$NVME/hf_cache" .hf_cache
echo "  .hf_cache -> $NVME/hf_cache"

ln -sfn "$NVME/tmp"      .tmp
echo "  .tmp      -> $NVME/tmp"

ln -sfn "$NVME/dataset"  dataset
echo "  dataset   -> $NVME/dataset ($(find "$NVME/dataset" -maxdepth 1 -name "*.txt" | wc -l) prompts)"

# ── 4. Download SD-Turbo ─────────────────────────────────────────────────
echo ""
echo "[4/4] Checking SD-Turbo model..."

if [ ! -f "$PYTORCH_ENV" ]; then
    echo "WARNING: PyTorch env not found at $PYTORCH_ENV — skipping model download"
    echo "Activate your env and run:"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-turbo', local_dir='models/sd-turbo')\""
else
    source "$PYTORCH_ENV"
    export HF_HOME="$REPO_DIR/.hf_cache"
    export TRANSFORMERS_CACHE="$REPO_DIR/.hf_cache"
    export TMPDIR="$REPO_DIR/.tmp"

    if [ -f "$REPO_DIR/models/sd-turbo/model_index.json" ]; then
        echo "  models/sd-turbo/ already exists — skipping download"
    else
        echo "  Downloading stabilityai/sd-turbo (~13GB) to models/sd-turbo/ ..."
        python - << 'PYEOF'
from huggingface_hub import snapshot_download
import warnings; warnings.filterwarnings('ignore')
snapshot_download(repo_id='stabilityai/sd-turbo', local_dir='models/sd-turbo')
print('  SD-Turbo download complete.')
PYEOF
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick smoke test:"
echo "  source $PYTORCH_ENV"
echo "  SA_NUM_PROMPTS=2 SA_BASE_MODEL=models/sd-turbo python prune/sensitivity_ld.py"
echo "  BASE_MODEL_ID=models/sd-turbo INFERENCE_TIMESTEPS=999,749,499,249 \\"
echo "    python prune/sp_apply.py --config prune/pruning_config_sdturbo_r1_8pct.json \\"
echo "    --warmup 5 --softmask 5 --rampup 10 --output models/smoke_test_pruned.safetensors"
echo "  BASE_MODEL=models/sd-turbo PRUNED_ST=models/smoke_test_pruned.safetensors \\"
echo "  PRUNED_CFG=models/smoke_test_pruned.config.json python -m prune.gen_test"
echo ""
echo "Full production run:"
echo "  # 1. Sensitivity Analysis (~85 min, 64 prompts from dataset/)"
echo "  SA_BASE_MODEL=models/sd-turbo SA_NUM_PROMPTS=64 python prune/sensitivity_ld.py \\"
echo "    --round 0 --label baseline"
echo ""
echo "  # 2. Pruning (~7 min, pass calib data saved by SA)"
echo "  BASE_MODEL_ID=models/sd-turbo INFERENCE_TIMESTEPS=999,749,499,249 \\"
echo "    python prune/sp_apply.py \\"
echo "      --config prune/pruning_config_sdturbo_r1_8pct.json \\"
echo "      --calib-data prune/calib_data.pt \\"
echo "      --warmup 200 --softmask 200 --rampup 1000 \\"
echo "      --output models/round1_pruned.safetensors --round 1"
echo ""
echo "  # 3. Distillation (~4.5 hr, prompt-only from dataset/*.txt)"
echo "  BASE_MODEL_ID=models/sd-turbo \\"
echo "  PRUNED_SAFETENS_PATH=models/round1_pruned.safetensors \\"
echo "  DATASET_DIR=dataset \\"
echo "  PHASE1_STEPS=15000 PHASE2_STEPS=30000 PHASE3_STEPS=5000 \\"
echo "    python prune/sp_distill.py"
