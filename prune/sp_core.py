"""Shared pruning core — all pruning implementation lives here.

Provides:
  - Zone configuration and loading
  - Module classification helpers
  - Taylor score accumulation, softmask, gradient masking
  - Component pruning (resnet, ffn, attention)
  - Physical zone-based pruning (DG / MetaPruner)
  - taylor_softmask_pipeline() — full Taylor+softmask+recovery orchestrator
  - Calibration dataset, diffusion batch preparation, model I/O
"""
import os
import gc
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

MODELS_DIR = Path("models")
LOCAL_SD_TURBO_DIR = MODELS_DIR / "sd-turbo"

def ensure_local_sd_turbo(local_dir: str = None, repo_id: str = "stabilityai/sd-turbo") -> str:
    """Ensure SD-Turbo exists locally under models/, download from HF if missing."""
    target = Path(local_dir) if local_dir else LOCAL_SD_TURBO_DIR
    target.mkdir(parents=True, exist_ok=True)

    has_model = (target / "model_index.json").exists() or (target / "unet").exists()
    if not has_model:
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading {repo_id} to {target} ...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as e:
            print(f"WARNING: failed to auto-download {repo_id}: {e}")
            print("Falling back to remote repo id for loading.")
            return repo_id
    return str(target)


def resolve_base_model_source(base_model_id: str) -> str:
    """Resolve model source path/id; auto-download SD-Turbo into models/ by default."""
    if not base_model_id:
        return ensure_local_sd_turbo()

    p = Path(base_model_id)
    if p.exists():
        return str(p)

    normalized = base_model_id.replace("\\", "/").lower()
    if normalized in {
        "stabilityai/sd-turbo",
        "sd-turbo",
        "models/sd-turbo",
        "./models/sd-turbo",
    }:
        return ensure_local_sd_turbo()

    return base_model_id


def get_default_zones():
    """Return all zone definitions used by the pruning pipeline.

    Zone hierarchy (innermost → outermost):
      Deep Zone   — mid_block + down_blocks.3 + up_blocks.0  (highest channel count, most prunable)
      Mid Zone    — down_blocks.2 + up_blocks.1              (medium depth)
      Shallow Zone— down_blocks.0/1 + up_blocks.2/3          (low channel count, light pruning)

    Per-zone keys:
      step_ratio      — pruning ratio applied to all target layers
      max_conv_prune  — cap on conv layer ratio (defaults to max_prune_ratio_per_layer arg)
      round_to        — channel-rounding granularity override (defaults to global round_to_val)
    """
    return [
        {
            "name": "Mid Block Zone",
            "keywords": ["mid_block"],
            "step_ratio": 0.45,
            "max_conv_prune": 0.45,
            "ffn_ratio": 0.20,
            "attn_self_ratio": 0.25,     # 2/8 heads
            "attn_cross_ratio": 0.25,    # 2/8 heads
        },
        {
            "name": "Deep Zone",
            "keywords": ["down_blocks.3", "up_blocks.0"],
            "step_ratio": 0.35,
            "max_conv_prune": 0.35,
            # no ffn/attn ratios — these blocks have no transformer layers
        },
        {
            "name": "Mid Zone",
            "keywords": ["down_blocks.2", "up_blocks.1"],
            "step_ratio": 0.20,
            "max_conv_prune": 0.20,
            "ffn_ratio": 0.15,
            "attn_self_ratio": 0.125,    # 1/8 heads
            "attn_cross_ratio": 0.25,    # 2/8 heads
        },
        {
            "name": "Shallow Zone",
            "keywords": ["down_blocks.0", "down_blocks.1", "up_blocks.2", "up_blocks.3"],
            "step_ratio": 0.0,
            "ffn_ratio": 0.0,
            "attn_self_ratio": 0.0,      # no self-attn pruning (sensitive)
            "attn_cross_ratio": 0.125,   # 1/8 heads
        },
    ]


def get_zones_config(zone=None):
    """Return list of zone dicts for the given zone filter.
    zone: one of None|'all'|'mid'|'deep'|'shallow'|'attn'
    """
    if zone is None:
        zone = os.environ.get("TAYLOR_ZONE", "all")
    zone = zone.lower()
    all_zones = get_default_zones()
    if zone == "all":
        return all_zones
    if zone == "mid":
        return [z for z in all_zones if z["name"] == "Mid Zone"]
    if zone == "deep":
        return [z for z in all_zones if z["name"] == "Deep Zone"]
    if zone == "shallow":
        return [z for z in all_zones if z["name"] == "Shallow Zone"]
    if zone == "attn":
        # attention mode currently maps to all zones; pruning code can
        # filter by module names (attention keywords) if desired.
        return all_zones
    return all_zones


def load_zones_from_config(config_path: str):
    """Load zone config from a sensitivity-based pruning config JSON.

    Creates one zone per block to ensure per-block ratios are respected
    (e.g., D3=60% vs U0=30% must NOT be merged into a single zone).

    Returns list of zone dicts compatible with prune_zones().
    """
    with open(config_path) as f:
        config = json.load(f)

    configs = config["configs"]
    config_key = list(configs.keys())[0]
    per_block = configs[config_key]["per_block"]

    # All possible blocks
    all_blocks = [
        "down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
        "mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3",
    ]

    zones = []
    for block in all_blocks:
        if block not in per_block:
            # Protected or not in config — explicit 0 for all ratios
            zones.append({
                "name": block,
                "keywords": [block],
                "step_ratio": 0.0,
                "max_conv_prune": 0.0,
                "ffn_ratio": 0.0,
                "attn_self_ratio": 0.0,
                "attn_cross_ratio": 0.0,
            })
            continue

        block_cfg = per_block[block]
        resnet_ratio = block_cfg.get("resnet", 0.0)
        ffn_ratio = block_cfg.get("ffn", 0.0)
        self_attn_ratio = block_cfg.get("self_attn", 0.0)
        cross_attn_ratio = block_cfg.get("cross_attn", 0.0)

        zone = {
            "name": block,
            "keywords": [block],
            "step_ratio": resnet_ratio,
            "max_conv_prune": resnet_ratio,
            "ffn_ratio": ffn_ratio,
            "attn_self_ratio": self_attn_ratio,
            "attn_cross_ratio": cross_attn_ratio,
        }
        zones.append(zone)

    return zones


# ──────────────────────────────────────────────────────────────
# Per-component pruning (replaces zone-based approach)
# ──────────────────────────────────────────────────────────────

def expand_config_to_components(config_path: str, unet) -> dict:
    """将 per-block config JSON 展开为 {module_name: ratio} 字典。

    对 UNet 的每个可剪枝模块，根据其所属 block 和类型分配比例：
      - resnets.*.conv1        → block 的 resnet ratio
      - ff.net.0.proj (GEGLU)  → block 的 ffn ratio
      - conv2/conv_shortcut    → 跳过（由 DG 级联自动处理）
      - attention 层           → 跳过（Phase 2 已单独处理）

    Returns:
        dict: {module_name_str: pruning_ratio_float}
    """
    with open(config_path) as f:
        config = json.load(f)

    configs = config["configs"]
    config_key = list(configs.keys())[0]
    per_block = configs[config_key]["per_block"]

    component_ratios = {}

    for name, module in unet.named_modules():
        # 只处理 Conv2d conv1 和 GEGLU proj
        if isinstance(module, nn.Conv2d) and name.endswith(".conv1"):
            # 找出所属 block: e.g. "down_blocks.1.resnets.0.conv1" → "down_blocks.1"
            block = _get_block_name(name)
            if block and block in per_block:
                ratio = per_block[block].get("resnet", 0.0)
                if ratio > 0:
                    component_ratios[name] = ratio
        elif isinstance(module, nn.Linear) and "ff.net.0.proj" in name:
            block = _get_block_name(name)
            if block and block in per_block:
                ratio = per_block[block].get("ffn", 0.0)
                if ratio > 0:
                    component_ratios[name] = ratio

    # 打印汇总
    print(f"\n  Component ratios from config ({len(component_ratios)} modules):")
    for n, r in sorted(component_ratios.items()):
        print(f"    {n}: {r:.0%}")

    return component_ratios


def _get_block_name(module_name: str) -> str:
    """从模块名提取所属 block 名。

    例如：
      "down_blocks.1.resnets.0.conv1" → "down_blocks.1"
      "mid_block.resnets.0.conv1"     → "mid_block"
      "up_blocks.2.attentions.1..."   → "up_blocks.2"
    """
    if module_name.startswith("mid_block"):
        return "mid_block"
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] in ("down_blocks", "up_blocks"):
        return f"{parts[0]}.{parts[1]}"
    return None


def prune_components(
    *,
    student_unet,
    pipe,
    component_ratios: dict,
    grad_inputs,
    round_to_val: int = 32,
    taylor_scores: dict = None,
    output_path: str = None,
    device: str = None,
):
    """逐组件物理剪枝，替代 zone-based prune_zones()。

    Args:
        student_unet: 要剪枝的模型（in-place 修改）
        pipe: StableDiffusionPipeline（兼容性）
        component_ratios: {module_name: ratio} 外部传入的逐组件剪枝比例
        grad_inputs: (sample, timestep, encoder_hidden_states) 用于 DG 追踪
        round_to_val: 通道对齐粒度（默认 32）
        taylor_scores: dict mapping nn.Module -> Taylor importance tensor
        output_path: 保存路径
        device: 设备
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if student_unet.is_gradient_checkpointing:
        student_unet.disable_gradient_checkpointing()
    student_unet.eval()

    sync_diffusers_attributes(student_unet)

    # 准备 example inputs
    example_inputs_dict = {
        "sample": grad_inputs[0],
        "timestep": grad_inputs[1],
        "encoder_hidden_states": grad_inputs[2],
    }
    try:
        fix_channels_hook_logic(student_unet, example_inputs_dict)
    except Exception as e:
        print(f"  Channel hook failed: {e}")

    # 构建 DependencyGraph（只构建一次）
    example_tuple = (
        example_inputs_dict["sample"],
        example_inputs_dict["timestep"],
        example_inputs_dict["encoder_hidden_states"],
    )
    try:
        dg = tp.DependencyGraph().build_dependency(student_unet, example_inputs=example_tuple)
    except Exception as e:
        dg = None
        print(f"  DependencyGraph build failed: {e}")

    # 获取 prune ops
    prune_conv_op = None
    if hasattr(tp, "prune_conv"):
        prune_conv_op = tp.prune_conv
    elif hasattr(tp, "prune_conv_out_channels"):
        prune_conv_op = tp.prune_conv_out_channels
    else:
        fn_mod = getattr(getattr(tp, "pruner", None), "function", None)
        if fn_mod and hasattr(fn_mod, "prune_conv_out_channels"):
            prune_conv_op = fn_mod.prune_conv_out_channels

    # 记录剪枝前参数
    params_before = sum(p.numel() for p in student_unet.parameters()) / 1e6
    print(f"\n  Pre-prune total params: {params_before:.2f}M")

    # 记录所有模块剪枝前的 shape
    pre_shapes = {}
    for n, m in student_unet.named_modules():
        if hasattr(m, "weight"):
            pre_shapes[n] = tuple(m.weight.shape)

    named_modules_dict = dict(student_unet.named_modules())

    # 分组：先处理所有 Conv2d conv1（DG 级联），再处理所有 GEGLU（手动）
    conv1_targets = []
    geglu_targets = []
    for comp_name, ratio in sorted(component_ratios.items()):
        module = named_modules_dict.get(comp_name)
        if module is None:
            print(f"  WARNING: Module {comp_name} not found, skipping")
            continue
        if isinstance(module, nn.Conv2d):
            conv1_targets.append((comp_name, module, ratio))
        elif isinstance(module, nn.Linear) and _is_geglu_proj(comp_name):
            geglu_targets.append((comp_name, module, ratio))

    # ── Pass 1: Conv2d conv1 via DependencyGraph ──
    print(f"\n  === Pass 1: Conv2d pruning ({len(conv1_targets)} targets) ===")
    for comp_name, module, ratio in conv1_targets:
        out_ch = module.out_channels
        n_prune = int(out_ch * ratio)
        # 对齐到 round_to_val 的倍数（直接对 n_prune 取整，避免小 ratio 时过度剪枝）
        if round_to_val > 1:
            n_prune = round(n_prune / round_to_val) * round_to_val
            n_prune = max(0, min(n_prune, out_ch - round_to_val))
        if n_prune <= 0:
            continue

        # Taylor or magnitude importance
        if taylor_scores is not None and module in taylor_scores:
            impvals = taylor_scores[module]
        else:
            with torch.no_grad():
                impvals = module.weight.detach().abs().view(module.out_channels, -1).sum(dim=1)

        if n_prune >= impvals.numel():
            idxs = list(range(impvals.numel()))
        else:
            _, idxs_t = torch.topk(impvals, k=n_prune, largest=False)
            idxs = idxs_t.cpu().tolist()

        if dg is not None and prune_conv_op is not None:
            try:
                pruning_group = dg.get_pruning_group(module, prune_conv_op, idxs)
                if pruning_group is None:
                    print(f"    {comp_name}: No pruning group, skipping")
                    continue
                if hasattr(pruning_group, "prune"):
                    pruning_group.prune()
                elif hasattr(pruning_group, "exec"):
                    pruning_group.exec()
                print(f"    {comp_name}: {out_ch} -> {out_ch - n_prune} ({ratio:.0%})")
            except Exception as e:
                print(f"    {comp_name}: DG pruning failed: {e}")
                continue
        else:
            print(f"    {comp_name}: No DG available, skipping conv pruning")

    # ── Pass 2: GEGLU ff.net.0.proj 手动配对剪枝 ──
    print(f"\n  === Pass 2: GEGLU FFN pruning ({len(geglu_targets)} targets) ===")
    named_modules_dict = dict(student_unet.named_modules())  # refresh after conv pruning
    for comp_name, _, ratio in geglu_targets:
        # 重新获取 module（conv pruning 可能改变了模块引用）
        module = named_modules_dict.get(comp_name)
        if module is None:
            print(f"    {comp_name}: Module gone after conv pruning, skipping")
            continue

        out_f = module.out_features
        half = out_f // 2
        n_prune_half = int(half * ratio)
        # 对齐：half 维度按 32 对齐
        round_half = 32
        remaining_half = half - n_prune_half
        rounded_remaining = (remaining_half // round_half) * round_half
        if rounded_remaining < round_half:
            rounded_remaining = round_half
        n_prune_half = half - rounded_remaining
        if n_prune_half <= 0:
            continue

        # Taylor or magnitude (paired)
        if taylor_scores is not None and module in taylor_scores:
            impvals_all = taylor_scores[module]
        else:
            with torch.no_grad():
                impvals_all = module.weight.detach().abs().view(module.out_features, -1).sum(dim=1)
        impvals_paired = impvals_all[:half] + impvals_all[half:]

        if n_prune_half >= impvals_paired.numel():
            half_idxs = list(range(impvals_paired.numel()))
        else:
            _, idxs_t = torch.topk(impvals_paired, k=n_prune_half, largest=False)
            half_idxs = idxs_t.cpu().tolist()

        # 展开为配对 indices
        idxs = []
        for i in half_idxs:
            idxs.append(i)
            idxs.append(i + half)

        try:
            _manual_prune_ffn(student_unet, comp_name, module, idxs)
            named_modules_dict = dict(student_unet.named_modules())
            new_module = named_modules_dict.get(comp_name)
            new_out = new_module.out_features if new_module else "?"
            print(f"    {comp_name}: {out_f} -> {new_out} ({ratio:.0%})")
        except Exception as e:
            print(f"    {comp_name}: Manual FFN pruning failed: {e}")
            continue

    # ── Cascade Report ──
    print(f"\n  === All Dimension Changes ===")
    attn_modified = False
    for n, m in student_unet.named_modules():
        if hasattr(m, "weight") and n in pre_shapes:
            new_shape = tuple(m.weight.shape)
            if new_shape != pre_shapes[n]:
                tag = ""
                if _is_attention_name(n):
                    tag = " [!ATTENTION CASCADE]"
                    attn_modified = True
                elif "time_emb_proj" in n:
                    tag = " [time_emb cascade]"
                elif "norm" in n:
                    tag = " [norm cascade]"
                print(f"    {n}: {pre_shapes[n]} -> {new_shape}{tag}")
    if attn_modified:
        print("  WARNING: Attention layers were modified by DG cascade!")
    else:
        print("  OK: No attention layers affected by cascade.")

    # ── Final stats ──
    params_after = sum(p.numel() for p in student_unet.parameters()) / 1e6
    removed = params_before - params_after
    print(f"\n  Post-prune total: {params_after:.2f}M  (removed {removed:.2f}M, {removed/params_before*100:.1f}%)")

    # Save
    if output_path:
        save_checkpoint(student_unet, output_path)


def summarize_zone_names(zones):
    return [z.get("name") for z in (zones or [])]


# ---------------- Shared heavy helpers ----------------

def fix_channels_hook_logic(unet, example_inputs):
    def fix_channels_hook(module, input):
        if module.__class__.__name__ == "Upsample2D":
            actual_channels = input[0].shape[1]
            if getattr(module, "channels", None) != actual_channels:
                setattr(module, "channels", actual_channels)
    hooks = []
    for module in unet.modules():
        if module.__class__.__name__ == "Upsample2D":
            hooks.append(module.register_forward_pre_hook(fix_channels_hook))
    with torch.no_grad():
        try:
            if isinstance(example_inputs, dict):
                unet(**example_inputs)
            else:
                sample, timestep, encoder_hidden_states = example_inputs
                unet(sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states)
        except Exception:
            pass
    for h in hooks:
        h.remove()


def sync_diffusers_attributes(student_unet):
    for m_name, module in student_unet.named_modules():
        if hasattr(module, "channels") and hasattr(module, "conv"):
            module.channels = module.conv.weight.shape[1]
        if hasattr(module, "in_channels") and hasattr(module, "conv1"):
            module.in_channels = module.conv1.weight.shape[1]
        if hasattr(module, "out_channels") and hasattr(module, "conv2"):
            module.out_channels = module.conv2.weight.shape[0]
        if hasattr(module, "to_q") and hasattr(module, "inner_dim"):
            if hasattr(module.to_q, "weight"):
                module.inner_dim = module.to_q.weight.shape[0]
                if hasattr(module, 'inner_kv_dim'):
                    module.inner_kv_dim = module.to_q.weight.shape[0]


def _print_table(title, headers, rows):
    if not rows:
        print(f"   {title}: (no rows)")
        return
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            l = len(str(c))
            if l > col_widths[i]:
                col_widths[i] = l
    print("   " + title)
    header_line = "   " + "   ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("   " + "   ".join("-" * col_widths[i] for i in range(len(headers))))
    for r in rows:
        line = "   " + "   ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers)))
        print(line)
    print()


ATTN_KEYWORDS = ["attn", "attention", "attentions", "to_q", "to_k", "to_v", "proj_in", "proj_out", "out_proj", "context"]
FFN_KEYWORDS = ["ff", "ffn", "feed_forward", "mlp"]


def _is_attention_name(name: str) -> bool:
    lname = name.lower()
    if any(k in lname for k in FFN_KEYWORDS):
        return False
    return any(k in lname for k in ATTN_KEYWORDS)


def _is_ffn_linear(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lname = name.lower()
    return any(k in lname for k in FFN_KEYWORDS)


def _is_geglu_proj(name: str) -> bool:
    """GEGLU projection 层：ff.net.0.proj, out_features = inner_dim × 2.
    chunk(2) 会将输出一分为二，所以 round_to 必须是 64 (保证 half 是 32 对齐)。"""
    return "ff.net.0.proj" in name


def _is_ffn_out_proj(name: str) -> bool:
    """FFN 输出层：ff.net.2，其 in_features 与 GEGLU chunk(2) 的 half 维度耦合。
    在 DG 路径中不应独立剪枝，应由 GEGLU proj 的剪枝通过手动剪枝调整。"""
    return "ff.net.2" in name and "ff.net.0" not in name


def _get_parent_and_attr(model: nn.Module, dotted_name: str):
    """Return (parent_module, attr_name) for setattr replacement."""
    parts = dotted_name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def _manual_prune_ffn(student_unet, proj_name, proj_module, idxs_to_remove):
    """Manually prune ff.net.0.proj out_features and ff.net.2 in_features.

    Does NOT use DG, so no cascade to attention/proj_in/proj_out via residual stream.

    GEGLU layout: ff.net.0.proj outputs inner_dim*2, chunk(2) splits into [gate, value].
    When removing output index i from first half, must also remove i+half (paired).
    ff.net.2.in_features = half, corresponding to the first-half indices.
    """
    # 1. Derive ff.net.2 name
    # proj_name: "...ff.net.0.proj" → ff_base: "...ff.net"
    ff_base = proj_name.rsplit(".net.0.proj", 1)[0] + ".net"
    ff_out_name = ff_base + ".2"

    named_modules = dict(student_unet.named_modules())
    ff_out_module = named_modules.get(ff_out_name)
    if ff_out_module is None:
        print(f"   [ManualFFN] WARNING: Cannot find {ff_out_name}, skipping manual FFN prune for {proj_name}")
        return False

    # 2. Compute keep indices for ff.net.0.proj
    old_out = proj_module.out_features
    half = old_out // 2
    all_indices = set(range(old_out))
    remove_set = set(idxs_to_remove)

    # Ensure paired removal: if removing i < half, also remove i+half; if removing i >= half, also remove i-half
    paired_remove = set()
    for idx in remove_set:
        paired_remove.add(idx)
        if idx < half:
            paired_remove.add(idx + half)
        else:
            paired_remove.add(idx - half)

    keep_indices = sorted(all_indices - paired_remove)
    if len(keep_indices) == 0:
        print(f"   [ManualFFN] WARNING: Would remove all channels from {proj_name}, skipping")
        return False

    keep_tensor = torch.tensor(keep_indices, device=proj_module.weight.device)

    # 3. Prune ff.net.0.proj (out_features)
    new_out = len(keep_indices)
    new_proj = nn.Linear(proj_module.in_features, new_out, bias=proj_module.bias is not None)
    new_proj.weight = nn.Parameter(proj_module.weight.data[keep_tensor, :])
    if proj_module.bias is not None:
        new_proj.bias = nn.Parameter(proj_module.bias.data[keep_tensor])
    new_proj = new_proj.to(device=proj_module.weight.device, dtype=proj_module.weight.dtype)

    parent, attr = _get_parent_and_attr(student_unet, proj_name)
    setattr(parent, attr, new_proj)

    # 4. Prune ff.net.2 (in_features) — only first-half indices map to ff.net.2 input
    kept_first_half = sorted([i for i in keep_indices if i < half])
    keep_in_tensor = torch.tensor(kept_first_half, device=ff_out_module.weight.device)
    new_in = len(kept_first_half)

    new_ff_out = nn.Linear(new_in, ff_out_module.out_features, bias=ff_out_module.bias is not None)
    new_ff_out.weight = nn.Parameter(ff_out_module.weight.data[:, keep_in_tensor])
    if ff_out_module.bias is not None:
        new_ff_out.bias = nn.Parameter(ff_out_module.bias.data.clone())
    new_ff_out = new_ff_out.to(device=ff_out_module.weight.device, dtype=ff_out_module.weight.dtype)

    parent2, attr2 = _get_parent_and_attr(student_unet, ff_out_name)
    setattr(parent2, attr2, new_ff_out)

    print(f"   [ManualFFN] {proj_name}: out {old_out} -> {new_out}, {ff_out_name}: in {half} -> {new_in}")
    return True


def _is_prunable_module(name: str, module: nn.Module) -> bool:
    if isinstance(module, nn.Conv2d):
        return True
    if _is_ffn_linear(name, module):
        return True
    return False


def _is_interface_conv(name: str) -> bool:
    """
    判断是否为 ResNet block 的"接口层"（输出通道受残差/跨块连接约束）。

    这类层的输出维度不应被直接剪枝，否则会通过 DependencyGraph
    传播到下游的 downsampler/upsampler，导致级联式的过度剪枝。
    只剪 conv1（瓶颈内部通道），让 DG 自动调整 conv2 的输入即可。

    接口层包括：
      - resnets.*.conv2        — ResNet 块的输出卷积
      - resnets.*.conv_shortcut — 残差跳连卷积
    """
    last = name.split('.')[-1]
    return last in ('conv2', 'conv_shortcut')


def _get_module_cap(name: str):
    """Per-block safety cap (ceiling). Zone config sets actual ratio, this prevents accidents."""
    # Deep zone: generous cap
    if "down_blocks.3" in name or "up_blocks.0" in name:
        return 0.65
    # Mid zone: allow up to 50%
    if "down_blocks.2" in name or "up_blocks.1" in name:
        return 0.50
    # Shallow Inner (down_1/up_2): safety cap above 8% zone ratio
    if "down_blocks.1" in name or "up_blocks.2" in name:
        return 0.15
    # Shallow Outer (down_0/up_3): very strict
    if "down_blocks.0" in name or "up_blocks.3" in name:
        return 0.05
    return None


def _is_sampler_module(name: str) -> bool:
    return ("downsamplers" in name) or ("upsamplers" in name)


def _refresh_grads_for_taylor(student_unet, example_inputs_dict):
    student_unet.zero_grad(set_to_none=True)
    out = student_unet(**example_inputs_dict).sample
    loss = out.float().mean()
    loss.backward()


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_checkpoint(model, path: str):
    """Save pruned model as safetensors + config.json (no .pt)."""
    from safetensors.torch import save_file
    ensure_dir(path)
    try:
        model_cpu = model.cpu()
        # Fix Upsampler.channels to match actual conv.in_channels
        for name, module in model_cpu.named_modules():
            if 'upsampler' in name.lower():
                if hasattr(module, 'channels') and hasattr(module, 'conv'):
                    if hasattr(module.conv, 'in_channels'):
                        actual_channels = module.conv.in_channels
                        if module.channels != actual_channels:
                            print(f"  修复 {name}: channels {module.channels} → {actual_channels}")
                            module.channels = actual_channels

        # Base path for .safetensors and .config.json
        base_path = path.rsplit('.', 1)[0] if '.' in path else path

        # 1. Save safetensors
        safetensors_path = base_path + '.safetensors'
        save_file(model_cpu.state_dict(), safetensors_path)
        param_count = sum(p.numel() for p in model_cpu.parameters())
        st_size = os.path.getsize(safetensors_path) / (1024**3) if os.path.exists(safetensors_path) else 0
        print(f"✓ Safetensors已保存: {safetensors_path}")
        print(f"  参数量: {param_count/1e6:.1f}M, 文件大小: {st_size:.2f}GB")

        # 2. Save config metadata
        cfg_path = base_path + '.config.json'
        metadata = {}
        try:
            if hasattr(model_cpu, 'config') and model_cpu.config is not None:
                try:
                    metadata['model_config'] = model_cpu.config.to_dict() if hasattr(model_cpu.config, 'to_dict') else dict(model_cpu.config)
                except Exception:
                    metadata['model_config'] = str(model_cpu.config)
        except Exception:
            metadata['model_config'] = None

        module_shapes = {}
        for name, module in model_cpu.named_modules():
            try:
                if isinstance(module, nn.Conv2d):
                    module_shapes[name] = {
                        'type': 'conv2d',
                        'out_channels': module.out_channels,
                        'in_channels': module.in_channels,
                        'kernel_size': list(module.kernel_size) if hasattr(module, 'kernel_size') else None,
                    }
                elif isinstance(module, nn.Linear):
                    module_shapes[name] = {
                        'type': 'linear',
                        'out_features': module.out_features,
                        'in_features': module.in_features,
                    }
            except Exception:
                continue
        metadata['module_shapes'] = module_shapes
        metadata['torch_version'] = torch.__version__
        try:
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            cfg_size = os.path.getsize(cfg_path) / 1024.0 if os.path.exists(cfg_path) else 0
            print(f"✓ 配置元数据已保存: {cfg_path} ({cfg_size:.1f}KB)")
        except Exception as e:
            print(f"保存配置元数据失败: {e}")

    except Exception as e:
        print(f"Failed to save checkpoint {path}: {e}")


def round_to_multiple(n: int, multiple: int) -> int:
    if multiple <= 0:
        return n
    return int((n + multiple - 1) // multiple * multiple)


def simple_log(path: str, message: str):
    ensure_dir(path)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


def prepare_diffusion_batch(pipe, pixel_values, prompts, device, model_dtype):
    """Prepare a diffusion training batch.

    Returns: (clean_latents, encoder_hidden_states, noise, timesteps, noisy_latents)
    """
    pixel_values = pixel_values.to(device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        clean_latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215
        clean_latents = clean_latents.to(dtype=model_dtype)
        text_input = pipe.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        encoder_hidden_states = pipe.text_encoder(text_input.input_ids)[0].to(dtype=model_dtype)

    noise = torch.randn_like(clean_latents, dtype=model_dtype)
    bsz = clean_latents.shape[0]
    num_train_steps = pipe.scheduler.config.num_train_timesteps
    _inference_ts = os.environ.get('INFERENCE_TIMESTEPS', '')
    if _inference_ts:
        _ts_list = [int(t) for t in _inference_ts.split(',')]
        _idx = torch.randint(0, len(_ts_list), (bsz,))
        timesteps = torch.tensor([_ts_list[i] for i in _idx], device=device).long()
    else:
        timesteps = torch.randint(0, num_train_steps, (bsz,), device=device).long()
    noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timesteps).to(dtype=model_dtype)

    return clean_latents, encoder_hidden_states, noise, timesteps, noisy_latents


def load_pipeline_and_student(base_model_id: str, device: str = None,
                              pruned_st: str = None, pruned_cfg: str = None):
    """Load teacher pipeline + student UNet.

    If pruned_st/pruned_cfg are given, the student is rebuilt from the
    previous round's safetensors (iterative pruning).  The teacher is
    always the original SD1.5.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_model_id = resolve_base_model_source(base_model_id)
    print(f"Loading teacher pipeline {base_model_id} to {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    ).to(device)

    if pruned_st and pruned_cfg:
        from prune.pruned_rebuild import create_unet_from_safetensors
        print(f"Loading pruned student from: {pruned_st}")
        student_unet = create_unet_from_safetensors(pruned_st, pruned_cfg)
        student_unet = student_unet.to(device=device, dtype=torch.float32)
        params_m = sum(p.numel() for p in student_unet.parameters()) / 1e6
        print(f"  Student UNet: {params_m:.1f}M params (pruned)")
    else:
        student_unet = UNet2DConditionModel.from_pretrained(
            base_model_id, subfolder="unet", torch_dtype=torch.float32
        ).to(device)

    student_unet.train()
    student_unet.enable_gradient_checkpointing()
    return pipe, student_unet


def prune_zones(*, student_unet, pipe, zones, grad_inputs, importance=None, round_to_val: int = 32, allow_conv_shortcut_prune: bool = True, ffn_max_prune: float = 0.10, max_prune_ratio_per_layer: float = 0.15, output_path: str = None, device: str = None, taylor_scores: dict = None):
    """Generic per-zone physical pruning using MetaPruner.

    - `student_unet`: model to prune (in-place)
    - `pipe`: pipeline (used for scheduler/vae/tokenizer compatibility)
    - `zones`: list of zone dicts
    - `grad_inputs`: tuple(sample, timestep, encoder_hidden_states) for tracing
    - `importance`: a torch_pruning importance instance (if None, MagnitudeImportance is used)
    - `ffn_max_prune`: maximum pruning ratio for FFN layers (default 0.10 = 10%)
    - `max_prune_ratio_per_layer`: maximum pruning ratio for conv layers (default 0.15 = 15%)
    - `taylor_scores`: dict mapping nn.Module -> per-channel Taylor scores tensor.
      If provided, Taylor scores are used for channel selection instead of magnitude.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if importance is None:
        importance = tp.importance.MagnitudeImportance(p=1, group_reduction='mean', normalizer=None)

    if student_unet.is_gradient_checkpointing:
        student_unet.disable_gradient_checkpointing()
    student_unet.eval()

    for zone in zones:
        ratio = zone.get("step_ratio", 0.0)
        mid_block_ratio = zone.get("mid_block_ratio", ratio)
        keywords = zone.get("keywords", [])
        zone_round_to = zone.get("round_to", round_to_val)          # per-zone 粒度覆盖
        zone_max_conv = zone.get("max_conv_prune", max_prune_ratio_per_layer)  # per-zone conv cap
        print(f"\n   [Zone: {zone.get('name')}] Base pruning ratio={ratio}, conv cap={zone_max_conv}, round_to={zone_round_to}...")

        sync_diffusers_attributes(student_unet)
        try:
            fix_channels_hook_logic(student_unet, {"sample": grad_inputs[0], "timestep": grad_inputs[1], "encoder_hidden_states": grad_inputs[2]})
        except Exception as e:
            print(f"Channel hook failed: {e}")

        named_modules_dict = dict(student_unet.named_modules())
        zone_module_names = [
            name for name, module in named_modules_dict.items()
            if any(k in name for k in keywords)
            and not _is_attention_name(name)
            and _is_prunable_module(name, module)
            and not _is_interface_conv(name)   # 排除 conv2/conv_shortcut 接口层，避免级联过度剪枝
        ]

        if not zone_module_names:
            print(f"   [Zone: {zone.get('name')}] No modules found, skipping...")
            continue

        pre_channels = {}
        applied_ratios = {}
        for name in zone_module_names:
            module = named_modules_dict.get(name)
            if isinstance(module, nn.Conv2d):
                pre_channels[name] = module.out_channels
            elif isinstance(module, nn.Linear):
                pre_channels[name] = module.out_features

        pre_params_by_module = {nm: sum(p.numel() for p in named_modules_dict[nm].parameters()) for nm in zone_module_names if nm in named_modules_dict}
        pre_params = sum(pre_params_by_module.values())
        print(f"   [Zone: {zone.get('name')}] Pre-prune params: {pre_params/1e6:.2f}M")

        example_inputs_dict = {"sample": grad_inputs[0], "timestep": grad_inputs[1], "encoder_hidden_states": grad_inputs[2]}

        class UNetWrapper(nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet
            def forward(self, sample, timestep, encoder_hidden_states):
                return self.unet(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states)

        wrapper = UNetWrapper(student_unet)
        try:
            with torch.no_grad():
                _ = wrapper(**example_inputs_dict)
        except Exception as e:
            print(f"   [Zone: {zone.get('name')}] Forward validation failed: {e}")
            continue

        pruning_ratio_dict = {}
        for name, module in named_modules_dict.items():
            if _is_sampler_module(name):
                pruning_ratio_dict[module] = 0.0
                continue
            if (not allow_conv_shortcut_prune) and ("conv_shortcut" in name):
                pruning_ratio_dict[module] = 0.0
                continue
            if any(k in name for k in ["to_out", "proj_out", "out_proj"]):
                pruning_ratio_dict[module] = 0.0
                continue
            if _is_attention_name(name):
                pruning_ratio_dict[module] = 0.0
                continue
            if name in zone_module_names and _is_prunable_module(name, module):
                effective_ratio = mid_block_ratio if "mid_block" in name else ratio
                if isinstance(module, nn.Conv2d):
                    applied_ratio = min(effective_ratio, zone_max_conv)
                    pruning_ratio_dict[module] = applied_ratio
                    applied_ratios[name] = applied_ratio
                else:
                    # FFN: use per-zone ffn_ratio if available, else fall back to global ffn_max_prune
                    zone_ffn = zone.get("ffn_ratio", ffn_max_prune)
                    effective = zone_ffn
                    cap = _get_module_cap(name)
                    if cap is not None and cap < effective:
                        print(f"   [Zone: {zone.get('name')}] _get_module_cap override: {name} capped {effective:.0%} -> {cap:.0%}")
                        effective = cap
                    pruning_ratio_dict[module] = effective
                    applied_ratios[name] = effective
            else:
                pruning_ratio_dict[module] = 0.0

        current_ignored = []
        for name, module in named_modules_dict.items():
            if any(k in name for k in ["time", "time_embed", "time_embedding", "time_proj"]):
                current_ignored.append(module)
            if _is_sampler_module(name):
                current_ignored.append(module)

        example_tuple = (example_inputs_dict["sample"], example_inputs_dict["timestep"], example_inputs_dict["encoder_hidden_states"])
        try:
            dg = tp.DependencyGraph().build_dependency(student_unet, example_inputs=example_tuple)
        except Exception as e:
            dg = None
            print(f"   [Zone: {zone.get('name')}] DependencyGraph build failed: {e} — falling back to MetaPruner")

        conv_ops = []
        if hasattr(tp, "prune_conv"):
            conv_ops.append(tp.prune_conv)
        if hasattr(tp, "prune_conv_out_channels"):
            conv_ops.append(tp.prune_conv_out_channels)
        pf_mod = getattr(tp, "pruner", None)
        if pf_mod is not None and hasattr(pf_mod, "function") and hasattr(pf_mod.function, "prune_conv_out_channels"):
            conv_ops.append(pf_mod.function.prune_conv_out_channels)

        linear_ops = []
        if hasattr(tp, "prune_linear"):
            linear_ops.append(tp.prune_linear)
        if hasattr(tp, "prune_linear_out_channels"):
            linear_ops.append(tp.prune_linear_out_channels)
        if pf_mod is not None and hasattr(pf_mod, "function") and hasattr(pf_mod.function, "prune_linear_out_channels"):
            linear_ops.append(pf_mod.function.prune_linear_out_channels)

        use_dg = (dg is not None) and (len(conv_ops) > 0) and (len(linear_ops) > 0)
        if not use_dg:
            # WARNING: MetaPruner 使用全局 round_to，无法对 GEGLU 层单独设 64。
            # GEGLU 层在此路径下可能产生未对齐维度。优先使用 DG 路径。
            has_geglu = any(_is_geglu_proj(n) for n in zone_module_names)
            if has_geglu:
                print(f"   [Zone: {zone.get('name')}] WARNING: MetaPruner fallback with GEGLU layers — round_to=64 not enforced")
            try:
                pruner = tp.pruner.MetaPruner(
                    wrapper,
                    example_inputs=tuple(example_inputs_dict.values()),
                    importance=importance,
                    pruning_ratio_dict=pruning_ratio_dict,
                    global_pruning=False,
                    ignored_layers=current_ignored,
                    round_to=zone_round_to,
                )
                _refresh_grads_for_taylor(student_unet, example_inputs_dict)
                pruner.step()
            except Exception as e:
                print(f"   [Zone: {zone.get('name')}] MetaPruner failed: {e}")
                continue
        else:
            # Capture ALL module shapes before pruning for cascade reporting
            pre_shapes = {}
            for n, m in student_unet.named_modules():
                if hasattr(m, 'weight'):
                    pre_shapes[n] = tuple(m.weight.shape)

            for name in zone_module_names:
                module = named_modules_dict.get(name)
                if module is None:
                    continue
                if not _is_prunable_module(name, module):
                    continue
                # 跳过 ff.net.2 的独立剪枝，由 ff.net.0.proj 的手动剪枝同步调整
                if _is_ffn_out_proj(name):
                    continue
                ratio_m = pruning_ratio_dict.get(module, 0.0)
                if ratio_m <= 0:
                    continue

                if isinstance(module, nn.Conv2d):
                    # Conv1: 使用 DG 剪枝（级联到 conv2 + norm2 + time_emb_proj，可接受）
                    out_ch = module.out_channels
                    n_prune = int(out_ch * float(ratio_m))
                    if zone_round_to > 1:
                        n_prune = round(n_prune / zone_round_to) * zone_round_to
                        n_prune = max(0, min(n_prune, out_ch - zone_round_to))
                    if n_prune <= 0:
                        continue
                    if taylor_scores is None or module not in taylor_scores:
                        raise RuntimeError(
                            f"[Zone: {zone.get('name')}] No Taylor score for Conv2d {name} — "
                            f"Taylor phase did not cover this module. Aborting prune."
                        )
                    impvals = taylor_scores[module]
                    if n_prune >= impvals.numel():
                        idxs = list(range(impvals.numel()))
                    else:
                        _, idxs_t = torch.topk(impvals, k=n_prune, largest=False)
                        idxs = idxs_t.cpu().tolist()
                    if hasattr(tp, "prune_conv"):
                        prune_op = tp.prune_conv
                    elif hasattr(tp, "prune_conv_out_channels"):
                        prune_op = tp.prune_conv_out_channels
                    else:
                        fn_mod = getattr(getattr(tp, "pruner", None), "function", None)
                        prune_op = fn_mod.prune_conv_out_channels if fn_mod is not None else None

                    if prune_op is None:
                        print(f"   [Zone: {zone.get('name')}] No prune op for {name}, skipping")
                        continue
                    try:
                        pruning_group = dg.get_pruning_group(module, prune_op, idxs)
                        if pruning_group is None:
                            print(f"   [Zone: {zone.get('name')}] No pruning group for {name}, skipping")
                            continue
                        if hasattr(pruning_group, 'prune'):
                            pruning_group.prune()
                        elif hasattr(pruning_group, 'exec'):
                            pruning_group.exec()
                        else:
                            print(f"   [Zone: {zone.get('name')}] Unknown pruning group type for {name}, skipping")
                            continue
                    except Exception as e:
                        print(f"   [Zone: {zone.get('name')}] DG pruning for {name} failed: {e}")
                        continue

                elif isinstance(module, nn.Linear) and _is_geglu_proj(name):
                    # FFN GEGLU proj: 手动剪枝，不使用 DG，避免级联到 attention 层
                    # GEGLU out_features = inner_dim * 2, chunk(2) 分成 gate 和 value
                    # 必须按 pair 剪枝（i 和 i+half），所以基于 half 维度计算
                    out_f = module.out_features
                    half = out_f // 2
                    n_prune_half = int(half * float(ratio_m))
                    # round_to=32 for half dimension (ensures full dim is 64-aligned)
                    round_half = 32
                    if round_half > 1:
                        remaining_half = half - n_prune_half
                        rounded_remaining = (remaining_half // round_half) * round_half
                        if rounded_remaining < round_half:
                            rounded_remaining = round_half
                        n_prune_half = half - rounded_remaining
                    if n_prune_half <= 0:
                        continue
                    if taylor_scores is None or module not in taylor_scores:
                        raise RuntimeError(
                            f"[Zone: {zone.get('name')}] No Taylor score for GEGLU Linear {name} — "
                            f"Taylor phase did not cover this module. Aborting prune."
                        )
                    impvals_all = taylor_scores[module]
                    impvals_paired = impvals_all[:half] + impvals_all[half:]
                    if n_prune_half >= impvals_paired.numel():
                        half_idxs = list(range(impvals_paired.numel()))
                    else:
                        _, idxs_t = torch.topk(impvals_paired, k=n_prune_half, largest=False)
                        half_idxs = idxs_t.cpu().tolist()
                    # Expand to paired indices (i and i+half)
                    idxs = []
                    for i in half_idxs:
                        idxs.append(i)
                        idxs.append(i + half)
                    try:
                        _manual_prune_ffn(student_unet, name, module, idxs)
                        # Refresh named_modules_dict after manual replacement
                        named_modules_dict = dict(student_unet.named_modules())
                    except Exception as e:
                        print(f"   [Zone: {zone.get('name')}] Manual FFN pruning for {name} failed: {e}")
                        continue

                elif isinstance(module, nn.Linear):
                    # Other Linear: 使用 DG 剪枝
                    out_f = module.out_features
                    n_prune = int(out_f * float(ratio_m))
                    if zone_round_to > 1:
                        n_prune = round(n_prune / zone_round_to) * zone_round_to
                        n_prune = max(0, min(n_prune, out_f - zone_round_to))
                    if n_prune <= 0:
                        continue
                    if taylor_scores is None or module not in taylor_scores:
                        raise RuntimeError(
                            f"[Zone: {zone.get('name')}] No Taylor score for Linear {name} — "
                            f"Taylor phase did not cover this module. Aborting prune."
                        )
                    impvals = taylor_scores[module]
                    if n_prune >= impvals.numel():
                        idxs = list(range(impvals.numel()))
                    else:
                        _, idxs_t = torch.topk(impvals, k=n_prune, largest=False)
                        idxs = idxs_t.cpu().tolist()
                    if hasattr(tp, "prune_linear"):
                        prune_op = tp.prune_linear
                    elif hasattr(tp, "prune_linear_out_channels"):
                        prune_op = tp.prune_linear_out_channels
                    else:
                        fn_mod = getattr(getattr(tp, "pruner", None), "function", None)
                        prune_op = fn_mod.prune_linear_out_channels if fn_mod is not None else None

                    if prune_op is None:
                        print(f"   [Zone: {zone.get('name')}] No prune op for {name}, skipping")
                        continue
                    try:
                        pruning_group = dg.get_pruning_group(module, prune_op, idxs)
                        if pruning_group is None:
                            print(f"   [Zone: {zone.get('name')}] No pruning group for {name}, skipping")
                            continue
                        if hasattr(pruning_group, 'prune'):
                            pruning_group.prune()
                        elif hasattr(pruning_group, 'exec'):
                            pruning_group.exec()
                        else:
                            print(f"   [Zone: {zone.get('name')}] Unknown pruning group type for {name}, skipping")
                            continue
                    except Exception as e:
                        print(f"   [Zone: {zone.get('name')}] DG pruning for {name} failed: {e}")
                        continue
                else:
                    continue

            # === Cascade Report: show ALL dimension changes ===
            print(f"\n   All Dimension Changes for {zone.get('name')}:")
            attn_modified = False
            for n, m in student_unet.named_modules():
                if hasattr(m, 'weight') and n in pre_shapes:
                    new_shape = tuple(m.weight.shape)
                    if new_shape != pre_shapes[n]:
                        tag = ""
                        if _is_attention_name(n):
                            tag = " [!ATTENTION CASCADE]"
                            attn_modified = True
                        elif "time_emb_proj" in n:
                            tag = " [time_emb cascade]"
                        elif "norm" in n:
                            tag = " [norm cascade]"
                        print(f"     {n}: {pre_shapes[n]} -> {new_shape}{tag}")
            if attn_modified:
                print(f"   WARNING: Attention layers were modified by DG cascade in {zone.get('name')}!")
            else:
                print(f"   OK: No attention layers affected by cascade.")

        named_modules_after = dict(student_unet.named_modules())
        post_params_by_module = {nm: sum(p.numel() for p in named_modules_after[nm].parameters()) for nm in zone_module_names if nm in named_modules_after}
        post_params = sum(post_params_by_module.values())
        delta = pre_params - post_params
        print(f"   [Zone: {zone.get('name')}] Post-prune params: {post_params/1e6:.2f}M  removed {delta/1e6:.2f}M")

        print(f"\n   Per-Layer Pruning Details for {zone.get('name')}:")
        print(f"   {'Layer Name':<60} {'Original':<10} {'Pruned':<10} {'Target%':<10} {'Actual%':<10}")
        print(f"   {'-'*100}")

        for name in sorted(zone_module_names):
            if name in pre_channels and name in named_modules_after:
                pruned_mod = named_modules_after[name]
                orig_ch = pre_channels[name]
                if isinstance(pruned_mod, nn.Conv2d):
                    pruned_ch = pruned_mod.out_channels
                elif isinstance(pruned_mod, nn.Linear):
                    pruned_ch = pruned_mod.out_features
                else:
                    continue
                if orig_ch != pruned_ch:
                    actual_ratio = (orig_ch - pruned_ch) / orig_ch * 100
                    target_ratio = applied_ratios.get(name, 0.0) * 100
                    print(f"   {name:<60} {orig_ch:<10} {pruned_ch:<10} {target_ratio:>6.2f}%   {actual_ratio:>6.2f}%")

    # Save the pruned model
    if output_path:
        save_checkpoint(student_unet, output_path)


# ──────────────────────────────────────────────────────────────
# Calibration dataset
# ──────────────────────────────────────────────────────────────

FIXED_TIMESTEPS = [999, 749, 499, 249]


def generate_calibration_data(pipe, dataset_dir, device, max_samples=64):
    """Run teacher 4-step SD-Turbo inference on prompts from dataset_dir.

    Each sample contains:
      noisy_latents, latents_in, timesteps, step_index,
      encoder_hidden_states, teacher_pred, teacher_next_latent

    This matches the format produced by sensitivity_ld.generate_calibration_data
    so both scripts share the same calib_data schema.
    """
    import glob as _glob
    scheduler = pipe.scheduler
    unet = pipe.unet
    dtype = next(unet.parameters()).dtype

    txt_paths = sorted(_glob.glob(os.path.join(dataset_dir, "*.txt")))
    if not txt_paths:
        raise ValueError(f"No .txt prompt files found in {dataset_dir}")
    if len(txt_paths) > max_samples:
        random.seed(42)
        random.shuffle(txt_paths)
        txt_paths = txt_paths[:max_samples]

    prompts = []
    for p in txt_paths:
        with open(p, "r", encoding="utf-8") as f:
            prompts.append(f.read().strip())

    data = []
    print(f"  Generating calib data: {len(prompts)} prompts × 4 timesteps = {len(prompts)*4} samples")

    unet.eval()
    for prompt in tqdm(prompts, desc="  Calib (teacher inference)"):
        with torch.no_grad():
            text_input = pipe.tokenizer(
                [prompt], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            ).to(device)
            enc_hs = pipe.text_encoder(text_input.input_ids)[0].to(dtype=dtype)

            scheduler.set_timesteps(4, device=device)
            latents = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)
            latents = latents * scheduler.init_noise_sigma

            for step_idx, t in enumerate(scheduler.timesteps):
                latents_in = latents.clone()
                scaled = scheduler.scale_model_input(latents, t).to(dtype=dtype)
                teacher_pred = unet(scaled, t, encoder_hidden_states=enc_hs).sample
                scheduler._step_index = step_idx
                teacher_next_latent = scheduler.step(teacher_pred, t, latents_in).prev_sample

                data.append({
                    'noisy_latents': scaled.detach(),
                    'latents_in': latents_in.detach(),
                    'timesteps': t.unsqueeze(0),
                    'step_index': step_idx,
                    'encoder_hidden_states': enc_hs.detach(),
                    'teacher_pred': teacher_pred.detach(),
                    'teacher_next_latent': teacher_next_latent.detach().float(),
                })
                latents = teacher_next_latent

    print(f"  Calib data ready: {len(data)} samples")
    return data


class CalibrationDataset(Dataset):
    """Prompt-only dataset. SD-Turbo generates from pure noise, so images are never used."""

    def __init__(self, root_dir, max_samples=64):
        self.root_dir = root_dir
        all_txts = sorted(glob.glob(os.path.join(root_dir, "*.txt")))
        if len(all_txts) == 0:
            raise ValueError(f"No .txt prompt files found in {root_dir}")
        if len(all_txts) > max_samples:
            random.seed(42)
            random.shuffle(all_txts)
            self.txt_paths = all_txts[:max_samples]
        else:
            self.txt_paths = all_txts

    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, idx):
        with open(self.txt_paths[idx], "r", encoding="utf-8") as f:
            return f.read().strip()


# ──────────────────────────────────────────────────────────────
# Taylor score accumulation & softmask utilities
# ──────────────────────────────────────────────────────────────

def accumulate_taylor_scores(score_accum, module, weight, grad, mode="avg"):
    """Accumulate per-channel Taylor importance: |weight * grad|.

    mode="avg": sum scores (caller divides by count later).
    mode="max": keep element-wise max across calls.
    """
    if weight is None or grad is None:
        return
    if weight.dim() >= 2:
        score = (weight * grad).abs().mean(dim=tuple(range(1, weight.dim())))
    else:
        score = (weight * grad).abs()
    if module not in score_accum:
        score_accum[module] = score.detach().clone()
    else:
        if mode == "max":
            score_accum[module] = torch.max(score_accum[module], score.detach())
        else:
            score_accum[module] += score.detach()


def build_soft_masks(score_accum, zone_modules, ratio):
    """Build binary soft masks from Taylor scores — 0 for pruned channels."""
    masks = {}
    for module in zone_modules:
        score = score_accum.get(module)
        if score is None:
            continue
        n_total = score.numel()
        n_prune = int(n_total * ratio)
        mask = torch.ones_like(score)
        if n_prune > 0:
            _, prune_idx = torch.topk(score, k=n_prune, largest=False)
            mask[prune_idx] = 0.0
        masks[module] = mask
    return masks


def apply_gradient_mask(masks):
    """Zero out gradients for masked channels during softmask training."""
    for module, mask in masks.items():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if module.weight.grad is not None:
            view_shape = [-1] + [1] * (module.weight.grad.dim() - 1)
            module.weight.grad.mul_(mask.view(*view_shape))
        if hasattr(module, "bias") and module.bias is not None and module.bias.grad is not None:
            module.bias.grad.mul_(mask)


# ──────────────────────────────────────────────────────────────
# Component pruning
# ──────────────────────────────────────────────────────────────

BLOCKS = [
    "down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
    "mid_block",
    "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3",
]


def classify_module(name: str) -> str:
    """Classify a module name as ffn/cross_attn/self_attn/resnet/other."""
    parts = name.lower()
    if "ff.net" in parts or "ff." in parts:
        return "ffn"
    if "attn2" in parts:
        return "cross_attn"
    if "attn1" in parts:
        return "self_attn"
    if "resnets" in parts:
        return "resnet"
    return "other"


def get_block_name(name: str) -> str:
    """Extract block name (e.g. 'down_blocks.2') from a module name."""
    for block in BLOCKS:
        if name.startswith(block):
            return block
    return "other"


def load_unet(model_path=None, base_model='stabilityai/sd-turbo'):
    """Load UNet from base model or from safetensors (previous round)."""
    if model_path:
        from prune.pruned_rebuild import create_unet_from_safetensors
        config_path = model_path.replace('.safetensors', '.config.json')
        unet = create_unet_from_safetensors(model_path, config_path)
    else:
        base_model = resolve_base_model_source(base_model)
        print(f"Loading UNet from {base_model}...")
        unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet", torch_dtype=torch.float32,
        )
    return unet


def prune_resnet_component(unet, dg, block_name, ratio, round_to=32):
    """Prune conv1 layers in the block's resnets via DependencyGraph."""
    conv_prune_op = None
    if hasattr(tp, "prune_conv"):
        conv_prune_op = tp.prune_conv
    elif hasattr(tp, "prune_conv_out_channels"):
        conv_prune_op = tp.prune_conv_out_channels
    else:
        fn_mod = getattr(getattr(tp, "pruner", None), "function", None)
        if fn_mod and hasattr(fn_mod, "prune_conv_out_channels"):
            conv_prune_op = fn_mod.prune_conv_out_channels
    if conv_prune_op is None:
        print(f"  [resnet] ERROR: No conv prune op available")
        return 0

    pruned_count = 0
    named_modules = dict(unet.named_modules())

    targets = []
    for name, module in named_modules.items():
        if not name.startswith(block_name):
            continue
        if "resnets" not in name:
            continue
        if not isinstance(module, nn.Conv2d):
            continue
        if _is_interface_conv(name):
            continue
        if "downsamplers" in name or "upsamplers" in name:
            continue
        if name.endswith(".conv1"):
            targets.append((name, module))

    if not targets:
        return 0

    for name, module in targets:
        out_ch = module.out_channels
        n_prune = int(out_ch * ratio)
        if round_to > 1:
            remaining = out_ch - n_prune
            rounded = (remaining // round_to) * round_to
            if rounded < round_to:
                rounded = round_to
            n_prune = out_ch - rounded
        if n_prune <= 0:
            continue

        with torch.no_grad():
            imp = module.weight.detach().abs().view(out_ch, -1).sum(dim=1)
            _, idxs = torch.topk(imp, k=min(n_prune, imp.numel()), largest=False)
            idxs = idxs.cpu().tolist()

        try:
            group = dg.get_pruning_group(module, conv_prune_op, idxs)
            if group is not None:
                if hasattr(group, 'prune'):
                    group.prune()
                elif hasattr(group, 'exec'):
                    group.exec()
                new_ch = out_ch - len(idxs)
                print(f"  [resnet] {name}: {out_ch} -> {new_ch} ({len(idxs)} removed)")
                pruned_count += 1
        except Exception as e:
            print(f"  [resnet] {name}: DG prune failed: {e}")

    return pruned_count


def prune_ffn_component(unet, block_name, ratio, round_to=32):
    """Prune FFN GEGLU proj layers via manual paired pruning."""
    pruned_count = 0
    named_modules = dict(unet.named_modules())

    targets = []
    for name, module in named_modules.items():
        if not name.startswith(block_name):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if _is_geglu_proj(name):
            targets.append((name, module))

    if not targets:
        return 0

    for name, module in targets:
        out_f = module.out_features
        half = out_f // 2
        n_prune_half = int(half * ratio)
        round_half = round_to
        if round_half > 1:
            remaining = half - n_prune_half
            rounded = (remaining // round_half) * round_half
            if rounded < round_half:
                rounded = round_half
            n_prune_half = half - rounded
        if n_prune_half <= 0:
            continue

        with torch.no_grad():
            imp_all = module.weight.detach().abs().view(out_f, -1).sum(dim=1)
            imp_paired = imp_all[:half] + imp_all[half:]
            _, half_idxs = torch.topk(imp_paired, k=min(n_prune_half, half), largest=False)
            half_idxs = half_idxs.cpu().tolist()

        idxs = []
        for i in half_idxs:
            idxs.append(i)
            idxs.append(i + half)

        try:
            ok = _manual_prune_ffn(unet, name, module, idxs)
            if ok:
                new_half = half - n_prune_half
                print(f"  [ffn] {name}: half {half} -> {new_half} (full {out_f} -> {new_half*2})")
                pruned_count += 1
        except Exception as e:
            print(f"  [ffn] {name}: manual prune failed: {e}")

    return pruned_count


def prune_attention_component(unet, block_name, attn_type, ratio):
    """Prune attention heads from self_attn or cross_attn."""
    attn_key = "attn1" if attn_type == "self_attn" else "attn2"
    pruned_count = 0

    attn_modules = []
    for name, module in unet.named_modules():
        if not name.startswith(block_name):
            continue
        if attn_key not in name:
            continue
        if not hasattr(module, 'to_q'):
            continue
        attn_modules.append((name, module))

    if not attn_modules:
        return 0

    for name, module in attn_modules:
        inner_dim = module.to_q.weight.shape[0]
        num_heads = getattr(module, 'heads', 8)
        if num_heads <= 1:
            continue
        head_dim = inner_dim // num_heads

        if ratio <= 0.08:
            continue
        n_remove = round(ratio * num_heads)
        if n_remove <= 0:
            continue
        if n_remove >= num_heads:
            n_remove = num_heads - 1

        with torch.no_grad():
            q_w = module.to_q.weight.detach().abs().view(num_heads, head_dim, -1).sum(dim=(1, 2))
            k_w = module.to_k.weight.detach().abs().view(num_heads, head_dim, -1).sum(dim=(1, 2))
            v_w = module.to_v.weight.detach().abs().view(num_heads, head_dim, -1).sum(dim=(1, 2))
            head_imp = q_w + k_w + v_w
            _, heads_to_remove = torch.topk(head_imp, k=n_remove, largest=False)

        remove_set = set()
        for h in heads_to_remove.cpu().tolist():
            start = h * head_dim
            remove_set.update(range(start, start + head_dim))

        keep_idxs = sorted(set(range(inner_dim)) - remove_set)
        keep_tensor = torch.tensor(keep_idxs, device=module.to_q.weight.device)
        new_inner_dim = len(keep_idxs)
        new_heads = num_heads - n_remove

        for proj_attr in ['to_q', 'to_k', 'to_v']:
            proj = getattr(module, proj_attr)
            new_proj = nn.Linear(proj.in_features, new_inner_dim, bias=proj.bias is not None)
            new_proj.weight = nn.Parameter(proj.weight.data[keep_tensor, :])
            if proj.bias is not None:
                new_proj.bias = nn.Parameter(proj.bias.data[keep_tensor])
            new_proj = new_proj.to(device=proj.weight.device, dtype=proj.weight.dtype)
            setattr(module, proj_attr, new_proj)

        out_proj = module.to_out[0]
        new_out = nn.Linear(new_inner_dim, out_proj.out_features, bias=out_proj.bias is not None)
        new_out.weight = nn.Parameter(out_proj.weight.data[:, keep_tensor])
        if out_proj.bias is not None:
            new_out.bias = nn.Parameter(out_proj.bias.data.clone())
        new_out = new_out.to(device=out_proj.weight.device, dtype=out_proj.weight.dtype)
        module.to_out[0] = new_out

        module.heads = new_heads
        module.inner_dim = new_inner_dim
        if hasattr(module, 'inner_kv_dim'):
            module.inner_kv_dim = new_inner_dim

        print(f"  [attn] {name}: {num_heads} heads -> {new_heads} (inner_dim {inner_dim} -> {new_inner_dim})")
        pruned_count += 1

    return pruned_count


# ──────────────────────────────────────────────────────────────
# Taylor + Softmask + Recovery pipeline
# ──────────────────────────────────────────────────────────────

def taylor_softmask_pipeline(
    pipe, student_unet, zones,
    calib_data, device,
    warmup_grad_steps=200,
    softmask_steps=200,
    round_to_val=32,
    ffn_max_prune=0.10,
    max_prune_ratio=0.40,
    per_block_config=None,
    output_path=None,
    reeval_interval=100,
    rampup_steps=1000,
    inference_timesteps=None,
    taylor_mode=None,
):
    """Taylor + softmask pruning pipeline.

    Uses shared calib_data (from generate_calibration_data) for both phases,
    so Taylor scores and LD scores are computed on the same teacher-trajectory
    inputs with a unified teacher-student MSE metric.

    Phases:
      1. Warmup: accumulate Taylor importance scores via forward/backward
      2. Softmask: apply gradient masks with periodic re-evaluation
         (channels can revive when masks are rebuilt)
      3. Physical pruning: zone-based DG pruning + attention head pruning
    """
    teacher_unet = pipe.unet
    teacher_unet.eval()
    teacher_unet.requires_grad_(False)

    MODEL_DTYPE = next(student_unet.parameters()).dtype

    if not calib_data:
        raise ValueError("calib_data is empty — run generate_calibration_data() first")
    if 'teacher_pred' not in calib_data[0]:
        raise ValueError("calib_data samples must contain 'teacher_pred' — regenerate calibration data")

    print(f"\n======== Taylor Softmask Pipeline ========")
    print(f"  Warmup: {warmup_grad_steps}, Softmask: {softmask_steps}")
    print(f"  Reeval interval: {reeval_interval}, Rampup: {rampup_steps} steps")
    print(f"  Calib samples: {len(calib_data)}")

    print(f"\n > Phase 1: Taylor warmup ({warmup_grad_steps} steps)...")

    # inference_timesteps: explicit param > env var > error
    _inference_ts_str = (
        ','.join(str(t) for t in inference_timesteps) if isinstance(inference_timesteps, (list, tuple))
        else str(inference_timesteps) if inference_timesteps
        else os.environ.get('INFERENCE_TIMESTEPS', '')
    )
    if not _inference_ts_str:
        raise ValueError(
            "inference_timesteps must be set — pass it via sp_apply_config.yaml "
            "or as INFERENCE_TIMESTEPS env var (e.g. '999,749,499,249')"
        )

    # taylor_mode: explicit param > env var > default 'avg'
    _taylor_mode = taylor_mode or os.environ.get('TAYLOR_MODE', 'avg')
    if _taylor_mode not in ('avg', 'max', 'weighted'):
        raise ValueError(f"taylor_mode must be 'avg', 'max', or 'weighted', got '{_taylor_mode}'")

    student_unet.zero_grad(set_to_none=True)
    grad_inputs = None
    score_accum = {}

    # Per-timestep Taylor scoring using teacher-student MSE on shared calib_data.
    # Bucket calib_data by timestep so each t_val gets its own sample pool,
    # matching the same bucketing strategy used by LD score.
    _fixed_ts = [int(t) for t in _inference_ts_str.split(',')]
    steps_per_ts = max(1, warmup_grad_steps // len(_fixed_ts))
    print(f"   Per-timestep Taylor ({_taylor_mode}): {len(_fixed_ts)} timesteps x {steps_per_ts} samples each")

    indices_by_t = defaultdict(list)
    for i, s in enumerate(calib_data):
        indices_by_t[int(s['timesteps'].item())].append(i)
    for t_val in _fixed_ts:
        if t_val not in indices_by_t:
            raise ValueError(f"No calib_data samples found for timestep {t_val}. "
                             f"Available timesteps: {list(indices_by_t.keys())}")

    per_ts_scores = {}  # {timestep: {module: score}}

    for t_val in _fixed_ts:
        ts_scores = {}
        t_indices = indices_by_t[t_val]
        for step_i in range(steps_per_ts):
            sample = calib_data[t_indices[step_i % len(t_indices)]]

            student_unet.zero_grad(set_to_none=True)
            noisy_latents = sample['noisy_latents'].to(device, dtype=MODEL_DTYPE)
            timesteps = sample['timesteps'].to(device)
            encoder_hidden_states = sample['encoder_hidden_states'].to(device, dtype=MODEL_DTYPE)
            teacher_pred = sample['teacher_pred'].to(device, dtype=MODEL_DTYPE)

            student_output = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(student_output, teacher_pred)
            loss.backward()

            for name, module in student_unet.named_modules():
                if _is_prunable_module(name, module) and not _is_attention_name(name):
                    if hasattr(module, "weight") and module.weight is not None and module.weight.grad is not None:
                        accumulate_taylor_scores(ts_scores, module, module.weight, module.weight.grad)

            grad_inputs = (noisy_latents.detach().clone(), timesteps.detach().clone(), encoder_hidden_states.detach().clone())

        # Average within this timestep
        for mod in ts_scores:
            ts_scores[mod] /= steps_per_ts
        per_ts_scores[t_val] = ts_scores
        print(f"     t={t_val}: scored {len(ts_scores)} modules ({steps_per_ts} samples)")

    # Combine across timesteps: avg (mean), max (element-wise), or weighted (Min-SNR)
    # weighted: middle timesteps (t≈500) get higher weight via Min-SNR scheme.
    # High-noise steps (t=999) have large but noisy gradients; near-clean steps
    # (t=0) have small gradients. Middle steps have the most informative signal.
    def _snr_weight(t_val: int, gamma: float = 5.0) -> float:
        # Min-SNR weighting (Hang et al. 2023): weight = min(SNR, γ) / SNR
        # Linear noise schedule approximation: alpha_bar ≈ 1 - t/1000
        alpha_bar = 1.0 - t_val / 1000.0
        snr = alpha_bar / max(1.0 - alpha_bar, 1e-8)
        return min(snr, gamma) / max(snr, 1e-8)

    n_ts = len(per_ts_scores)
    if _taylor_mode == 'weighted':
        raw_weights = {t_val: _snr_weight(t_val) for t_val in per_ts_scores}
        total_w = sum(raw_weights.values())
        ts_weights = {t_val: w / total_w for t_val, w in raw_weights.items()}
        print(f"   Timestep weights (Min-SNR): " +
              ", ".join(f"t={t}:{w:.3f}" for t, w in sorted(ts_weights.items(), reverse=True)))

    for t_val, ts_scores in per_ts_scores.items():
        w = ts_weights[t_val] if _taylor_mode == 'weighted' else 1.0
        for mod, score in ts_scores.items():
            if mod not in score_accum:
                score_accum[mod] = score.clone() * w
            else:
                if _taylor_mode == 'max':
                    score_accum[mod] = torch.max(score_accum[mod], score)
                else:  # avg or weighted
                    score_accum[mod] = score_accum[mod] + score * w

    if _taylor_mode == 'avg':
        for mod in score_accum:
            score_accum[mod] = score_accum[mod] / n_ts
    # 'weighted': already normalised by total_w via ts_weights
    # 'max': no normalisation needed
    print(f"   Combined {n_ts} timesteps via {_taylor_mode} -> {len(score_accum)} modules")

    # Phase 2: Build softmasks and run masked gradient steps with periodic re-evaluation
    print(" > Phase 2: Building soft masks from Taylor scores...")
    named_modules_dict = dict(student_unet.named_modules())

    # Build zone → module mapping (used for mask rebuilding)
    zone_module_map = {}
    zone_ratios = {}  # cache ratios per zone for re-evaluation
    for zone in zones:
        zone_modules = []
        for name, module in named_modules_dict.items():
            is_match = any(k in name for k in zone["keywords"])
            is_attn = _is_attention_name(name)
            is_iface = _is_interface_conv(name)
            if is_match and not is_attn and not is_iface and _is_prunable_module(name, module):
                zone_modules.append((name, module))
        zone_module_map[zone["name"]] = zone_modules
        zone_ratios[zone["name"]] = {
            "conv": zone.get("max_conv_prune", zone["step_ratio"]),
            "ffn": zone.get("ffn_ratio", ffn_max_prune),
        }

    def _rebuild_soft_masks(scores, progress=1.0):
        """Rebuild soft masks from Taylor scores (allows channel revival).
        progress: 0.0-1.0 rampup multiplier applied to all pruning ratios."""
        masks = {}
        for zone in zones:
            ratios = zone_ratios[zone["name"]]
            conv_ratio = ratios["conv"] * progress
            zone_entries = zone_module_map.get(zone["name"], [])
            if conv_ratio <= 0 or not zone_entries:
                continue
            conv_modules = [m for n, m in zone_entries if not _is_ffn_linear(n, m)]
            ffn_modules = [m for n, m in zone_entries if _is_ffn_linear(n, m)]
            if conv_modules:
                masks.update(build_soft_masks(scores, conv_modules, conv_ratio))
            zone_ffn = ratios["ffn"] * progress
            if ffn_modules and zone_ffn > 0:
                masks.update(build_soft_masks(scores, ffn_modules, zone_ffn))
        return masks

    # Initial mask build from warmup scores (start at rampup progress for step 0)
    initial_progress = min(1.0, reeval_interval / rampup_steps) if rampup_steps > 0 else 1.0
    soft_masks = _rebuild_soft_masks(score_accum, progress=initial_progress)
    for zone in zones:
        ratios = zone_ratios[zone["name"]]
        if ratios["conv"] > 0:
            print(f"   [Zone: {zone['name']}] Target: conv={ratios['conv']:.0%}, ffn={ratios['ffn']:.0%}")

    n_masked = sum(int((m == 0).sum().item()) for m in soft_masks.values())
    n_total = sum(m.numel() for m in soft_masks.values())
    print(f"   Initial masks: {n_masked}/{n_total} channels masked (rampup progress={initial_progress:.0%})")

    # Softmask training with rampup + periodic re-evaluation (channel revival)
    # Scores accumulate continuously (never reset) so importance ranking stabilizes over time
    print(f" > Running softmask gradient steps ({softmask_steps} steps, reeval every {reeval_interval}, rampup {rampup_steps})...")
    reeval_log = []  # collect (step, loss, masked, total, flips, revived, progress) for charting
    # Continuous score accumulator — initialized from warmup scores, never reset
    running_scores = {m: s.clone() for m, s in score_accum.items()}
    running_count = warmup_grad_steps  # count from warmup phase
    for step in range(1, softmask_steps + 1):
        sample = calib_data[(step - 1) % len(calib_data)]

        student_unet.zero_grad(set_to_none=True)
        noisy_latents = sample['noisy_latents'].to(device, dtype=MODEL_DTYPE)
        timesteps = sample['timesteps'].to(device)
        encoder_hidden_states = sample['encoder_hidden_states'].to(device, dtype=MODEL_DTYPE)
        teacher_pred = sample['teacher_pred'].to(device, dtype=MODEL_DTYPE)

        student_output = student_unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(student_output, teacher_pred)
        loss.backward()

        # Continuously accumulate Taylor scores (never reset — scores stabilize over time)
        for name, module in named_modules_dict.items():
            if _is_prunable_module(name, module) and not _is_attention_name(name):
                if hasattr(module, "weight") and module.weight is not None and module.weight.grad is not None:
                    accumulate_taylor_scores(running_scores, module, module.weight, module.weight.grad,
                                             mode=_taylor_mode)
        running_count += 1

        apply_gradient_mask(soft_masks)

        # Periodic re-evaluation: rampup ratio + rebuild masks (channels can revive!)
        if step % reeval_interval == 0:
            # Rampup: linearly increase pruning ratio from 0 to target over rampup_steps
            progress = min(1.0, step / rampup_steps) if rampup_steps > 0 else 1.0

            # In max mode, scores are already max-pooled — use directly
            # In avg mode, divide by count for stable ranking
            if _taylor_mode == 'max':
                avg_scores = running_scores
            else:
                avg_scores = {m: s / running_count for m, s in running_scores.items()}

            old_masks = {m: mask.clone() for m, mask in soft_masks.items()}
            soft_masks = _rebuild_soft_masks(avg_scores, progress=progress)

            # Count flips (channels that changed state: revived or newly masked)
            total_flips = 0
            total_revived = 0
            for m in soft_masks:
                if m in old_masks:
                    diff = (old_masks[m] != soft_masks[m])
                    total_flips += diff.sum().item()
                    # Revived = was 0 (masked), now 1 (active)
                    total_revived += ((old_masks[m] == 0) & (soft_masks[m] == 1)).sum().item()

            n_masked = sum(int((m == 0).sum().item()) for m in soft_masks.values())
            n_total = sum(m.numel() for m in soft_masks.values())
            print(f"   [Step {step}/{softmask_steps}] loss={loss.item():.4f} "
                  f"| ratio={progress:.0%} of target | masked={n_masked}/{n_total} "
                  f"| flips={int(total_flips)} | revived={int(total_revived)}")
            reeval_log.append({
                "step": step, "loss": loss.item(),
                "masked": n_masked, "total": n_total,
                "flips": int(total_flips), "revived": int(total_revived),
                "progress": progress,
            })

    # Save reeval log for charting
    if reeval_log and output_path:
        log_path = output_path.replace('.safetensors', '.reeval_log.json')
        import json as _json
        with open(log_path, 'w') as _f:
            _json.dump(reeval_log, _f, indent=2)
        print(f"   Reeval log saved: {log_path}")

    # Use converged running scores for physical pruning (averaged over all steps)
    score_accum = {m: s / running_count for m, s in running_scores.items()}

    print("\n > Phase 3 (in-pipeline recovery) removed — run sp_distill.py after pruning")

    # Phase 4: Physical pruning
    has_attn_pruning = per_block_config is not None and any(
        per_block_config.get(b, {}).get("self_attn", 0) > 0 or
        per_block_config.get(b, {}).get("cross_attn", 0) > 0
        for b in per_block_config
    )
    save_path = None if has_attn_pruning else output_path

    prune_zones(
        student_unet=student_unet,
        pipe=pipe,
        zones=zones,
        grad_inputs=grad_inputs,
        importance=None,
        round_to_val=round_to_val,
        allow_conv_shortcut_prune=False,
        ffn_max_prune=ffn_max_prune,
        max_prune_ratio_per_layer=max_prune_ratio,
        output_path=save_path,
        device=device,
        taylor_scores=score_accum,
    )

    # Attention head pruning
    if has_attn_pruning:
        print("\n=== Attention Head Pruning ===")
        pre_attn_params = sum(p.numel() for p in student_unet.parameters())
        for block_name, block_cfg in per_block_config.items():
            for attn_type in ['self_attn', 'cross_attn']:
                ratio = block_cfg.get(attn_type, 0)
                if ratio > 0:
                    prune_attention_component(student_unet, block_name, attn_type, ratio)
        post_attn_params = sum(p.numel() for p in student_unet.parameters())
        print(f"  Attention pruning removed: {(pre_attn_params - post_attn_params)/1e6:.2f}M params")

        sync_diffusers_attributes(student_unet)
        if output_path:
            save_checkpoint(student_unet, output_path)

    gc.collect()
    return score_accum
