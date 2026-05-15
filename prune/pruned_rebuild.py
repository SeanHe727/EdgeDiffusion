#!/usr/bin/env python3
"""
从 safetensors + config.json 精确重建剪枝后的 UNet 结构。

关键思路：
  不使用 align_tensor 填充零值（会污染已学习的权重）。
  而是先把标准 UNet 里的每个 Conv2d/Linear 替换为 safetensors
  中实际形状对应的新模块，再用 load_state_dict 加载。
"""

import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# 核心：从 safetensors + config.json 重建剪枝模型
# ---------------------------------------------------------------------------

def _get_parent_and_attr(model: nn.Module, dotted_name: str):
    """返回 (parent_module, attr_name)，用于 setattr 替换子模块。"""
    parts = dotted_name.split('.')
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def _find_num_groups(original_num_groups: int, new_num_channels: int) -> int:
    """找到能整除 new_num_channels 的最大 num_groups（不超过 original_num_groups）。"""
    ng = original_num_groups
    while ng > 1:
        if new_num_channels % ng == 0:
            return ng
        ng //= 2
    return 1


def _replace_layers_to_match_shapes(unet: nn.Module, st: dict) -> int:
    """
    遍历 unet 所有 Conv2d / Linear / GroupNorm，
    若 safetensors 中对应权重形状不同，则替换为正确尺寸的新模块。
    返回替换的层数量。
    """
    replaced = 0
    for name, module in list(unet.named_modules()):
        weight_key = name + '.weight'
        if weight_key not in st:
            continue

        w = st[weight_key]
        has_bias = (name + '.bias') in st

        if isinstance(module, nn.Conv2d):
            out_c, in_c = w.shape[0], w.shape[1]
            if out_c != module.out_channels or in_c != module.in_channels:
                new_mod = nn.Conv2d(
                    in_c, out_c,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=has_bias,
                )
                parent, attr = _get_parent_and_attr(unet, name)
                setattr(parent, attr, new_mod)
                replaced += 1

        elif isinstance(module, nn.Linear):
            out_f, in_f = w.shape[0], w.shape[1]
            if out_f != module.out_features or in_f != module.in_features:
                new_mod = nn.Linear(in_f, out_f, bias=has_bias)
                parent, attr = _get_parent_and_attr(unet, name)
                setattr(parent, attr, new_mod)
                replaced += 1

        elif isinstance(module, nn.GroupNorm):
            new_num_ch = w.shape[0]
            if new_num_ch != module.num_channels:
                ng = _find_num_groups(module.num_groups, new_num_ch)
                new_mod = nn.GroupNorm(ng, new_num_ch, eps=module.eps, affine=module.affine)
                parent, attr = _get_parent_and_attr(unet, name)
                setattr(parent, attr, new_mod)
                replaced += 1

        elif isinstance(module, nn.LayerNorm):
            # transformer_blocks.*.norm1/2/3 使用 LayerNorm，normalized_shape=[dim]
            new_dim = w.shape[0]
            if list(module.normalized_shape) != [new_dim]:
                new_mod = nn.LayerNorm(new_dim, eps=module.eps, elementwise_affine=module.elementwise_affine)
                parent, attr = _get_parent_and_attr(unet, name)
                setattr(parent, attr, new_mod)
                replaced += 1

    return replaced


def _fix_internal_attrs(unet: nn.Module):
    """
    更新 diffusers UNet 内部依赖于通道数的属性
    （Upsample2D.channels、ResnetBlock2D.in_channels 等）。
    """
    for name, module in unet.named_modules():
        if hasattr(module, 'channels') and hasattr(module, 'conv'):
            if hasattr(module.conv, 'in_channels'):
                module.channels = module.conv.in_channels
        if hasattr(module, 'in_channels') and hasattr(module, 'conv1'):
            if hasattr(module.conv1, 'in_channels'):
                module.in_channels = module.conv1.in_channels
        if hasattr(module, 'out_channels') and hasattr(module, 'conv2'):
            if hasattr(module.conv2, 'out_channels'):
                module.out_channels = module.conv2.out_channels
        if hasattr(module, 'to_q') and hasattr(module, 'inner_dim'):
            if hasattr(module.to_q, 'weight'):
                new_inner_dim = module.to_q.weight.shape[0]
                old_inner_dim = module.inner_dim
                module.inner_dim = new_inner_dim
                if hasattr(module, 'inner_kv_dim'):
                    module.inner_kv_dim = new_inner_dim
                # Update heads: head_dim is invariant, recompute heads count
                if hasattr(module, 'heads') and module.heads > 0 and old_inner_dim > 0:
                    head_dim = old_inner_dim // module.heads
                    if head_dim > 0 and new_inner_dim % head_dim == 0:
                        module.heads = new_inner_dim // head_dim
                        if hasattr(module, 'sliceable_head_dim'):
                            module.sliceable_head_dim = module.heads


def create_unet_from_safetensors(safetensors_path: str, config_path: str = None) -> nn.Module:
    """
    从 safetensors + config.json 精确重建剪枝后的 UNet。

    流程：
      1. 加载 safetensors（获取实际张量形状）
      2. 从 config_path 中的 model_config 构建标准 UNet
      3. 将形状不匹配的 Conv2d/Linear 替换为正确尺寸
      4. load_state_dict
      5. 修复内部属性
    """
    from diffusers import UNet2DConditionModel

    # 1. 加载 safetensors
    print(f"加载 safetensors: {safetensors_path}")
    st = load_file(safetensors_path)
    total_params = sum(v.numel() for v in st.values())
    print(f"  safetensors 共 {len(st)} 个张量，{total_params/1e6:.1f}M 参数")

    # 2. 读取 model_config
    if config_path is None:
        config_path = safetensors_path.replace('.safetensors', '.config.json')

    model_config = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        model_config = meta.get('model_config')
        print(f"  读取配置: {config_path}")

    # 回退到默认 SD 1.5 配置
    if not model_config or not isinstance(model_config, dict):
        print("  ⚠️  未找到 model_config，使用 SD 1.5 默认配置")
        model_config = {
            "sample_size": 64,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 2,
            "block_out_channels": [320, 640, 1280, 1280],
            "down_block_types": [
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D", "DownBlock2D"
            ],
            "up_block_types": [
                "UpBlock2D", "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
            ],
            "cross_attention_dim": 768,
            "attention_head_dim": 8,
        }

    # 3. 构建标准 UNet
    print("  构建标准 UNet 架构...")
    unet = UNet2DConditionModel(**model_config)

    # 4. 将形状不匹配的层替换为正确尺寸
    replaced = _replace_layers_to_match_shapes(unet, st)
    print(f"  替换了 {replaced} 个形状不匹配的层")

    # 5. 加载 safetensors 权重
    missing, unexpected = unet.load_state_dict(st, strict=False)
    if missing:
        print(f"  ⚠️  缺失键: {len(missing)} 个（例如 {missing[:3]}）")
    if unexpected:
        print(f"  ⚠️  多余键: {len(unexpected)} 个")

    # 6. 修复内部属性
    _fix_internal_attrs(unet)

    param_count = sum(p.numel() for p in unet.parameters())
    print(f"  OK 重建完成，参数量: {param_count/1e6:.1f}M")
    return unet


# ---------------------------------------------------------------------------
# 验证：前向推理测试
# ---------------------------------------------------------------------------

def verify_forward(unet: nn.Module, device: str = 'cpu') -> bool:
    """对重建的模型跑一次前向推理，验证输出形状正确。"""
    unet = unet.to(device).eval()
    with torch.no_grad():
        sample = torch.randn(1, 4, 64, 64, device=device)
        timestep = torch.tensor([1], device=device)
        enc_hs = torch.randn(1, 77, 768, device=device)
        try:
            out = unet(sample, timestep, encoder_hidden_states=enc_hs)
            assert tuple(out.sample.shape) == (1, 4, 64, 64), \
                f"输出形状异常: {out.sample.shape}"
            print(f"  前向推理 OK，输出形状: {tuple(out.sample.shape)}")
            return True
        except Exception as e:
            print(f"  ❌ 前向推理失败: {e}")
            import traceback
            traceback.print_exc()
            return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_st = os.path.join(root_dir, 'models', 'distill_step_40000.safetensors')
    safetensors_path = os.environ.get(
        'PRUNED_SAFETENS_PATH',
        default_st
    )
    config_path = safetensors_path.replace('.safetensors', '.config.json')

    print("=" * 60)
    print("从 safetensors + config.json 重建剪枝 UNet")
    print("=" * 60)

    unet = create_unet_from_safetensors(safetensors_path, config_path)
    ok = verify_forward(unet)

    if ok:
        print("\n✅ 模型重建成功，可直接用于推理/蒸馏！")
    else:
        print("\n❌ 模型重建后前向推理失败，请检查配置")


if __name__ == '__main__':
    main()
