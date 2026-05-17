# Structured Pruning Decision Note

## 目的

这份记录用于总结本轮 diffusion model structured pruning 的尝试、观察和技术决策。

结论不是“pruning work 失败”，而是：我们完成了一轮比较完整的 pruning feasibility study，并基于实验结果判断，当前 few-step diffusion / consistency-style base model 更适合优先走 quantization，而不是继续把 structured channel pruning 作为主压缩路线。

这部分工作仍然可以体现在简历和面试中，作为模型压缩研究判断力、实验设计能力和 failure analysis 的证据。

## 背景

初始目标是压缩 SD-Turbo / LCM 类少步扩散模型，让模型更适合 edge GPU 或消费级 GPU 推理。

我们参考 Diff-Pruning 类思路，但做了几处适配：

- 原始 Diff-Pruning 依赖 diffusion timestep 训练/评估流程；我们面对的是 Turbo / LCM 这种 few-step inference model。
- 我们使用 sensitivity analysis 来估计不同层、不同 timestep 对最终输出的影响。
- 我们尝试 progressive pruning，而不是一次性大比例剪枝。
- 剪枝后尝试 prediction、feature、latent、rollout 等蒸馏方式恢复质量。

## 已完成的工作

### 1. Baseline 和生成评估

我们先用固定 prompt/seed 生成 baseline 和 pruned model 输出，用视觉结果作为主要判断依据。

重要经验：

- 对 LCM，生成设置必须使用合适的 native setup。
- LCM 在 `4 steps / 512` 下质量明显偏糊。
- 切换到 `8 steps / 768 / guidance=8.0 / original_inference_steps=50` 后，LCM baseline 明显好于 SD-Turbo，适合作为更强 base model。

相关脚本：

```text
prune/gen_lcm_baseline.py
prune/gen_test_lcm.py
```

### 2. Timestep sensitivity ablation

我们做过 hybrid timestep ablation：teacher 正常 rollout，但在某一个 timestep 替换成 student/pruned UNet，观察最终 latent/image drift。

结果显示，SD-Turbo 4-step 中前两个 timestep 对最终输出影响最大。

| Student timestep | Timestep | Mean final latent MSE | Mean image MSE |
| --- | ---: | ---: | ---: |
| 0 | 999 | 0.247112 | 0.0337911 |
| 1 | 749 | 0.0872579 | 0.0128603 |
| 2 | 499 | 0.0138337 | 0.00246655 |
| 3 | 249 | 0.00195839 | 0.00049505 |

由此得到的 pruning score timestep 权重近似为：

```yaml
timestep_weights:
  999: 0.70
  749: 0.25
  499: 0.05
  249: 0.05
```

这说明 few-step diffusion 中早期 step 的误差会强烈传播，不能简单平均所有 timestep。

详细记录见：

```text
prune/TIMESTEP_WEIGHT_ABLATION.md
```

### 3. Progressive pruning pipeline

我们重新设计了 progressive pruning pipeline，核心思路包括：

- 长数据积累替代 soft masking，减少实现复杂度和不稳定因素。
- 使用 avg score 和 max score 的综合排名，避免只看平均值导致偶发重要通道被剪。
- 使用 mask stability check，筛掉 A/B 数据划分下排名不稳定的通道。
- 多轮渐进式剪枝，每次只剪一小部分 channel。
- 每轮剪枝后做 teacher-forced pred loss benchmark。

LCM smoke 脚本单独实现为：

```text
prune/sp_prune_lcm_smoke.py
```

这样避免污染原本 Turbo pipeline。

### 4. Distillation recovery attempts

剪枝后我们尝试过多种恢复方式：

- prediction distillation
- feature distillation
- latent distillation
- rollout latent distillation
- prediction + feature combined distillation
- gradient accumulation / larger effective batch

代表性观察：

- 单纯 prediction MSE 可以下降，但图像质量不一定同步提升。
- feature loss 初期下降很快，例如 40 steps 内从约 `0.27` 到 `0.20`，但视觉改善有限。
- rollout latent + feature 在部分实验中 loss 很高且不稳定，图像出现 blur，说明多条训练分支可能互相干扰。
- pred + feat 是相对基础、稳定的恢复设置，但对结构性构图漂移的修复能力有限。

之前的 pruned/distilled model 与 baseline 的 LPIPS 大约在 `0.27` 左右，这个距离偏高，说明即使蒸馏后仍存在明显 perceptual gap。

相关记录：

```text
prune/FEATURE_DISTILL_EXPERIMENT.md
```

## 关键实验数据

### SD-Turbo progressive smoke

一次 `~2.5%` 实际剪枝后，teacher-forced pred loss 出现明显上升：

```text
post_pred_loss = 0.00559961
```

虽然绝对数值看起来不大，但生成图已经出现可见差异。

### LCM native smoke

LCM 使用 native/manual sampling 重新测试：

```text
base_model = models/lcm-dreamshaper-v7
steps = 8
guidance_scale = 8.0
original_inference_steps = 50
resolution = 768x768
```

剪枝设置：

```text
nominal prune ratio = 2%
round_to = 32
actual per-layer prune = 32 / 1280 = 2.5%
```

结果：

```text
pre_pred_loss  = 0.00000022
post_pred_loss = 0.00663466
```

输出模型：

```text
models/progressive_lcm_native_smoke_p02.safetensors
models/progressive_lcm_native_smoke_p02.config.json
```

对比图：

```text
gen_test_output/progressive_lcm_native_smoke_p02_8steps_native_fixed
```

观察：

- 即使 post pred loss 只有 `0.0066`，画面 degradation 仍然肉眼可见。
- degradation 不只是轻微纹理变化，还包括主体位置、构图、脸部/动物细节的漂移。
- 这说明 teacher-forced pred loss 对 LCM 的最终视觉质量不够敏感。

## 为什么决定暂时放弃 pruning 作为主路线

### 1. Few-step models 对 structured pruning 更敏感

Turbo / LCM 都属于少步生成模型。相比普通 SD 1.5 的 20-50 step 采样，few-step 模型每一步承担的信息量更大。

在这种设置下，一个 timestep 的小误差更容易传播到最终 latent，导致构图和主体结构漂移。换句话说，后续 step 没有足够空间把前面的偏差慢慢修回来。

### 2. 小比例 raw pruning 已经产生可见损伤

LCM 在实际 `2.5%` channel pruning 下就出现明显视觉差异。

这不一定意味着 pruning 完全不可用，但说明当前 naive structured channel pruning 的安全窗口很窄。如果每一轮只能剪 `1%` 左右，还需要昂贵蒸馏恢复，那么整体收益会显著变低。

### 3. Loss 和视觉质量不一致

teacher-forced pred loss 低，不代表最终 rollout 图像稳定。

原因包括：

- pred loss 测的是同一 latent/timestep 输入下的单步输出差异。
- 真实生成是 autoregressive-style rollout，上一步偏差会改变下一步输入。
- MSE 容易低估结构性错误，例如脸部、主体位置、构图变化。
- LCM 的少步轨迹会放大这种结构性漂移。

因此，pruning gate 不能只依赖 pred loss，必须引入 rollout latent drift、image MSE、LPIPS、CLIP-IQA 或人工视觉检查。

### 4. 蒸馏可以修复一部分，但不是免费的

蒸馏有可能恢复纹理、局部细节和一部分风格一致性。

但如果 raw prune 后已经产生构图级别漂移，蒸馏需要修复的是结构性偏移，而不只是轻微 feature mismatch。这会增加训练成本和不确定性。

### 5. Quantization 更适合作为当前主线

相比 structured pruning，quantization 更容易形成稳定、可交付的压缩路线：

- 不改变网络拓扑，部署风险更低。
- 对 few-step diffusion 的结构性轨迹影响可能更小。
- 可以做 layer-wise mixed precision，避开敏感层。
- 指标更容易表达：latency、VRAM、model size、LPIPS/FID/CLIP score。
- 简历和工程交付上更容易被理解。

因此当前战略建议是：

```text
Quantization as the main compression path.
Pruning as a completed feasibility study and research supplement.
```

## 这部分 pruning work 如何体现在简历中

这部分不应写成“成功压缩了多少”，而应写成“完成了 pruning feasibility study，并基于实验发现做出技术路线调整”。

可用英文 bullet：

```text
Investigated structured channel pruning for few-step diffusion models by building a progressive pruning and distillation pipeline; found LCM/Turbo-style models highly sensitive to small channel pruning and designed rollout/image-level evaluation gates beyond teacher-forced prediction loss.
```

更工程导向版本：

```text
Built a sensitivity-driven pruning pipeline for Turbo/LCM diffusion UNets, including timestep-weighted channel scoring, progressive pruning, mask stability checks, and post-pruning distillation; identified that few-step consistency models exhibit visible image drift even under ~2.5% structured pruning, motivating a quantization-first compression strategy.
```

中文解释版本：

```text
我并不是简单尝试 pruning 后放弃，而是完整实现了 sensitivity scoring、progressive pruning、distillation recovery 和 rollout evaluation。实验发现 few-step diffusion 对结构剪枝非常敏感，低 pred loss 仍可能对应明显视觉退化，因此最终把主压缩路线转向更稳定的 mixed-precision quantization。
```

## 后续如果继续 pruning，建议怎么做

如果之后仍要保留 pruning 作为支线，可以考虑更保守的方向：

- 使用 `round_to=16`，让实际 per-layer pruning 从 `2.5%` 降到约 `1.25%`。
- 不全层一起剪，优先在稳定候选层做更小比例剪枝。
- attention 暂时不动，除非非 attention 层已经接近极限。
- 每轮 pruning 后不仅看 pred loss，还要看 fixed seed rollout image drift。
- 用 LPIPS / image MSE / final latent MSE 作为 gate。
- 剪枝后只做轻量 pred+feat distillation，避免复杂 loss 分支互相干扰。

但从当前收益/风险比看，主线应转向 quantization。
