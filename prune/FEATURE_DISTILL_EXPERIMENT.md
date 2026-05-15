# Feature Distillation Smoke Test

## Goal

Run a short SD-Turbo post-pruning recovery distillation to test whether intermediate UNet feature alignment improves image quality beyond the previous prediction-MSE baseline.

The current problem is that prediction loss can become very low, but generated quality still plateaus. The hypothesis is that prediction-space MSE is too weak once the model is already "okay", so the student should instead match the teacher's internal block representations at each SD-Turbo timestep.

## Current Loss

The configured loss is:

```text
0.1 * prediction_loss + 1.0 * normalized_feature_loss
```

Prediction loss is only an anchor. Feature loss is the main distillation signal.

Prediction loss:

```text
MSE(student_pred, teacher_pred) + lambda_l1 * L1(student_pred, teacher_pred)
```

Feature loss:

At every timestep, both teacher and student run on the same teacher latent. The script hooks these large UNet blocks:

```text
down_blocks.0
down_blocks.1
down_blocks.2
down_blocks.3
mid_block
up_blocks.0
up_blocks.1
up_blocks.2
up_blocks.3
```

For each block, the output tensor is normalized before MSE. If teacher/student shapes match, it uses z-score normalized tensor MSE. If pruning changed the channel shape, it falls back to a normalized spatial energy map MSE.

Both prediction and feature losses use per-timestep EMA normalization, so the four SD-Turbo timesteps contribute on similar scale.

## Configuration

The short test is configured in `prune/sp_distill_config.yaml`:

```yaml
output_dir: models/distill_feat_4steps_2h
main_steps: 1200
finetune_steps: 0
n_steps_min: 4
n_steps_max: 4
lambda_pred: 0.1
lambda_feat: 1.0
lambda_attn: 0.0
teacher_cache_dir: null
```

Teacher cache mode is intentionally disabled because cached trajectories do not contain teacher intermediate features.

## Run Command

```bash
python prune/sp_distill.py
```

Expected output:

```text
models/distill_feat_4steps_2h/distill_final.safetensors
models/distill_feat_4steps_2h/distill_final.config.json
models/distill_feat_4steps_2h/distill.log
```

## What To Check

Do not judge by training loss alone. The previous method already reached very low raw prediction loss without enough visual improvement.

After the run, generate the same fixed prompt/seed comparison used for earlier distilled checkpoints and compare:

```text
original SD-Turbo teacher
previous pruned/distilled baseline
new feature-distilled checkpoint
```

Primary question:

```text
Does feature distillation visibly reduce structure/detail drift compared with the previous baseline?
```

If yes, continue with a longer run. If no, the next candidate loss should probably move closer to clean latent or final image/perceptual alignment rather than adding more MSE-style terms.
