# Timestep Weight Ablation for SD-Turbo Pruning

## Purpose

We ran a hybrid timestep ablation to estimate which SD-Turbo inference
timestep is most sensitive to pruned/student UNet errors.

For each prompt, the teacher model performed the normal 4-step rollout.  Then
we repeated the rollout four times, replacing exactly one timestep with the
student/pruned UNet while keeping all other timesteps on the teacher UNet.

This isolates the final-image damage caused by student error at each timestep.

## Model Tested

Student checkpoint:

```text
models/distill_pred_feat_from_step600_accum8_4steps/distill_step_300.safetensors
```

Output directory:

```text
gen_test_output/timestep_hybrid_accum8_step300_full8
```

Metrics:

```text
gen_test_output/timestep_hybrid_accum8_step300_full8/metrics.csv
gen_test_output/timestep_hybrid_accum8_step300_full8/summary.csv
```

## Results

Mean final latent MSE and image MSE over 8 fixed prompts/seeds:

| Student timestep | Timestep | Mean final latent MSE | Mean image MSE |
| --- | ---: | ---: | ---: |
| 0 | 999 | 0.247112 | 0.0337911 |
| 1 | 749 | 0.0872579 | 0.0128603 |
| 2 | 499 | 0.0138337 | 0.00246655 |
| 3 | 249 | 0.00195839 | 0.00049505 |

Normalized blend using:

```text
0.5 * normalized_final_latent_mse + 0.5 * normalized_image_mse
```

gave:

| Timestep | Data-derived weight |
| ---: | ---: |
| 999 | 0.6934 |
| 749 | 0.2542 |
| 499 | 0.0446 |
| 249 | 0.0078 |

## Chosen Weights

For pruning-score accumulation, we will use a smoothed version:

```yaml
timestep_weights:
  999: 0.70
  749: 0.25
  499: 0.05
  249: 0.05
```

The first two timesteps dominate the measured final-output damage.  Replacing
only `t=999` with the student causes the largest drift, and `t=749` is the
second largest.  The last two timesteps have much smaller measured impact.

We still keep a small nonzero weight for `t=249` instead of using the raw
data-derived value near zero.  This prevents the final refinement step from
being completely ignored during channel scoring.

## Implication

New pruning-score accumulation should not average all four SD-Turbo timesteps
equally.  It should strongly protect channels that matter for `t=999` and
`t=749`, because early-step errors propagate through the rest of the short
4-step trajectory and dominate final image drift.
