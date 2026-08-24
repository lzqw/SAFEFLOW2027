# PointGoal four-method final comparison

All methods use three training seeds (`2027–2029`), an equal `200k` training budget, and 100 final evaluation episodes per training seed. The independent statistical unit is the training seed (`n=3`); pooled episodes are not additional training replicates.

| Method | Return | Success | Mean cost | CVaR-20 | Train cost | High-risk V_N/V_T |
|---|---:|---:|---:|---:|---:|---:|
| C0 / isotropic FLAC | 0.4621 | 0.227 | 13.567 | 67.750 | 7305.3 | 0.9982 |
| SACLag | 25.2192 | 1.000 | 53.190 | 108.933 | 12874.3 | n/a |
| SHAPO-PIDLag | 1.5040 | 0.513 | 24.020 | 108.383 | 9655.5 | n/a |
| P4.7 / Ours | 0.6545 | 0.253 | 9.433 | 46.800 | 5582.3 | 0.8744 |

## Frozen P4.7 conclusion

Relative to isotropic FLAC, P4.7 changes return by `+0.1924` and success by `+2.7` percentage points, while reducing mean cost by `30.5%`, CVaR-20 by `30.9%`, and mean cumulative training cost by `23.6%`.

This remains an aggregate three-seed Pareto statement, not an every-seed Pareto claim. SACLag and SHAPO-PIDLag are reported as preregistered prior-method comparisons; their outcomes were not used to select or modify P4.7.

## Evaluation semantics

- C0 and P4.7 reuse the sealed P4.7 fixed-random-flow artifacts.
- SACLag and SHAPO-PIDLag use their native stochastic action distributions with fixed Torch RNG streams and the same five base environment seeds.
- Every base seed advances through 20 sequential episodes without reseeding between episodes.
- SHAPO-PIDLag uses the observation running statistics captured during training and keeps them frozen during evaluation.
