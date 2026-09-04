# Ant 2M C0/P4.7 formal comparison

Both methods use training seeds `2027-2029`, an equal `2M` environment-step budget, and `100` fixed-stream final episodes per checkpoint. The independent statistical unit is the training seed (`n=3`). P4.7 seed 2027 is reused from the sealed prior run.

| Method | Return | Mean cost | CVaR-20 | Failure | Train cost | High-risk V_N/V_T | Outward |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 / isotropic FLAC | 5905.978 | 0.170 | 0.850 | 17.0% | 599.7 | 0.8707 | 0.3718 |
| P4.7 / Ours | 6227.666 | 0.070 | 0.350 | 7.0% | 538.3 | 0.7955 | 0.2974 |

## Aggregate comparison

Return relative change: `+5.4%`.
Mean-cost relative change: `-58.8%`.
CVaR-20 relative change: `-58.8%`.

These are aggregate three-training-seed results. They do not establish that every individual seed is a Pareto improvement.
