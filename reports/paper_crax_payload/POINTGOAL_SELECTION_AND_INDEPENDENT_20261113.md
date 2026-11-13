# PointGoal common selection and independent test — 2026-11-13

All nine formal training runs completed at 10,000,128 (Flow) or 10,000,384
(native SAC) actual transitions. Common validation used the same five seeds
61027–61031 and 50 episodes per seed for every method, training seed, and five
checkpoint opportunities: 225 frozen candidate/seed evaluations in total.
No optimizer, dual, replay, or checkpoint state was changed during validation.

The risk rule was mean episode cost ≤25.0. The frozen per-method candidates
were:

| method | selected training run/checkpoint | common mean cost | common goals/episode | common at-least-one goal | status |
|---|---|---:|---:|---:|---|
| SAC-Lag | seed 51029 / 2,016,256 | 11.63 | 1.66 | 0.808 | risk qualified |
| Stable-Iso | seed 51028 / 10,000,128 | 24.02 | 27.52 | 1.000 | risk qualified |
| EA | seed 51029 / 8,000,000 | 22.00 | 35.26 | 1.000 | risk qualified |

The overall task-performing risk-qualified candidate under the frozen rule is
EA/51029/8M. The selection freeze digest is
`bb57d6466973774d5d956090e1bb85abf79fbae96da5104dc727492fbbfec718`.

## Locked independent test

The exact three selected policy hashes were evaluated on fresh seeds
71027–71031, 100 episodes per seed (500 episodes per selected policy):

| selected policy | mean cost | goals/episode | at-least-one goal | budget-compliant fraction |
|---|---:|---:|---:|---:|
| SAC-Lag / 51029 / 2,016,256 | 13.20 | 1.93 | 0.840 | 0.880 |
| Stable-Iso / 51028 / 10M | 26.85 | 28.29 | 1.000 | 0.562 |
| EA / 51029 / 8M | 21.32 | 35.43 | 1.000 | 0.732 |

Independent testing did not alter selection. Full policy hashes, receipts, and
the compact per-seed summary are stored in
`pointgoal-formal-selection-20261113/`. No physical robot interaction occurred.
