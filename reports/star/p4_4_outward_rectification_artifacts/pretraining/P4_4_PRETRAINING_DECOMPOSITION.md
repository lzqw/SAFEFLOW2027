# STAR P4.4 pretraining outward-drift decomposition

Decision: **p4_4_training_authorized**

The nine immutable P4.3 final checkpoints were evaluated on 64 held-out near-boundary states with 256 common-random-number action samples per state.

| arm | mode | V_N | V_T | trace | crossing | normal displacement | r_out | v_out | clip |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | A | 0.040023 | 0.039964 | 0.079987 | 0.5021 | -0.001353 | 0.0000 | 0.000000 | 0.1046 |
| S0 | B | 0.036322 | 0.035990 | 0.072312 | 0.5010 | -0.003326 | 0.0000 | 0.000000 | 0.0811 |
| S0 | C | 0.000000 | 0.000000 | 0.000000 | 0.4115 | -0.125177 | 0.4349 | 0.164326 | 0.1315 |
| S0 | D | 0.023339 | 0.024333 | 0.047672 | 0.4180 | -0.124781 | 0.4361 | 0.166121 | 0.1437 |
| S1 | A | 0.036607 | 0.043443 | 0.080050 | 0.5129 | -0.000336 | 0.0000 | 0.000000 | 0.0514 |
| S1 | B | 0.034713 | 0.040618 | 0.075331 | 0.5118 | -0.001942 | 0.0000 | 0.000000 | 0.0415 |
| S1 | C | 0.000000 | 0.000000 | 0.000000 | 0.6823 | 0.064379 | 0.6589 | 0.386873 | 0.1003 |
| S1 | D | 0.019840 | 0.023444 | 0.043284 | 0.6714 | 0.063163 | 0.6593 | 0.384934 | 0.1155 |
| S2 | A | 0.034594 | 0.045334 | 0.079928 | 0.5100 | -0.000343 | 0.0000 | 0.000000 | 0.0787 |
| S2 | B | 0.032177 | 0.041392 | 0.073568 | 0.5137 | -0.000053 | 0.0000 | 0.000000 | 0.0621 |
| S2 | C | 0.000000 | 0.000000 | 0.000000 | 0.5000 | 0.001018 | 0.5026 | 0.258441 | 0.0990 |
| S2 | D | 0.019139 | 0.023822 | 0.042962 | 0.5025 | 0.000141 | 0.4962 | 0.261549 | 0.1135 |

## Critic-normal agreement gate

- Valid cosine states: 566 / 576
- Negative cosines: 260
- Pooled negative fraction: 0.459364
- Per-checkpoint minimum valid states: 58
- Training authorized: True

A blocked decision requires critic repair before any R0/R1 training; decomposition differences do not override that gate.
