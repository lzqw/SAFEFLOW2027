# STAR P4.4 seed-2028 pilot decision

Decision: **stop_after_seed_2028_pilot**

| arm | crossing D | executed v_out D | V_T D | success | stall | cost | CVaR-20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | 0.5152 | 0.109496 | 0.036589 | 0.350 | 0.050 | 104.250 | 310.250 |
| R1 | 0.6423 | 0.170585 | 0.036831 | 0.300 | 0.050 | 115.850 | 307.000 |

## Frozen pilot gates

- FAIL — r1_outward_speed_reduced_by_at_least_half
- FAIL — r1_crossing_reduced_by_at_least_ten_percent
- PASS — r1_tangent_variance_preserved
- PASS — r1_success_within_0_10_of_r0
- PASS — r1_stall_within_0_10_of_r0
- FAIL — r1_mean_cost_no_higher_than_r0
- PASS — r1_cvar_within_ratio
- PASS — r0_numerically_stable
- PASS — r1_numerically_stable
- PASS — r0_heldout_has_both_classes
- PASS — r1_heldout_has_both_classes

Expansion authorized: **False**
