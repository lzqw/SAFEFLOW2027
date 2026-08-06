# STAR P4.3 safety-conditioned stochastic-flow decision

Decision: **hard_stop_reward_td_divergence**

All nine preregistered SafetyPointGoal1-v0 runs stopped at 50k. S0, S1, and S2 use one policy and the same Bellman probability critic; candidate probing is absent.

| arm | return | cost | CVaR-20 | success | stall | V_N (conditional) | V_T (conditional) | crossing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 0.727 | 37.767 | 116.750 | 0.267 | 0.050 | 0.023339 | 0.024333 | 0.418 |
| S1 | 1.496 | 42.667 | 142.000 | 0.500 | 0.050 | 0.019840 | 0.023444 | 0.671 |
| S2 | 1.873 | 49.900 | 160.917 | 0.500 | 0.067 | 0.019133 | 0.023857 | 0.495 |

## Frozen gates

- FAIL — no_reward_td_divergence
- PASS — boundary_evidence_every_s1_s2_seed
- FAIL — critic_evidence_s1
- PASS — critic_evidence_s2
- FAIL — s1_normal_variance_reduced
- FAIL — s1_tangent_variance_preserved
- PASS — s1_tangent_entropy_increased
- FAIL — s1_crossing_reduced
- PASS — anisotropic_entropy_exceeds_matched_isotropic_low
- PASS — s1_functional_success
- PASS — s1_functional_stall
- PASS — s2_functional_success
- PASS — s2_functional_stall
- PASS — s2_more_inward_than_s1
- PASS — s2_crossing_no_higher_than_s1
- FAIL — s2_mean_cost_no_higher_than_s1
- FAIL — s2_cvar_no_higher_than_s1
- PASS — s2_success_within_0_10_of_s1
- PASS — s2_stall_within_0_10_of_s1

This is a three-seed directional mechanism screen, not a global safety guarantee, a significance claim, or automatic authorization to enter P5.
