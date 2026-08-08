# P4.2 B0-R3 three-seed backbone decision

Status: passed all preregistered B0 criteria; the Bellman probability critic is authorized.

## Immutable result

All three seeds used training source `b3c490309d465b89a5ae5a6ed235835587074abb`
and the frozen reward/cost/actor target coefficients 0.001/0.005/0.005. Each run completed
exactly 50,000 environment transitions, 48,000 outer updates, 96,000 reward-critic updates,
21,000 actor updates, and a 50,000-transition replay. Every seed has milestone-0,
milestone-50k, latest-full, and final-full checkpoints. Total measured training time was
7,777.170 seconds and the recorded launcher cost was 3.206435 CNY.

The fixed-random-flow evaluation used seeds 12027--12046 and 20 episodes per checkpoint per
training seed. Pooled results were:

| metric | 0 | 50k | criterion |
|---|---:|---:|---|
| mean return | -1.34258 | 6.15516 | strict improvement: pass |
| episode success | 0.08333 | 0.73333 | strict increase: pass |
| stall ratio | 0.01667 | 0.00000 | no increase: pass |
| mean step displacement | 0.010015 | 0.010111 | no decline: pass |
| mean action norm | 0.81792 | 0.65801 | diagnostic only |

The step-displacement margin is small but positive (`+0.00009594`, about 0.96%). The registered
metric is consecutive simulator displacement; start/end displacement remains diagnostic and
declined from 2.33556 to 1.18250.

No seed triggered reward-TD divergence:

| seed | final TD | historical minimum | final/min | last four strictly rising |
|---:|---:|---:|---:|---|
| 2027 | 0.002331 | 0.001761 | 1.324 | no |
| 2028 | 0.002878 | 0.002598 | 1.108 | no |
| 2029 | 0.004194 | 0.002624 | 1.599 | no |

All five machine criteria are true. The frozen decision is
`proceed_to_bellman_probability_critic`. This opens only the next P4.2 mechanism stage;
`final_p4_verdict` remains null, and neither continuation beyond 50k nor the formal
four-environment experiment was launched.

## Compact evidence seals

| artifact | SHA-256 |
|---|---|
| exact machine decision | `90252b4b7690e37ef65513053b8e45896fb28e531f4d1e7cc740721cbe1d106b` |
| combined six-row external CSV | `f8cff54ddbeb30960a116c0dbadb4d76c828981dfe2429d4535d140ecebf2212` |
| seed-2027 external CSV | `c8d5e2bc8f53276f11fd4cce98f7535cf5cfa1b191d0ef10d8376d373078c125` |
| seed-2028 external CSV | `805284dee5ed6bd5c33706b8b937680e55e1ee5d8efa075c0ab110a9cb8a8938` |
| seed-2029 external CSV | `55206d03c1e0e71dcc36e22337a8009029e0b9615f9465ee4109b0ad3a32d876` |
| seed-2027 receipt | `d66dca966684f238e38ef0cbd18381ab7adb7bdff4abcb87a26d5fa25295c647` |
| seed-2028 receipt | `fff78342d39e8e66d5532b08757d94d666803e2101226be24c04f2daa05e0301` |
| seed-2029 receipt | `951b1a26ca5b8452ec1f62d2a71dfe19444af25aab47a70b3826a8bddb895dd4` |
| seed-2027 diagnostics | `44e1757255c5fd944f90320b465f3de0084d18950cb206725f23815d3a543cee` |
| seed-2028 diagnostics | `f76577667e71963b45af95309a5f30f7ac02746f835804ba1f2d4bad8f1fc5fd` |
| seed-2029 diagnostics | `b36393f89fdb56150ea4955d21e64cce7df4288c5eedb2a3894f15dc30221b27` |

Full runs and checkpoints remain under `/root/autodl-tmp/SAFE/runs/active`; combined compact
evidence remains under
`/root/autodl-tmp/SAFE/artifacts/p4.2/backbone-r3-b3c4903/three-seed`. No checkpoint or replay
buffer is committed to Git.
