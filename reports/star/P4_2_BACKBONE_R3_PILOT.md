# P4.2 B0-R3 seed-2028 pilot

Status: passed the preregistered feasibility hard stops; seeds 2027 and 2029 were authorized.

## Immutable run result

The pilot used training source `b3c490309d465b89a5ae5a6ed235835587074abb`, seed 2028,
global safety off, local geometry off, and the complete R2 tracking repair. R3 changed only the
reward-target Polyak coefficient from 0.005 to 0.001; cost-target and target-actor coefficients
remained 0.005. It completed exactly 50,000 environment transitions, 48,000 outer updates,
96,000 reward-critic updates, 21,000 actor updates, and a 50,000-transition replay. All four
required checkpoints exist. Training took 2,451.427 seconds and the recorded end-to-end receipt
cost was 1.010807 CNY.

The frozen external fixed-random-flow evaluation used seeds 12027--12046 and 20 episodes at each
endpoint:

| endpoint | return | episode success | goals | stall | step displacement | path length |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -3.31098 | 0.15 | 3 | 0.0000 | 0.02023 | 20.2073 |
| 50k | 2.77015 | 0.55 | 20 | 0.0000 | 0.00532 | 5.3192 |

Return and success improved and endpoint stall stayed below 0.5. Step displacement declined by
73.7%; this is not a pilot hard stop, but remains binding evidence for the unchanged three-seed
functional gate.

The reward-TD series at 5k intervals was
`[0.00543, 0.00359, 0.00292, 0.00286, 0.00296, 0.00260, 0.00263, 0.00280,
0.00290, 0.00288]`. The last four values do not rise strictly, and the final/historical-minimum
ratio is 1.1078. The registered reward-TD divergence hard stop is therefore false. All four pilot
hard stops are false, so the frozen machine decision is `launch_remaining_r3_repair_seeds`.

This decision authorizes only the remaining B0-R3 seeds. It does not promote the backbone to the
Bellman probability critic. `final_p4_verdict` remains null; no continuation beyond 50k and no
formal four-environment experiment was launched.

## Compact evidence seals

| artifact | SHA-256 |
|---|---|
| launcher receipt | `fff78342d39e8e66d5532b08757d94d666803e2101226be24c04f2daa05e0301` |
| training summary | `719dae7c0b36ad77ffc050f86cff2a5c69c77a79e126b5d0b3a2126073535564` |
| run manifest | `406fd89316dad1abccd529fd23cbb9e90a977e6824aa76dd9837d9b244944154` |
| 5k diagnostics | `f76577667e71963b45af95309a5f30f7ac02746f835804ba1f2d4bad8f1fc5fd` |
| external evaluation CSV | `805284dee5ed6bd5c33706b8b937680e55e1ee5d8efa075c0ab110a9cb8a8938` |
| checkpoint-0 evaluation JSON | `248ff8cc562b4a64a58c0e695c378cc38835bb300a65a4fdf0835abae3ebdd85` |
| checkpoint-50k evaluation JSON | `f9a3a1f0cf3ca08db05b96afd7e46aefc04ba2989b6a7e3eb011225c7a513bb1` |
| machine pilot decision | `7475bee0218c474b8441cdc00c03bace92e91715d489b3264a7099ee324a3810` |
| milestone-0 checkpoint (server only) | `9e8d1bd8bb8627950f172a91b029d472775c2698bd7fcd666bfe82de8fe87b8e` |
| milestone-50k checkpoint (server only) | `41c638bd9493d0a1c55deebc2bf614271ecb20c5dd76492ddb755d60c5948f76` |
| latest full checkpoint (server only) | `3aa7357ee8e0a9120b4fde2fa1c2943d18dca1eb9260c71f20ba3186a5b45b3c` |
| final full checkpoint (server only) | `fd6386dc85fd95babdf4ad4df5ada967e570aedbc2e2e24bcfb961e3b1bc0327` |

The full run and checkpoints remain under
`/root/autodl-tmp/SAFE/runs/active/p4.2-r3-b3c4903-b0-repair-50k-pilot/seed-2028`.
Compact evaluation evidence remains under
`/root/autodl-tmp/SAFE/artifacts/p4.2/backbone-r3-b3c4903/pilot-seed-2028`. No checkpoint is
committed to Git.
