# P4.2 B1–B4 causal-screen decision

Status: completed the preregistered 50k single-seed screen; extend B4 and B1 to 100k × 3.

Training source: `5e994d7d7a89b4dda8ee94d22a1e503f83b409de`

Screen seed: `2028`

Environment: `SafetyPointGoal1-v0`

## Immutable result

All four arms completed exactly 50,000 environment transitions with finite core optimizer signals
and the registered config fingerprints. Training-time policy evaluation was disabled; the step-0
and step-50k checkpoints were evaluated afterwards in isolated environments using fixed random-flow
bases and seeds `12027` through `12046`. Total measured training time was 15,188.498 seconds and
the recorded launcher cost was 6.256927 CNY.

The 50k primary external results were:

| arm | cost CVaR-20 | mean cost | episode success | stall ratio | mean return | external Brier skill | external AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 206.25 | 52.35 | 0.15 | 0.15 | 0.55891 | -0.08775 | 0.78024 |
| B2 | 239.25 | 91.80 | 0.20 | 0.05 | 0.11827 | -0.03034 | 0.83503 |
| B3 | 239.25 | 91.80 | 0.20 | 0.05 | 0.11827 | -0.03034 | 0.83503 |
| B4 | 109.00 | 32.90 | 0.65 | 0.05 | 0.86666 | 0.31747 | 0.92449 |

Every arm passed the preregistered performance-eligibility guard. No arm met the two-part reward-TD
divergence definition: B1 had final/minimum ratio 1.466 without a rising final tail; B2 and B3 had
ratio 2.093 without a rising final tail; B4 had ratio 2.127 without a rising final tail.

## Mechanism localization

B2 and B3 were numerically identical at 50k. Their held-out probability evidence ended at AUROC
0.67727 and Brier skill -1.18152 with only 51 effective boundary samples, so B3's held-out dynamic
geometry gate correctly remained closed. This screen therefore does not attribute a behavioral
effect to dynamic geometry by itself.

B4 added only the registered K=16 candidate-probing path relative to B3. It ended with held-out
AUROC 0.97511 and positive Brier skill 0.03165, but only one effective boundary sample, so its
dynamic geometry gate also remained closed. B4 nevertheless improved all leading external ranking
metrics in this one-seed screen. The causal screen localizes that difference to the probing path,
without treating a single seed as a final efficacy claim.

## Locked selection

The registered lexicographic ranking was `B4`, `B1`, `B2`, `B3`. The two extension slots therefore
went to B4 and B1, both on performance eligibility rather than fallback. Each is extended to 100k
at seeds `2027`, `2028`, and `2029`; seed 2028 resumes its exact 50k full checkpoint, while the other
two seeds start fresh.

`final_p4_verdict` remains null. No continuation beyond 100k and no formal four-environment
experiment was launched.

## Compact evidence seal

The exact machine-readable decision is `reports/star/p4_2_b1_b4_screen_decision.json`. Its SHA-256
is `30e664e7f4764fd747a4afe3531d4e1b44840666ca38941984b9423c2e5554aa`; the file is 35,131 bytes.
Full runs, replay buffers, and checkpoints remain on the training server and are not committed to
Git.
