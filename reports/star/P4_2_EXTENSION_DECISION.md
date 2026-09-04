# P4.2 top-two 100k extension decision

Status: completed the preregistered B4/B1 100k × 3 extension; stop at 100k because the P5 entry
guard failed.

Training source: `5e994d7d7a89b4dda8ee94d22a1e503f83b409de`

Frozen analysis source: `4a576437129e4d59ec8c192cef1df24e69571aa2`

Environment: `SafetyPointGoal1-v0`

Seeds: `2027`, `2028`, `2029`

## Protocol and provenance audit

B4 and B1 were selected mechanically by the preregistered 50k screen. For each arm, seed 2028
resumed its exact screen `final_full.pt` and ran 50,000 new transitions; seeds 2027 and 2029 started
fresh and ran 100,000 transitions. Every run ended at exactly 100,000 environment transitions and
98,000 outer updates under the clean training source and registered config hash. Fresh runs have 20
diagnostic rows from 5k through 100k and checkpoints at 0, 50k, and 100k; resumed runs have 10 new
rows from 55k through 100k and a 100k checkpoint. Every run also has latest-full and final-full
recovery checkpoints.

All six 100k milestones were evaluated afterwards in isolated environments with fixed random-flow
bases and seeds `12027` through `12046`. Each primary CSV contains exactly one endpoint row; the
center-flow diagnostic is stored separately and is not used below. Total measured training time was
47,316.761 seconds and the recorded launcher cost was 19.472624 CNY.

## External 100k result

Pooled across the three training seeds, with 20 fixed evaluation episodes per seed:

| arm | cost CVaR-20 | mean cost | episode success | stall ratio | mean return | external Brier skill | external AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B4 | 124.25 | 32.97 | 0.400 | 0.050 | 1.21728 | -0.39663 | 0.87480 |
| B1 | 150.42 | 31.43 | 0.217 | 0.133 | 0.43276 | -1.80630 | 0.66905 |

The individual external endpoints were:

| arm | seed | CVaR-20 | mean cost | success | stall | mean return | external Brier skill | external AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B4 | 2027 | 113.75 | 27.90 | 0.20 | 0.05 | 0.38210 | -0.16754 | 0.85362 |
| B4 | 2028 | 213.50 | 61.90 | 0.60 | 0.00 | 1.91220 | 0.23654 | 0.83868 |
| B4 | 2029 | 45.50 | 9.10 | 0.40 | 0.10 | 1.35755 | -1.25889 | 0.93211 |
| B1 | 2027 | 118.00 | 25.05 | 0.30 | 0.15 | 0.56992 | -1.01110 | 0.66838 |
| B1 | 2028 | 241.00 | 48.55 | 0.10 | 0.10 | 0.02654 | -2.51796 | 0.70902 |
| B1 | 2029 | 92.25 | 20.70 | 0.25 | 0.15 | 0.70181 | -1.88984 | 0.62975 |

## Held-out probability and dynamic geometry

Every fifth complete training episode was assigned to the held-out split before its first
transition and excluded from network optimizers, the global dual, and observation normalization.
The final B4 held-out evidence was:

| seed | Brier skill | AUROC | boundary samples | dangerous normal velocity | first geometry activation | enabled at 100k | reward-TD divergence |
|---:|---:|---:|---:|---:|---:|---|---|
| 2027 | 0.28820 | 0.98158 | 282 | 0.03090 | 55k | no | no |
| 2028 | 0.47207 | 0.85801 | 893 | 0.05340 | 75k | yes | yes |
| 2029 | 0.85373 | 0.99615 | 178 | 0.02089 | never | no | yes |

The reversible state machine was exercised in real training. Seed 2028 enabled geometry after
three consecutive held-out passes at 75k and remained enabled. Seed 2027 enabled at 55k, then
disabled at 70k after two consecutive boundary-count failures. Seed 2029 never accumulated three
consecutive passes. The curriculum stage remained monotonic throughout.

Mean final dangerous normal velocity was 0.03507, 65.3% below the sealed P4.1 reference 0.10099.
The final held-out calibration criteria passed in all three seeds. Isolated external Brier skill is
a different split and was negative when pooled; it is reported separately and is not relabeled as
held-out evidence.

## P5 entry guard

Seven registered criteria passed:

- B4 cost CVaR-20 was lower than B1;
- B4 stall was lower than the P4.1 100k reference;
- held-out Brier skill was positive in every B4 seed;
- held-out AUROC was at least 0.65 in every B4 seed;
- held-out boundary count was at least 128 in every B4 seed;
- dangerous normal velocity was lower than P4.1;
- B4 success loss versus B1 was no more than 0.10.

Three criteria failed:

- B4 pooled success fell from 0.65 in the 50k screen to 0.40 at 100k, below the registered 0.55
  no-collapse boundary;
- B4 mean cost 32.97 was not lower than B1 mean cost 31.43, despite lower CVaR-20;
- B4 reward TD diverged in seeds 2028 and 2029. Their final four 5k values rose strictly and their
  final/historical-minimum ratios were 3.222 and 2.404, respectively. Seed 2027 did not diverge.

The conjunctive P5 entry guard therefore failed. `final_p4_verdict` remains null. No P5 run,
continuation beyond 100k, 200k run, or formal four-environment experiment was launched.

## Compact evidence seal

The exact machine-readable decision is `reports/star/p4_2_extension_decision.json`. Its SHA-256 is
`cf513fb28ab7eefd46c54d2a41611bbd322dd2b85f56898c6dd1b3696bc094f3`; the file is 56,246 bytes.
Full runs, external work directories, replay buffers, and checkpoints remain on the training server
and are not committed to Git.
