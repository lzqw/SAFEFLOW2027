# SAFEFlow CRAX/Payload/Piper experiment status — 2026-11-13

## Completed scientific gates

### CRAX PointGoal

- Formal training: 3 methods × 3 seeds, all nine runs completed at the frozen
  10M-transition budget.
- Common validation: 225 candidate/validation-seed evaluations (five
  checkpoints, five validation seeds, 50 episodes) with frozen policies.
- Selection: all three methods have a risk-qualified candidate under mean
  episode cost ≤25.0. The overall task-performing candidate is EA, seed 51029,
  8M checkpoint.
- Locked independent test: the exact SAC-Lag, Stable-Iso, and EA selected bytes
  were tested on five fresh seeds ×100 episodes. EA averaged 35.43 goals/episode
  at 21.32 mean cost; Stable-Iso averaged 28.29 at 26.85; SAC-Lag averaged
  1.93 at 13.20.

### Payload

- All 30 registered common-validation candidates completed.
- SAC-Lag/41028/300k is risk-qualified and deployment-eligible; Stable-Iso and
  EA are retained as explicit non-qualified Pareto trade-offs because of offset
  stress risk.
- Exact selected policies were independently tested on five held-out condition
  families ×100 episodes. The offset route diagnostic was also run; its strict
  alternative-route claim was not verified because some shifted initial states
  were already unsafe.

### Mechanisms and Piper preparation

- Arm mechanisms M1–M4 and the small PointGoal common-state mechanism analysis
  are retained in the mechanism reports.
- Piper assets, model identity, simulation interfaces, dry-run boundaries, and
  no-motion physical safety constraints are prepared. Iso and Iso-Low Reach
  simulations are complete; EA Reach simulation is still running.

## Remaining gate

Only the Piper EA Reach simulation has not reached its registered 300k-step
endpoint. No physical robot interaction is authorized or performed. Large
checkpoints remain on the servers; GitHub contains code, configs, receipts, and
compact summaries only.
