# Interim paper experiment results — 2026-11-12

This is development progress, not a completed paper comparison. Independent
branch from `fc5716e`; historical results remain unchanged. Registered budgets,
seeds and selection rules: [protocol](EXPERIMENT_PROTOCOL.md).

## 1. Is PointGoal fast and effective?

The official native task runs on A100, with 1000 control steps, goal refresh,
raw cumulative safety costs, observation dimension 62 and action dimension 2.
Budget is 25 per episode. No physics or hazards were simplified.

| Development run, seed 41027 | Actual training | Goals/episode | Mean cost | Evaluation |
|---|---:|---:|---:|---|
| SAC, no normalization | 1,000,192 | 0.30 | 72.23 | common validation, 50 episodes |
| SAC, normalized, dual LR .01 | 3,000,064 | 33.86 | 84.44 | native diagnostic, 50 episodes |
| SAC, normalized, dual LR .1 | 3,000,064 | 28.90 | 74.44 | native diagnostic, 50 episodes |
| Flow-Iso, normalized, episode dual | 500,224 | 0.12 | 2.60 | common validation, 50 episodes |
| Flow-Iso, normalized, native-unit dual | 500,224 | 0.00 | 0.082 | common validation, 50 episodes |

Native diagnostic and common-layout validation are not interchangeable;
the two normalized SAC runs are awaiting the same common validation. SAC learns
the task but has not met the safety budget. Iso's low cost currently reflects
stagnation, not success. No effective final Flow configuration or PointGoal EA
benefit has been established. The latest Iso training-episode bookkeeping is
under a focused boundary diagnosis; raw replay and all checkpoints are kept.

SAC LR .1 end-to-end throughput was 5701 transitions/s including compilation,
native evaluation and saves. Iso's latest full 500k run was 1005 transitions/s,
with 240/498 seconds spent evaluating. Compiled evaluation reuse is implemented
prospectively. See [measured profiles](CRAX_FAST_PROFILE.md).

## 2. Engineering gains versus algorithm changes

Batched head VJPs and joint target draws increased Fast EA profile steady training
throughput from 1082 to 2482 transitions/s at the same E/B/G and sample counts.
This is an engineering profile, not a learned task result; whole short-run
throughput including compilation/saves is substantially lower. Profiles have
different replay exposure and cannot establish method superiority.

Normalization and the revised dual recipe are explicit scientific configuration
changes. Old results are retained. A frozen gradient probe found weighted cost
gradient norm 18.44 times reward gradient norm, but positive alignment and
negative cost-Q predictions prevent a simple causal explanation of stagnation.
[Full diagnostic and limitations](CRAX-GRADIENTS-ISO500K-CEC.md).

## 3. Fair selection and independent EA benefit

Not established. No formal selection has occurred and independent 71027-series
tests remain unused. Up to five prospective policy checkpoints per training seed
are retained. A learned Iso baseline is required before PointGoal EA comparison;
formal three-method, three-seed training has not started.

## 4. Full endpoints, seeds and failures

Development seeds are 41027/41028; formal seeds 51027/51028/51029 remain registered.
All available checkpoints, intermediate validation and weak endpoints remain.
Three initial profiling runs failed in preheat logging, before optimizer updates;
their native counters were not persisted. Source inference is 29,696 collected
transitions, and the full 98,304 requested budget is conservatively reserved.
Costs are unknown, not zero. [Failure accounting](CRAX_PROFILE_V1_FAILURES.json).

One actual native SAC full-state recovery check passed learner/replay/key byte
comparison and native-state numerical comparison, not whole-state bitwise
identity. Small derived COM/angular-momentum differences are disclosed in its
comparison artifacts. No historical long evaluation was rerun.

## 5. Payload, generalization, mechanisms and Piper

- Payload: two first-seed 300k endpoints completed. Both SAC and Iso reached
  100% nominal safe success with zero violations in 30 development episodes.
  Across four stress conditions, SAC safe success was 90.0%, Iso 74.2%.
  Offset-obstacle safe success was only 63.3%/43.3%, respectively. These are
  development conditions, not independent selected-policy tests. EA first
  seed and SAC second seed are running; two other jobs remain queued.
- Reach M1–M3: completed on 128 fresh common states × 64 draws, all four old
  actors, plus 720 route episodes. Directional-Full improves risk versus
  Iso-High but does not dominate Iso-Low; pullback does not win every actor.
  [Actual findings](ARMMECHANISM_REPORT.md). M4 four synthetic checks completed.
- Independent generalization, PointGoal common-state mechanisms and formal
  Payload confirmation remain pending. Constructive detours are feasibility
  checks, not learned-policy success or route-adaptation evidence.
- Piper: user confirmed standard arm and standard gripper. Pinned official model,
  Reach/Payload simulation, genuine Iso-Low engineering save/resume, offline
  recording and timeout handling are implemented. Six native simulation
  candidates are being prepared; none is yet a trained deployment candidate.
  Hardware calibration/firmware/TCP/limits remain onsite inputs. Physical
  transport and motion remain disabled; zero real interactions.

## 6. Actual costs and remaining work

[Trial ledger](TRIAL_LEDGER.csv) and [native snapshot](NATIVE_PROGRESS.json) retain
actual completed segments and observed active lower bounds. Additional engineering
costs include Payload 3220 training/222 nontraining controls; Piper 3220 training/
130 nontraining controls including its separate original smoke; native SAC
recovery 8448 training; CRAX bridge 10 nontraining controls; Reach mechanisms
437,685 nontraining controls. Frozen gradient analysis uses no environment steps.
Failed-profile conservative reservation is separate from observed actual totals.

The whole objective remains active: effective shared Flow/EA development, fair
formal PointGoal comparison, Payload development and confirmation, selected-policy
independent tests, PointGoal mechanism supplement and native Piper deployment
candidates/dry-run bundles are still required. Optional CVPO is nonblocking.
Physical experiments require a separate onsite enabling decision.
