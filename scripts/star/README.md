# STAR server reproduction

This directory reproduces the P0/P1 engineering gates and the P2-P4 STAR route-A validation.
The development branch starts from OmniSafe commit
`15603dd7a654a991d0a4648216b69d60b81a6366`; FLAC
`61a25eef42935fc395b1e04071e2779311fcb705` and SHAPO
`1bb770437bbbe40a89938661343744243e164bf9` are read-only references.

## Confirmed server layout

The rented server's 50 GiB XFS data disk is mounted at `/root/autodl-tmp`. All mutable state is
kept below `/root/autodl-tmp/SAFE`:

```text
/root/autodl-tmp/SAFE/
  code/SAFEFLOW2027/
  code/FLAC/
  code/SHAPO/
  envs/star-p0-p1/
  cache/{pip,conda,torchinductor,tmp}/
  runs/{active,completed}/
  artifacts/{p0,p1,p2,p3,p4}/
```

The system disk is not used for environments, caches, runs, checkpoints, or videos. P0 requires
at least 25 GiB to remain free on the data disk.

Create the three checkouts on the data disk and detach the two references at their audited commits:

```bash
cd /root/autodl-tmp/SAFE/code
git clone git@github.com:lzqw/SAFEFLOW2027.git
git -C SAFEFLOW2027 switch codex/star-p0-p1
git clone https://github.com/bytedance/FLAC.git
git -C FLAC checkout --detach 61a25eef42935fc395b1e04071e2779311fcb705
git clone https://github.com/montrealrobotics/shapo.git SHAPO
git -C SHAPO checkout --detach 1bb770437bbbe40a89938661343744243e164bf9
```

## Environment

From the server checkout:

```bash
cd /root/autodl-tmp/SAFE/code/SAFEFLOW2027
SAFE_ROOT=/root/autodl-tmp/SAFE scripts/star/setup_server_env.sh
```

The environment is Python 3.11 with PyTorch 2.8.0/CUDA 12.8. Safety-Gymnasium 1.0.0 has two
upstream Python 3.11 compatibility issues: mutable NumPy dataclass defaults and its exact
Pygame 2.1.0 metadata pin. `patch_safety_gymnasium_py311.py` applies a version-checked,
idempotent patch; Pygame 2.6.1 is used because 2.1.0 has no Python 3.11 wheel. NumPy remains
1.23.5 for Gymnasium-Robotics 1.2.2, so MoviePy is held on 1.0.3. The complete resolver guard is
`constraints-py311.txt` and the captured result is `environment.lock.txt`.

Do not install `mujoco_py`. The four tasks use Gymnasium 0.28.1, Safety-Gymnasium 1.0.0, and
MuJoCo 2.3.3.

## Gates

Run each gate from the server. Runtime outputs stay on the data disk.

```bash
scripts/star/launch_p0_p1.sh p0-capture
scripts/star/launch_p0_p1.sh p0-env-smoke
scripts/star/launch_p0_p1.sh p0-saclag-5k
scripts/star/launch_p0_p1.sh p0-benchmark-point
scripts/star/launch_p0_p1.sh p0-benchmark-mujoco
scripts/star/launch_p0_p1.sh p1-tests
scripts/star/launch_p0_p1.sh p1-smoke-20k
```

The throughput benchmark holds global batch size and UTD constant while changing concurrency.
Every worker pre-fills replay, performs real warmup updates, waits at a barrier, then measures at
least 600 seconds of rollout, backward updates, and periodic evaluation. It records aggregate FPS,
update/s, GPU utilization and memory, CPU quota use, RAM, temperature, power, and the estimated
cost per million transitions:

```text
1.48 * 1e6 / (3600 * aggregate_env_fps)
```

P1 adds a dedicated `FailureCostMujocoEnv` for Ant-v4 and Walker2d-v4. Only unhealthy termination
has cost 1. A time-limit truncation has cost 0. Replay stores `terminated` and `truncated`
separately, and the Bellman `done` mask is exactly `terminated`, including the rare case where
termination and truncation coincide.

## P2-P4 validation

Run the focused module and loop tests before launching GPU work:

```bash
python -m pytest tests/star -q
```

P4.1 deliberately separates the legacy training-length diagnosis from the corrected algorithm.
Legacy checkpoints are reconstructed with `legacy_checkpoint_compatibility=True` only for
read-only evaluation; they cannot enter the P4.1 training loop. Corrected runs use one state for
all actor/reward/cost/probability networks:

```text
[normalized observation, remaining budget / episode budget, (H - t) / H]
```

Reward backup stops only on `terminated`; cost backup stops on `terminated | truncated`. The
distributional cost critic remains the global cost utility. A separate ensemble probability head
is trained with masked BCE only after an episode has supplied
`1[empirical cost-to-go > remaining budget]`. Its action gradients drive W3 geometry. Replay keeps
the continuous Gaussian boundary score centered at probability 0.5, while W3 additionally requires
resident positive and negative labels, positive Brier skill, AUROC, valid action gradients, and
bounded ensemble disagreement. The actor objective is
`(alpha_E * E - Q_R + lambda * Q_C) / (1 + lambda)`.

The promotion ladder is unit/synthetic risk tests, four environments at 5k, PointGoal at 20k for
three seeds, PointGoal at 100k for three seeds, then 100k chunks up to 300k/600k/1M subject to the
registered early-stop rules. Do not mix a legacy continuation artifact into a P4.1 verdict.

The P3 compatibility matrix runs STAR for 20k steps on PointGoal, PointButton, Ant, and Walker.
The throughput benchmark is a workload measurement and explicitly labels forced-W3 measurements;
it is not evidence that a training run passed the W3 gate:

```bash
python scripts/star/run_star_smoke_matrix.py --help
python scripts/star/benchmark_star_throughput.py --help
```

One preregistered P4 PointGoal seed is launched as follows. Use seeds 2027, 2028, and 2029 as
three independent processes for the final validation; all three use the same fixed evaluation
seeds.

```bash
python scripts/star/train_star.py \
  --config configs/star/SafetyPointGoal1-v0.yaml \
  --steps 300000 --steps-per-epoch 5000 --seed 2027 --device cuda:0 \
  --batch-size 256 --replay-size 500000 --random-steps 2000 \
  --eval-episodes 20 \
  --eval-seeds 12027,12028,12029,12030,12031,12032,12033,12034,12035,12036,12037,12038,12039,12040,12041,12042,12043,12044,12045,12046 \
  --eval-milestones 0,50000,100000,200000,300000 \
  --log-dir /root/autodl-tmp/SAFE/runs/active/star-p4-300k-seed2027/runs \
  --result /root/autodl-tmp/SAFE/artifacts/p4/star-p4-300k-seed2027.json \
  --checkpoint-replay --no-tensorboard
```

Resume only from a full checkpoint, preserving the final total-step target:

```bash
scripts/star/resume_star.sh FINAL_FULL_CHECKPOINT SafetyPointGoal1-v0 300000 RESUME_LOG_DIR \
  --device cuda:0 --result RESUME_RECEIPT.json --no-tensorboard
```

After running `analyze_pointgoal_replay.py` for every final full checkpoint, aggregate the three
run directories, three audit directories, and three launcher receipts. First re-evaluate every
milestone checkpoint on the same CUDA device. The primary policy evaluation uses a local random
generator seeded from each fixed evaluation seed to sample uniform flow base actions. The former
zero-base result is retained separately and explicitly marked `center-flow-diagnostic`; it is not
the primary policy estimate. Development runs use 20 fixed seeds by default, while a formal run
uses 50:

```bash
python scripts/star/evaluate_pointgoal_checkpoints.py \
  --checkpoints CHECKPOINTS_FOR_ALL_THREE_SEEDS \
  --device cuda:0 --evaluation-tier development \
  --output-dir /root/autodl-tmp/SAFE/artifacts/p4/checkpoint-evaluation \
  --work-dir /root/autodl-tmp/SAFE/runs/analysis/p4-checkpoint-evaluation

# Replace the tier for a preregistered 50-seed result.
python scripts/star/evaluate_pointgoal_checkpoints.py \
  --checkpoints CHECKPOINTS_FOR_ALL_THREE_SEEDS \
  --device cuda:0 --evaluation-tier formal \
  --output-dir /root/autodl-tmp/SAFE/artifacts/p4/formal-checkpoint-evaluation \
  --work-dir /root/autodl-tmp/SAFE/runs/analysis/p4-formal-checkpoint-evaluation
```

`pointgoal_checkpoint_evaluation.csv` contains only the fixed-random-flow primary rows;
`pointgoal_center_flow_diagnostic.csv` contains the zero-base diagnostic rows. Per-checkpoint JSON
also records goal completions, goals per episode, start/end displacement, mean action norm, minimum
goal distance, episode and step stall ratios, sample standard deviations, and normal-approximation
95% confidence intervals. The default episode stall definition is zero goals and displacement no
larger than 0.25 simulator units (smaller than one PointGoal goal diameter); both stall thresholds
are CLI-configurable. Tie-correct Spearman remains JSON `null`/blank in CSV when undefined.

Then run the strict aggregator with that corrected fixed-device table:

```bash
python scripts/star/summarize_pointgoal_p4.py \
  --run-dirs RUN_2027 RUN_2028 RUN_2029 \
  --audit-dirs AUDIT_2027 AUDIT_2028 AUDIT_2029 \
  --receipts RECEIPT_2027.json RECEIPT_2028.json RECEIPT_2029.json \
  --checkpoint-evaluation-csv \
    /root/autodl-tmp/SAFE/artifacts/p4/checkpoint-evaluation/pointgoal_checkpoint_evaluation.csv \
  --output-dir /root/autodl-tmp/SAFE/artifacts/p4/summary
```

The 100k promotion checkpoint is a minimum mechanism test, not a final P4 verdict. Pass both the
initial and resumed logger segments and receipts, select the exact endpoint, and use the scoped
assessment so the machine-readable `final_p4_verdict` remains JSON `null`:

```bash
python scripts/star/summarize_pointgoal_p4.py \
  --run-dirs RUN_2027_20K RUN_2027_100K RUN_2028_20K RUN_2028_100K \
    RUN_2029_20K RUN_2029_100K \
  --audit-dirs AUDIT_2027_100K AUDIT_2028_100K AUDIT_2029_100K \
  --receipts RECEIPT_2027_20K.json RECEIPT_2027_100K.json \
    RECEIPT_2028_20K.json RECEIPT_2028_100K.json \
    RECEIPT_2029_20K.json RECEIPT_2029_100K.json \
  --checkpoint-evaluation-csv COMBINED_0_20K_50K_100K.csv \
  --target-steps 100000 --assessment-scope minimum-mechanism \
  --output-dir /root/autodl-tmp/SAFE/artifacts/p4/minimum-100k-summary
```

The summarizer emits per-run and cross-seed evaluation CSVs, a mechanism CSV, and a scoped
machine-readable assessment. In final scope, a seed passes only with positive mean goal
completions, no stall, mean cost at or below the Point budget, probability-head AUROC above 0.65,
positive Brier skill, W3 evidence, reduced dangerous normal velocity, and at least 0.8 tangent
preservation. In particular, dangerous normal velocity is compared from the final W2 diagnostic
interval to the selected final interval; toggling a gate on one already-trained actor is not
misreported as a flow-motion counterfactual. Mechanism rendering also emits the fixed
`1.0/0.5/0.2/0.0` budget by
`safe/boundary/high-risk` table and visualization with P95-P5 risk span, action-gradient norm,
ensemble disagreement, and mean normal.

For checkpointed promotion runs, pass every chronological logger segment in `--run-dirs` and the
matching launcher receipt for every segment in `--receipts`. The summarizer groups them by seed,
verifies that their `run_env_steps` counters form an unbroken chain from zero, retains the early W2
diagnostics, and uses the last segment for final W3 evidence and counterfactuals. Single-segment
runs remain supported.

Use the same corrected checkpoint table for the aggregate diagnostic figure:

```bash
python scripts/star/plot_star_diagnostics.py \
  --run-dirs RUN_2027 RUN_2028 RUN_2029 \
  --checkpoint-evaluation-csv \
    /root/autodl-tmp/SAFE/artifacts/p4/checkpoint-evaluation/pointgoal_checkpoint_evaluation.csv \
  --output-dir /root/autodl-tmp/SAFE/artifacts/p4/figures
```

With this option, fixed-milestone return/cost and tie-correct Spearman values come from the offline
CUDA re-evaluation. The legacy online per-batch rank statistic remains in the raw diagnostic CSV
for provenance, but it is not plotted or used as formal correlation evidence.

The same sequence can be run unattended after launching all three training processes:

```bash
nohup python scripts/star/run_pointgoal_p4_postprocess.py \
  --source-dir ANALYSIS_WORKTREE --python PYTHON_IN_PINNED_ENV \
  --run-parents RUN_PARENT_2027 RUN_PARENT_2028 RUN_PARENT_2029 \
  --receipts RECEIPT_2027.json RECEIPT_2028.json RECEIPT_2029.json \
  --pid-files RUN_PARENT_2027/pid RUN_PARENT_2028/pid RUN_PARENT_2029/pid \
  --output-root /root/autodl-tmp/SAFE/artifacts/p4/final-300k \
  --work-root /root/autodl-tmp/SAFE/runs/analysis/p4-final-300k \
  --device cuda:0 --training-source-commit TRAINING_COMMIT \
  --analysis-source-commit ANALYSIS_COMMIT >p4-postprocess.log 2>&1 &
```

The launcher first validates exact 300k/298k counters, seeds, milestones, and non-empty checkpoint
files. Its `postprocess_receipt.json` status only records pipeline completion; the scientific
PASS/FAIL remains exclusively in `summary/pointgoal_p4_verdict.json`.

## Monitoring and publication

Port 6006 is reserved for TensorBoard:

```bash
scripts/star/launch_p0_p1.sh tensorboard
```

Port 6008 is reserved for read-only monitoring and is never a training dependency. Long runs must
be detached from the SSH connection with the server's available process supervisor (the validated
server used `nohup` with separate PID and console-log files).

After a gate passes, sync source and compact evidence back to the host. Do not copy environments,
caches, videos, or full run directories into the repository. Long formal SHAPO runs are the one
exception for disaster recovery: keep exactly one newest, semantically validated, replay-bearing
`latest_full.pt` per run in an owner-private host directory. Refresh it hourly with
`backup_shapo_p47_checkpoints_to_host.py`; this engineering backup must not inspect scientific
metrics or participate in checkpoint/configuration selection.

Commit and push source plus compact reports from the host branch to the private
`lzqw/SAFEFLOW2027` repository. Never add replay-bearing checkpoint payloads or the private backup
tree to Git.
