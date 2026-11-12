# Payload fair selection and frozen generalization: prepared, not executed

This adds NEW files only. The active4080 `payload_train.py`, learners and
environment are unchanged. Existing native training had in-loop development
evaluation but no standalone common validation/test CLI. No validation rollout,
independent test, new training or simulator query was executed for this work.
The generated manifests are prospective commands, not measured policy results.

## Equal opportunity and preserved endpoints

Protocol: `configs/paper_crax_payload/payload_selection_v1.json`.
Development registers SAC-Lag/Iso/EA ×seeds41027/41028 ×five checkpoint fractions,
exactly30 candidate IDs and ten allowed candidates/method. Native300k endpoints
remain retained for both seeds, even if a different checkpoint is selected.
The generic cohort interface also registers future confirmation seeds51027–29
at a common600k budget:45 candidates/fifteen allowed per method. This does not
launch or authorize those training runs. A common1M extension requires a new
prospective protocol version for ALL methods, not an EA-only extension.

Generated command manifests are under `payload-selection-v1/`:
`DEVELOPMENT_CANDIDATE_MANIFEST.json` and `CONFIRMATION_CANDIDATE_MANIFEST.json`.
Both record their canonical protocol digest. Selection uses only complete new
validation; the existing30-episode development scores cannot be relabelled as
50-episode validation. Unknown/duplicate IDs, extra checkpoints, unequal native
training configurations, pending candidates or unexplained missing files fail
closed. A genuine terminal missing candidate may retain an explicit
`unavailable_verified_terminal` disposition with a separately inspected terminal
process and retained failure receipt. It still counts as an allowed candidate
and its failure is retained; it is not replaced by an extra convenient checkpoint.
No method needs every seed to win; it needs the same declared selection opportunity.

## Strict safety-first rule, no mean-risk loophole

Risk qualification is nominal violation≤.05 AND **EACH** registered stress
condition violation≤.10. Stress mean risk is diagnostic only. Offset risk33%
does not qualify merely because other conditions reduce the mean below10%.
Among qualified candidates: macro safe success across all five conditions first,
then macro capped completion time, then candidate ID only for an exact tie.
Native return is retained but is NOT an extra tie priority.

If no candidate qualifies, retain the full Pareto front over per-condition risk,
macro task success and capped time. The same task/time ordering selects an
explicitly labelled research representative within that front; it is
`pareto_tradeoff_not_risk_qualified`, **not eligible for deployment**. No threshold
is loosened. Every failed/timeout episode keeps the full4s capped horizon, so a
motionless safe policy or fast failure cannot masquerade as fast task completion.
Train seeds, not evaluation episodes, remain the statistical replication unit.

## Common validation CLI and exact frozen provenance

For each registered candidate, `payload_frozen.py` checks native method/seed/
phase/configuration and actual transition count. It requires the atomically
completed matching full checkpoint (published after its policy file), but loads
only the small policy state for evaluation. Policy SHA256 and native config
digest are recorded. Actor, online/EMA margin, critics/targets, normalizer and
dual values are fingerprinted before/after. All modules are frozen; geometry
input gradients remain enabled (no global inference mode). The normalizer is
the native explicit-unit identity45D observation, not a fitted test normalizer.
Training Python/NumPy/Torch CPU/CUDA randomness is isolated; each split has its
own registered action-noise stream. Actual evaluator source digests and software
versions are recorded. Each condition remains separately reported.

Example on a host with the native files, after source pin/resource admission:

```bash
python -m scripts.paper_crax_payload.payload_frozen \
  --plan configs/paper_crax_payload/payload_selection_v1.json \
  --training-config configs/paper_crax_payload/payload_v1.json \
  --cohort development --candidate-id development:sac_lag:41027:000060000 \
  --run-root /home/carla/4T/SAFEFLOW2027_PAPER/runs \
  --split validation --device cpu --source-commit EXACT_EVALUATOR_COMMIT \
  --output NEW_VALIDATION_DIRECTORY/development_sac_lag_41027_000060000.json
```

The prepared manifests contain the30/45 matching commands. Run them only when
the native checkpoints are published; do not create dummy missing results for
still-live jobs. The selector never launches processes:

```bash
python -m scripts.paper_crax_payload.payload_select \
  --plan configs/paper_crax_payload/payload_selection_v1.json \
  --cohort development --validation-directory COMPLETE_COMMON_VALIDATION_DIR \
  --output NEW_PAYLOAD_SELECTION_FREEZE.json
```

All30 registered IDs must be accounted for (complete validation or explicit
verified terminal dispositions). Result includes every candidate, failure,
per-condition score, Pareto front and a tamper-evident freeze digest. The root
paper `MODEL_SELECTION.json` may reference this artifact; this helper does not
overwrite the root registry/collector or discard fixed endpoint tables.

## Independent generalization, locked until selection

Final selected-policy testing is100episodes/condition at fresh71027–71126
layouts and a separate noise stream. Conditions are preregistered before use:
nominal, radius.075, payload half-extents(.032,.022,.024), action-scale.80, and
the fixed offset(.045,0,.04). Radius/shape/scale values were not evaluated by
this work. The offset parameter was already used in development; it is NOT
called an unseen parameter. Its new test episodes are independent of selection.
Only the action-scale error family is used; no additional delay family is mixed in.

Use the same standalone command with `--split independent_test` and
`--selection-freeze NEW_PAYLOAD_SELECTION_FREEZE.json`. Execution requires the
complete unchanged freeze and the exact selected policy bytes for that method.
An altered/missing freeze, another checkpoint or development/test scores supplied
to selection are rejected. A Pareto representative stays labelled non-qualified
in its test report. Nothing in the manifest runs independent tests automatically.

## Offset is only a candidate until actual route proof

No route claim is made from a payload XYZ polyline. The native arm AND payload
interval geometry must be evaluated. After common selection, the locked
`--split route_calibration --selection-freeze ... --trace-directory EXTERNAL_4T_DIR`
evaluates only the exact selected policies on common validation-only62027–62036
layouts, ten nominal episodes/method. It records private witnesses with initial
full observed physics state/config/seed/policy hash, raw and executed actions,
joint/payload observations and TCP per control interval. Optional small-subset
action witnesses are also supported by `--trace-episodes N` for ordinary splits.
Witness paths include split/candidate ID to avoid collisions across methods.
Each executed action is logged with a per-interval flush; full traces stay on4T, compact summaries
and witness hashes can be published. Interrupted evaluations preserve their
observed query prefix/failure and are not silently retried.

`payload_rollouts.replay_open_loop_witness` performs **real simulator open-loop
replay of executed nominal actions**, not oracle-selected actions. It first uses
the same strict nominal task configuration and restores the recorded observed
state; explicitly checks arm/finger positions/velocities, payload pose/velocities
and goal; THEN applies the fixed obstacle translation. It asserts every policy
observation coordinate except obstacle XYZ is unchanged AND the physical
obstacle really has the translated pose. It does not accidentally restore away
the offset or pretend this diagnostic is a native training resume with altered
configuration. Bullet solver warm-start caches are not claimed bitwise portable.
Every real replay step uses the original20-substep arm/payload interval minimum.
Initial states made unsafe by the declared offset remain explicit failures, not
silently resampled, and prevent qualification as a valid alternative-route
benchmark. Stops are native terminal stops, not action reranking.

The usual route is the preregistered8-sector x/z crossing proxy, modal over
successful nominal paths with equal method weights and deterministic ties.
Require at least50% of that usual-route pool to become violating under real
offset replay, and at least80% safe success for the SAME fixed high detour
controller on identical calibration layouts. No oracle feedback chooses learned
actions, no new offset is fitted after test responses, and no alternate path is
searched per episode. `run_predeclared_alternative` supplies the same controller
as the original feasibility probe, with separately counted real control steps.
`qualify_offset_route` validates complete selected-policy/offset provenance and
common-layout evidence. Missing successful nominal routes or failed criteria
yield **offset geometry stress only / alternative-route claim not verified**.
This diagnostic has not been executed; it cannot be claimed as existing proof.

## Focused implementation evidence and cost

Seven focused tests cover equal30/45 opportunities, each-stress risk, return-free
tie ranking, complete/frozen test authorization, explicit failed candidates,
route-qualification refusal when not blocked, and actual executed-action/offset
bookkeeping against a FAKE simulator (not a claimed physical route result).
The fingerprint test catches model/dual mutation. No historical test suite or
18,000 episodes were rerun. Actual new environment transitions0, simulator
queries0, physical robot interactions0. Synthetic test fixtures and manifest
generation do not become training or independent-test evidence.
