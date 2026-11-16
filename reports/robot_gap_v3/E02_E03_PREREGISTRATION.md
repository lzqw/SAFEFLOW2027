# E02/E03 frozen experiment registration, before new evaluation labels

2026-11-14. No new RL, new policy seeds or hardware operations. CPU-only on
A100 host with one Torch thread; foreign GPUs remain untouched,5080 unused.
All large/raw output and temporary files go to the A100 data filesystem.

## E02: equal opportunities on existing useful Payload candidates

Retain previously selected Iso60K and EA120K (training seed41027), not their
regressed300K endpoints. Both consumed300K Payload training and share a300K
Reach parent. The existing identity hashes are bound in
`configs/paper_crax_payload/piper_frozen_frontier_v3.json` before any new labels.
This is a good-candidate experiment, not across-training-seed statistics.

Exactly THREE common execution multipliers each:0.6,0.8,1.0. They multiply the
frozen native stochastic proposal, then the specified condition's scale, then
clip to[-1,1]. No action search, noise retuning, critic update or new candidate
weights. Same50 common validation layouts81027–81076 for nominal,55mm sphere,
and native0.85 execution variation. Old selection labels61027/71027 are not
reused. Episode-local action seeds are shared across methods/scales.

After all900 validation episodes, freeze selected scales under nominal AND
every-stress-condition empirical risk caps5% and10%. Primary eligibility is
each-condition violation≤cap with no inadmissible layouts; maximize macro safe
success, then minimize failure-capped time, then smaller multiplier. If no
candidate qualifies, explicitly choose the lowest worst-condition violation
tradeoff, not a claimed risk solution. A zero-task eligible candidate is labelled
stagnant. Freeze ALL three scale identities as the predetermined frontier.

Only then independently evaluate ALL THREE frozen settings/method on100 new
layouts91027–91126/condition (1800 episodes), irrespective of test outcomes.
Retain the entire frontier, including dominated points. Do not reselect using
these labels. Risk-cap selection is NOT proof of equal actual risks: compare
actual rates and Wilson95 intervals; say matched-risk advantage only if the
empirical task/risk frontier actually supports it, not because both were given
the same budget label. No significance across training seeds is claimed.

Report safe success, actual violations, all-failures-at-full-horizon capped time,
payload path INCLUDING the initial-to-first-step segment, success-only path as
a supplementary diagnostic, minimum interval clearance, clipping and counts.
Violations and timeouts get the full4s horizon, so quick failure cannot win time.
No hardware timing or physical performance is inferred from CPU simulation.

## E03: actual parameter extrapolation, no tuning

Use the existing fixed400K matched Reach-adaptation endpoints, Iso and EA,
not a newly selected checkpoint. These were trained on45/50/55mm radii;
old validation used45/55mm and execution multipliers1/0.85. New geometry family
is57.5/60mm, both outside the training maximum55mm. One new execution-error
family is scale1.15 at nominal geometry, with native clipping retained.

100 fixed new layouts101027–101126 for each of these THREE conditions and each
method:600 attempted episodes. No new nominal-condition test is repeated.
Native static safe-start/reachable-safe-goal rejection remains unchanged.
Record an unadmissible layout explicitly, never replace it by an easier seed
or lower radius. Show attempted/admitted denominators; shared static admissibility
must agree across methods. No further unseen-family substitution based on
policy outcomes and no claim of globally safe geometric generalization.

This is parameter-family extrapolation under the same provisional simulator,
not calibrated dynamics or sim-to-real transfer. Safety flags remain disabled.
E04 alternative-route evidence is not claimed by these sphere-radius tests.

Finite cost: maximum3300 new simulated episodes, zero RL transitions. Keep raw
condition files/trajectories on the data disk, publish compact CSV/plots/receipts
through the local machine to GitHub. New tests cover failure-capped time,
initial path segment, honest selection labels and zero-count risk uncertainty.
