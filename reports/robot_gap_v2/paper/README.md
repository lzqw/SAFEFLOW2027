# PointGoal paper material

`pointgoal_three_seed_learning` shows all three training seeds at all five
common validation checkpoints. Thin lines are individual training seeds; bands
are sample SD across three training seeds. Each policy's point already averages
the same five validation seeds,50 episodes/seed. The nine fixed10M endpoints
are in `pointgoal_fixed_10m_by_seed.csv`; their three-seed summaries are separate.

| Fixed10M method | Goals/episode, mean ± training-seed SD | Cost, mean ± SD |
|---|---:|---:|
| SAC-Lag |36.24±0.12|41.28±2.11|
| Stable-Iso |11.87±13.79|22.26±2.51|
| EA |27.47±8.58|21.97±5.07|

EA preserves more task progress than Iso at roughly similar mean cost; its
cross-training-seed variability remains substantial. SAC at10M completes more
goals but exceeds the25 cost budget on average. Do not replace these endpoint
results with the best8M EA or2M SAC policy.

`pointgoal_selected_independent` instead shows the previously frozen selected
policy of each method: EA51029/8M, Iso51028/10M, SAC51029/2,016,256. Each has500
independent episodes. Test-seed dots are evaluation variation of that one
selected policy, not three trained replicas. No selection was repeated.

Original complete training SPS (compile/evaluation/save included), averaged
across the three runs: SAC10,269; Iso2,061; EA1,165. These are the historical
native PointGoal jobs under uncontrolled shared GPU load, not Piper throughput
and not batch1 inference latency. The efficiency CSV retains per-run timings,
update counts and missing stage timings as blank; it invents no missing data.

`pointgoal-latency-cpu.json` measures the exact selected Flow policies on the
A100 server CPU, one thread, including normalization, geometry/tail/VJPs and
host return: Iso p50/p951.849/1.890ms, EA9.617/9.764ms. It does not claim latency
on the physical deployment computer. The old4M common-state pool is a timing
fixture only. EA uses4 base integration steps; a complete action has20 measured
velocity forwards and4 batched-head VJP traversals including the tail work.

`pointgoal-native-sac-latency-cpu.json` completes the exact selected native SAC
measurement on that same timing fixture: p50/p950.301/0.337ms;20warmup+200
measurements. It includes native normalization, stochastic policy, RNG split
and host return; JIT compile0.154s is excluded. No arm policy is substituted.
The JAX CPU runtime/thread configuration differs from the Torch CPU timing;
this is not an exactly matched backend or certified deployment latency.

Generation uses `scripts.paper_crax_payload.pointgoal_paperpack`, completed
formal receipts, the existing common-selection JSON and locked independent
summary. Figure inputs and receipt hashes are retained. A trailing literal
`\\n` in the old independent-summary JSON was corrected to a real file ending;
all scientific values are unchanged. No training or simulation was executed to
produce these figures.

The final selected-model64×32/four-intervention mechanism is now complete in
[final_mechanism/](final_mechanism/README.md). Actor,EMA-margin and normalizer
are bound to the exact selected policies. Native CPU execution took50.73s and
16,704 actual controls,zero RL. This avoids the unavailable8GiB GPU window,
not native physics or queries; CPU/GPU Gaussian generators are not bitwise
equivalent. No earlier GPU mechanism query/output existed before dispatch.
The historical4M study is not relabelled.

Endpoint VN/VT changes Iso0.3482→0.3274 and EA0.4514→0.4308 for reference→
pullback. Local/direct ratios are lower still; this does not demonstrate
pullback superiority. Native mean costs change Iso0.009484→0.009440 and
EA0.009718→0.009733: no consistent improvement. All methods/samplers have
the same6.25% cost-positive fraction. More importantly, within each of64
states the event rates are identical across all four samplers (60 always-zero,
4 always-positive states). This fixed pool cannot show an event-safety gain;
it is retained without reselection or further sampling. Native proxy labels
are not arm control-interval minimum distances or safety certificates.

Reproduce the eight-row table and figure from its completed records without
loading models, simulating, choosing states or drawing more actions:

```bash
python -m scripts.paper_crax_payload.pointgoal_mechanism_pack \
  --input /path/to/retained/pointgoal-final64x32 \
  --selection reports/paper_crax_payload/pointgoal-formal-selection-20261113/POINTGOAL_MODEL_SELECTION.json \
  --output reports/robot_gap_v2/paper/final_mechanism
```

The command refuses the old32×16 study, partial receipts, wrong model steps,
missing sampler arms or duplicate states. It preserves nondiscrimination:
even a mixed-cost pool does not show an event-safety effect if every sampler
has identical event rates within each state. The actual results and original
source records are now included.

`selected_examples/` now separately delivers two genuine native trajectories
on the prospectively fixed first development layout61027. Iso36goals/17.76cost,
EA37goals/28.31cost: EA exceeds the25 budget in this example. The unfavorable
episode is retained without resampling. All2000 control records and model/
physical-state hashes are included. This illustrative CPU rollout is not the
500-episode independent test or final-model mechanism study.
