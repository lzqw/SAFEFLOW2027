# P4.4 seed-2028 pilot evidence

This directory is the transferred evidence for the preregistered P4.4 R0/R1 pilot on
`SafetyPointGoal1-v0`. Both arms were trained from zero to exactly 30,000 environment steps and
28,000 updates with seed 2028 from clean source commit
`9ba6250f44d008631612c25ffe01c4f3d35fe008`. The locked protocol SHA-256 is
`005c42348a22f664448c688b3cecc16eba7cbbea55d7727537caeef5d3b69457`.

The strict result is `stop_after_seed_2028_pilot`, with expansion authorization set to false.
R1 failed three frozen gates:

- executed dangerous outward speed was `0.170585` versus R0 `0.109496`;
- crossing rate was `0.642273` versus R0 `0.515198`;
- mean external cost was `115.85` versus R0 `104.25`.

R1 retained tangent variance, success and stall stayed within tolerance, CVaR-20 passed, both
arms remained numerically stable, and each final held-out set contained 2,000 positive and 4,000
negative labels. Because every pilot gate was required, no seed-2027/2029 or 50k expansion was
run.

Evidence layout:

- `P4_4_PILOT_DECISION.md` and `pilot_decision.json`: human-readable and strict machine decision;
- `R0/` and `R1/receipt.json`: clean-source launch and completion receipts;
- `R0/` and `R1/training/`: configs, progress, six 5k diagnostics, numerical status, held-out
  evidence, manifests, and summaries;
- `R0/` and `R1/postprocess/evaluation/`: fixed seeds 12027 through 12046, aggregate evaluation,
  per-episode diagnostics, and raw trajectories;
- `R0/` and `R1/postprocess/audit.json`: 64-state x 256-action-sample four-mode decomposition;
- `checksums.sha256`: SHA-256 seal for every transferred artifact other than the manifest itself.

The full checkpoints and replay buffers remain at the remote experiment root
`/root/autodl-tmp/SAFE/experiments/p4.4-pilot-9ba6250`. Audit records bind the R0 and R1 final
checkpoints to SHA-256 values `e0be8d0900b49046b3c8b42d1e05ea9c3267eba3d7c91584befdfba4b442feaf`
and `a083bfe750567b8246a9074f57645a9c960748968f50ce336cb44c913df24d8f`, respectively.

Run `sha256sum --check checksums.sha256` from this directory to verify the transfer seal.
