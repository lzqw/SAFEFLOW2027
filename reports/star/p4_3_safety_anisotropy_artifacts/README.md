# STAR P4.3 compact evidence bundle

This directory is a compact, verbatim mirror of the formal remote experiment at
`seeta-nmb2:/root/autodl-tmp/SAFE/experiments/p4.3-v2-43f48fc`.

- Protocol: `p4.3-safety-anisotropy-v2`
- Protocol SHA-256: `a05cb7dcaaee95cdbb75e005371e07ea04cfb59e765c4f5d70b9772d8bd29430`
- Training source commit: `43f48fc05c3d63f4b13d08e88b1f4af707aba7d7`
- Formal runs: S0/S1/S2 at seeds 2027, 2028, and 2029; 50,000 environment steps each
- External evaluation seeds: 12027 through 12046
- Strict decision: `hard_stop_reward_td_divergence`

All nine training receipts, compact run configurations and diagnostics, nine external evaluation
JSON files, nine held-out directional audits, completion markers, and the post-processing log are
included. The authoritative report and machine-readable decision are under `postprocess/` and are
also copied to the parent `reports/star/` directory for convenient discovery.

The formal summarizer was rerun independently against the same sealed inputs. Its output was
byte-identical to the watchdog output:

- Decision JSON SHA-256: `ae8f324f93bb36732cb6c51b0980cdb37c5cf149b9b84d8c1cb8269bd5457505`
- Decision Markdown SHA-256: `95b3b900c12f3adb26b66ec0da1735af11fcb8890ce5399d3d5e65d18cf8e4ac`

The nine approximately 39 MB checkpoints, post-processing work directories, and trajectory CSVs
are intentionally excluded from Git. The checkpoints remain on the remote experiment volume;
`postprocess/checkpoint_sha256.txt` records their identities. Absolute remote paths inside the
decision JSON are retained unchanged so that the copied file remains byte-identical to the formal
output.
