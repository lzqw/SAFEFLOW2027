# Payload common validation and selection — 2026-11-12

The registered development cohort was evaluated without relabelling native
development scores: 3 methods × 2 training seeds × 5 checkpoint candidates,
with 50 episodes on each of five common conditions (7,500 validation episodes
in total). All 30 candidates completed; no candidate was silently omitted and
no independent-test episode was used for selection.

Risk qualification required nominal violation ≤5% and **each** stress
condition violation ≤10%. Selection then ranked macro safe success and capped
completion time. The resulting freeze digest is
`143372ca6767a5f9da6e8d125fbcdf4c4dafaea94c6ffea6b3e2ffe6ee22b227`.

| method | selected candidate | selection status | macro safe success | per-condition violation (nominal, radius, payload, scale, offset) | deployment eligible |
|---|---|---|---:|---|---|
| SAC-Lag | seed 41028, 300k | risk qualified | 0.956 | 0.00, 0.06, 0.08, 0.02, 0.06 | yes |
| Stable-Iso | seed 41028, 120k | Pareto trade-off, not risk qualified | 0.852 | 0.00, 0.02, 0.00, 0.00, 0.72 | no |
| EA | seed 41028, 240k | Pareto trade-off, not risk qualified | 0.908 | 0.00, 0.06, 0.08, 0.00, 0.32 | no |

The offset-route condition is reported separately; the large Iso/EA offset risk
is why those candidates are not deployment-eligible under the preregistered
rule. Their fixed candidates and full evidence remain retained for scientific
comparison. The exact freeze is stored in
`payload-selection-20261112-1201/PAYLOAD_SELECTION_FREEZE_DEVELOPMENT.json`.

The next locked step is independent testing of these exact three selected
candidate bytes on fresh layout seeds. This does not change selection and does
not authorize physical Piper motion.
