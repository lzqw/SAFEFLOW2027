# Payload offset-route diagnostic — 2026-11-12

The three exact selected policy bytes were evaluated on the predeclared
validation-only route-calibration layouts (10 layouts per method). Nominal
executed actions were replayed in native arm-and-payload physics after the
fixed obstacle translation; actions were never selected from the shifted
geometry. The complete witness-based diagnostic is stored in
`payload-route-proof-20261112-1252/PAYLOAD_ROUTE_PROOF.json`.

Result: **`offset_initial_admissibility_not_verified`**. At least one declared
shifted initial state is unsafe before any action, so the stricter alternative
route claim is not verified. This is retained as an explicit offset-geometry
stress result, not converted into a favorable route-generalization claim.

No independent-test result or model-selection decision was changed by this
diagnostic, and no physical Piper motion occurred.
