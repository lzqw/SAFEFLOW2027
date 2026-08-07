# P4.4 outward-rectification evidence

The `pretraining/` directory is the read-only decomposition of all nine immutable P4.3 final
checkpoints. It was run on the remote RTX 3090 before any P4.4 R0/R1 training, using audit source
commit `8ab48e839cecd15009548ef00b0ae04988d6d773` and the locked protocol SHA-256
`005c42348a22f664448c688b3cecc16eba7cbbea55d7727537caeef5d3b69457`.

The strict aggregate decision is `p4_4_training_authorized`: all checkpoints supplied at least
58 valid probability/cost critic action-gradient cosines, and 260 of 566 pooled valid cosines
were negative (`0.4593639576 < 0.5`). No P4.4 training preceded this decision.

Raw checkpoint files remain on the server. The tracked JSON files contain checkpoint paths,
checkpoint SHA-256 identities, selected replay indices, per-state four-mode metrics, and critic
alignment records. `pretraining/checksums.sha256` seals every transferred evidence artifact.
