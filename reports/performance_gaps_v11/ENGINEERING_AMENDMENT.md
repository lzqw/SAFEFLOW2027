# Explicit device-RNG fork mapping, before any new training transition

The first A100 launch encountered `IndexError` restoring a two-GPU parent RNG
into a single visible shared-GPU slot. Native parent manifests show `cuda:1`,
with no CUDA_VISIBLE_DEVICES mask. Both Reach and Payload failed during setup,
before the training loop; no new env.step calls, learner updates or DEV episodes
were executed. Original failed directories/logs and queue history are preserved.

The repaired explicit fork maps the parent's **cuda:1 generator** onto the one
visible new shared slot. Python, NumPy and CPU Torch RNG remain unchanged.
This is a hardware-index mapping, not a new RNG draw or candidate/seed search.
Own v11 resume keeps its already-single-device native RNG without remapping.
Panda's native generator index is0. Historical restore_rng/loaders are unchanged.

The reviewed A100 relaunch uses separate attempt2 paths and a new source commit,
not an automatic queue retry. The original queue must be terminal and its
native handles absent before relaunch. The600K/900K step budgets do not change.
