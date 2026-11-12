# CRAX runtime setup: actual GPU interoperability passed

New A100 environment: `/home/lzqw/SAFEFLOW2027_PAPER/env`, created with system-site
visibility. No Torch upgrade, global package install, old experiment-env edit,
foreign-process interruption, firewall change, or physical robot interaction.

The original public-index A100 download was approximately0.18–0.21MB/s. Exact
public artifacts were instead cached on4080's4T disk. Eleven resolved CUDA12
wheels totaled2,723,968,861bytes and completed in804.64s using bounded parallel
downloads. All ordinary wheels matched official PyPI SHA256; Gym0.26.2 has no
official prebuilt wheel, so its cached wheel was built from the official,
SHA256-verified source archive and is explicitly distinguished in the manifest.

All90 final artifacts,3,281,793,156bytes, were reverified on A100 against the
authenticated official-source manifest. The89-package complete lock includes
hashes and a NumPy2-compatible SciPy1.17.1. No multi-GB artifact was staged on
the local host. Data used a temporary read-only HTTP listener bound only to
4080's172.21.221.238:18881, serving public wheelhouse contents. The listener was
stopped after the final canonical-cache verification; no persistent service
remains. Slow duplicate serial cuDNN transfer was stopped only after an exact
full-wheel hash-verified chunk reconstruction made every canonical artifact
available. Download partials/logs remain, not scientific training failures.

Installation succeeded offline into the new environment with `--no-index
--require-hashes` and `crax_runtime_requirements.lock`; full native installer
report/log are `/home/lzqw/SAFEFLOW2027_PAPER/install-offline-report.json` and
`install-offline.log`. Inherited system-site dependency warnings (old numba,
Anaconda/botocore etc.) were retained, not broadly repaired. This is not a
claim that every unrelated inherited package passes `pip check`.

The4.91s actual single-process A100 check passed JAX CUDA JIT, device-side
Torch→JAX→Torch DLPack, Torch GPU convolution and CPU NumPy logging conversion.
Torch remains2.7.1+cu118 at `/home/lzqw/anaconda3/lib/python3.11/site-packages/torch`;
the process loaded cuDNN92801 from the new CUDA12 runtime. Flax0.10.6,
Optax0.2.4, Orbax0.11.13, MuJoCo/MJX3.11.0, NumPy2.2.6 and SciPy1.17.1 imported.
See the actual JSON receipt, not this prose alone.

Tested process environment: `CUDA_VISIBLE_DEVICES=0`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; put the new environment's
`lib/python3.11/site-packages/nvidia/*/lib` directories before existing
`LD_LIBRARY_PATH` **before Python starts**. This is process-local; no shell
startup/global configuration was changed. The same loader configuration is
needed by real training commands; do not silently fall back to a different
inherited cuDNN. Library coexistence was tested, not assumed.

This setup/GPU-array check constructed **zero environments**, performed
**zero environment steps**, **zero training transitions**, and **zero real
interactions**. It is not an environment-semantic test, MJX training-speed
profile or learning result; those remain separate counted operations.
