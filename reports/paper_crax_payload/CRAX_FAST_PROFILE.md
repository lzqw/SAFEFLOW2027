# CRAX complete-training profile — pending measurement

No verified fastest configuration or2000/5000SPS claim yet. A100 both GPUs are
online and shared with other projects; do not interrupt foreign work. Initial
inspection found about78GiB free per card, driver535.230.02. Pinned official
CRAX63504b5 is cloned; dependencies are being installed in a NEW environment
`/home/lzqw/SAFEFLOW2027_PAPER/env`, not the historical Arm environment.

One registered short profile per Balanced(E256/B1024/G4), Fast(E512/B1024/G4),
Learn-heavy(E128/B512/G8), F1. Exposure16/8/32 differs: a speed difference is NOT
pure engineering acceleration. Profile-only workloads explicitly remove critic
preheat and force full geometry to time the expensive path without pretending
random uncalibrated margins have5000 trained updates. All actual profile sampling
and SGD counts are charged, separate from scientific candidates. Candidate
learning retains the registered1000critic-only/1000→5000margin ramp.

Reports will distinguish JIT/reset/firststep compile, steady action/env/learner
work, actual optimizer rate, walltime including eval/save, allocated memory and
shared GPU load. No simulator-only benchmark is a full-training speed result.
Torch/JAX conversion requires same device and actual pre-reset next observation.

All new CRAX Torch Flow controls use `float32_matmul_precision=high`, recorded
in each configuration. This permits TF32 internal matmul arithmetic on supported
CUDA hardware while retaining float32 outputs; it is not bitwise-equivalent to
highest precision, nor a separately measured TF32 causal speedup. See
[installed-version PyTorch2.7 API](https://docs.pytorch.org/docs/2.7/generated/torch.set_float32_matmul_precision.html).
Historical Arm/Payload running jobs are not modified. JAX's native precision
configuration is separately recorded; different update exposures remain explicit.

The CRAX geometry label is a risk-positive **endpoint-clearance proxy in metres**
against native hazard boundaries, not a claimed control-interval minimum and not
the cumulative cost critic. Native costs include every native source unchanged.
Payload separately uses genuine physical-substep interval minimum distance.
