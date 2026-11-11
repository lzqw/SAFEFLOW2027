# Native late-stage continuation choice,2026-11-11

Select the unchanged first S-EA package (rho0.25,EMA0.005,ramp10k–30k)
for the two planned geometry continuations. Both EA/Local fixed300k seed
pairs are now available. Iso2028 is still finishing an engineering recovery;
its absence is not used to claim superiority over a two-seed Iso baseline.
There is no favorable600k/best-checkpoint substitution or new formal run.

| Fixed300k trial | Nominal SS | Stress mean SS / violation | Radius8cm violation | Native / accounted training violations |
|---|---:|---:|---:|---:|
| EA2027 |100%|97.5% /2.5%|10%|1007 / at least1022|
| EA2028 |100%|98.75% /1.25%|5%|1001 /1001|
| Local2027 |100%|88.75% /11.25%|40%|756 /756|
| Local2028 |100%|83.75% /16.25%|65%|817 /817|

The unit of comparison is the training seed:the two-seed EA mean radius8cm
violation is7.5%,versus52.5% for Local;stress safe success is98.125%,versus
86.25%. Each stress condition has only20 development episodes per seed.
These are encouraging matched development outcomes,not new-seed confirmation
or evidence of1M stability. Nominal has30 episodes/seed. All middle failures
remain in the raw tables and curves,including Local2027's90% radius8cm
violation at200k and EA2028's60% at100k.

The safety-training tradeoff is unfavorable:EA's accounted mean training
violations are at least1011.5 versus786.5 for Local (+28.6%). A monotone
cumulative counter already above350 at300k cannot reach the350/1M effort
target merely by continuing training. Do not reset counters or hide the
discarded CUDA-interruption tail. Long runs test late performance and risk;
they do not erase that failure.

No additional short candidate is selected at this point. Nominal task
learning is intact in both EA seeds,and no positive-Q blow-up,dominant dual
overshoot or persistent global cost-Q saturation has been demonstrated in
these fixed300k probes. The allowed0–4 single-factor candidate budget stays
reserved for an actual relevant late-stage issue rather than a blind sweep.
The small cost-critic probe below is not a calibration certificate;EA's
underprediction on some failed replay actions remains a limitation.

The native long matrix stays Iso/EA x2027/2028,300k-to1M (2.8M incremental).
Iso2027 is already running on A100 GPU1. EA2028 can use4080 now;EA2027 waits
for the currently running A100 GPU0 recovery slot,then starts automatically.
Iso2028 will receive its declared700k continuation after its protected300k
snapshot exists,and can use4080 to keep both methods represented on both
hosts. This is a resource dependency,not a scientific performance threshold.

At10:45 the4080 queue observed CPU load25.95/28 and waited;at10:48 it had
fallen to8.09,with over50GiB available RAM. A30-minute resource backoff would
miss that admitted slot. The waiting4080 controller is replaced explicitly
with the same finite trial and guards but a5-minute admission retry. No native
training child had started,and none is stopped. External checks stay30min.

All use the same3c308e4 learner/geometry/profile. Driver94ffff3 only adds
explicit engineering recovery and initial-counter metadata. Large snapshots
stay on the servers/4080 4T. The conditional new-seed9M remains unqueued.

## Bounded frozen cost-critic probe

Three protected300k full snapshots,CPU-only,one128-action uniform replay
batch and one128-action failed replay batch each,no SGD and no physics.
Each row below summarizes256 predictions (two critic heads x128 actions).

| Trial | Failed-action mean Q_C | Failed predictions below0.1 | Mean MSE/BCE logit-gradient factor |
|---|---:|---:|---:|
| Iso2027 |0.66259|5.47%|0.28410|
| Local2028 |0.64986|3.13%|0.30871|
| EA2028 |0.28594|30.08%|0.27798|

The gradient factor is2q(1-q),the analytic ratio for the same binary target;
it is not an optimizer experiment or proof that BCE would improve returns.
Uniform-batch mean Q_C values are0.01280,0.00307,0.00822 respectively;low
uniform mean alone does not establish failure-action saturation. Sampled
indices are fixed by NumPy default_rng(9830027),sequentially uniform then
failure batches in the displayed trial order. This is one retrospective
snapshot per trial,not evidence of persistent calibration over training.
