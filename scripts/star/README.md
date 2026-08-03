# STAR P0/P1 server reproduction

This directory reproduces only the P0 and P1 engineering gates. It does not launch STAR P2+
training. The development branch starts from OmniSafe commit
`15603dd7a654a991d0a4648216b69d60b81a6366`; FLAC
`61a25eef42935fc395b1e04071e2779311fcb705` and SHAPO
`1bb770437bbbe40a89938661343744243e164bf9` are read-only references.

## Confirmed server layout

The rented server's 50 GiB XFS data disk is mounted at `/root/autodl-tmp`. All mutable state is
kept below `/root/autodl-tmp/SAFE`:

```text
/root/autodl-tmp/SAFE/
  code/SAFEFLOW2027/
  code/FLAC/
  code/SHAPO/
  envs/star-p0-p1/
  cache/{pip,conda,torchinductor,tmp}/
  runs/{active,completed}/
  artifacts/{p0,p1}/
```

The system disk is not used for environments, caches, runs, checkpoints, or videos. P0 requires
at least 25 GiB to remain free on the data disk.

Create the three checkouts on the data disk and detach the two references at their audited commits:

```bash
cd /root/autodl-tmp/SAFE/code
git clone git@github.com:lzqw/SAFEFLOW2027.git
git -C SAFEFLOW2027 switch codex/star-p0-p1
git clone https://github.com/bytedance/FLAC.git
git -C FLAC checkout --detach 61a25eef42935fc395b1e04071e2779311fcb705
git clone https://github.com/montrealrobotics/shapo.git SHAPO
git -C SHAPO checkout --detach 1bb770437bbbe40a89938661343744243e164bf9
```

## Environment

From the server checkout:

```bash
cd /root/autodl-tmp/SAFE/code/SAFEFLOW2027
SAFE_ROOT=/root/autodl-tmp/SAFE scripts/star/setup_server_env.sh
```

The environment is Python 3.11 with PyTorch 2.8.0/CUDA 12.8. Safety-Gymnasium 1.0.0 has two
upstream Python 3.11 compatibility issues: mutable NumPy dataclass defaults and its exact
Pygame 2.1.0 metadata pin. `patch_safety_gymnasium_py311.py` applies a version-checked,
idempotent patch; Pygame 2.6.1 is used because 2.1.0 has no Python 3.11 wheel. NumPy remains
1.23.5 for Gymnasium-Robotics 1.2.2, so MoviePy is held on 1.0.3. The complete resolver guard is
`constraints-py311.txt` and the captured result is `environment.lock.txt`.

Do not install `mujoco_py`. The four tasks use Gymnasium 0.28.1, Safety-Gymnasium 1.0.0, and
MuJoCo 2.3.3.

## Gates

Run each gate from the server. Runtime outputs stay on the data disk.

```bash
scripts/star/launch_p0_p1.sh p0-capture
scripts/star/launch_p0_p1.sh p0-env-smoke
scripts/star/launch_p0_p1.sh p0-saclag-5k
scripts/star/launch_p0_p1.sh p0-benchmark-point
scripts/star/launch_p0_p1.sh p0-benchmark-mujoco
scripts/star/launch_p0_p1.sh p1-tests
scripts/star/launch_p0_p1.sh p1-smoke-20k
```

The throughput benchmark holds global batch size and UTD constant while changing concurrency.
Every worker pre-fills replay, performs real warmup updates, waits at a barrier, then measures at
least 600 seconds of rollout, backward updates, and periodic evaluation. It records aggregate FPS,
update/s, GPU utilization and memory, CPU quota use, RAM, temperature, power, and the estimated
cost per million transitions:

```text
1.48 * 1e6 / (3600 * aggregate_env_fps)
```

P1 adds a dedicated `FailureCostMujocoEnv` for Ant-v4 and Walker2d-v4. Only unhealthy termination
has cost 1. A time-limit truncation has cost 0. Replay stores `terminated` and `truncated`
separately, and the Bellman `done` mask is exactly `terminated`, including the rare case where
termination and truncation coincide.

## Monitoring and publication

Port 6006 is reserved for TensorBoard:

```bash
scripts/star/launch_p0_p1.sh tensorboard
```

Port 6008 is reserved for read-only monitoring and is never a training dependency. Runs are safe
to execute in `tmux`; browser disconnection does not affect them.

After a gate passes, sync only source and small reports back to the host. Do not copy environments,
caches, replay, checkpoints, videos, or full run directories. Commit and push from the host branch
`codex/star-p0-p1` to the private `lzqw/SAFEFLOW2027` repository.
