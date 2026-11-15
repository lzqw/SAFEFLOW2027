#!/usr/bin/env bash
# Resource-path adapter only: keep the registered CVPO source/config unchanged.
set -euo pipefail
task_root=/home/carla/4T/SAFEFLOW2027_NONHARDWARE_V4
logical_root=/storage_0/SAFEFLOW2027_lzqw_uid1007/nonhardware_v4
task_env="$task_root/e08-fsrl/env"
arm_env=/home/carla/4T/SAFEFLOW2027_ARM/env
shared_torch_env=/home/carla/SAFEFLOW2027_4080/env
test -x "$task_env/bin/python"
test -f "$task_root/e08-fsrl/formal-source-v1/configs/paper_crax_payload/cvpo_formal_v4.json"
mkdir -p "$task_root/e08-fsrl/numba-cache" "$task_root/e08-fsrl/jax-cache" \
    "$task_root/e08-fsrl/cache" "$task_root/e08-fsrl/mpl-cache" "$task_root/e08-fsrl/tmp"
exec bwrap --unshare-user --new-session \
    --ro-bind /usr /usr --ro-bind /etc /etc --ro-bind /sys /sys \
    --symlink usr/bin /bin --symlink usr/sbin /sbin \
    --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
    --proc /proc --dev-bind /dev /dev --tmpfs /tmp \
    --ro-bind /home/carla/anaconda3 /home/carla/anaconda3 \
    --ro-bind "$arm_env" "$arm_env" \
    --ro-bind "$shared_torch_env" "$shared_torch_env" \
    --bind "$task_root" "$logical_root" \
    --setenv PYTHONDONTWRITEBYTECODE 1 \
    --setenv PYTHONPATH "$logical_root/e08-fsrl/overlay:$logical_root/e08-fsrl/upstream-fixed:$logical_root/e08-fsrl/CRAX:$logical_root/e08-fsrl/env/lib/python3.11/site-packages:$arm_env/lib/python3.11/site-packages:$shared_torch_env/lib/python3.11/site-packages" \
    --setenv LD_LIBRARY_PATH "$logical_root/e08-fsrl/env/lib/python3.11/site-packages/nvidia/cudnn/lib" \
    --setenv JAX_PLATFORMS cuda --setenv XLA_PYTHON_CLIENT_PREALLOCATE false \
    --setenv TF_FORCE_GPU_ALLOW_GROWTH true \
    --setenv NUMBA_CACHE_DIR "$logical_root/e08-fsrl/numba-cache" \
    --setenv JAX_COMPILATION_CACHE_DIR "$logical_root/e08-fsrl/jax-cache" \
    --setenv XDG_CACHE_HOME "$logical_root/e08-fsrl/cache" \
    --setenv MPLCONFIGDIR "$logical_root/e08-fsrl/mpl-cache" \
    --setenv TMPDIR "$logical_root/e08-fsrl/tmp" \
    --setenv OMP_NUM_THREADS 1 --setenv OPENBLAS_NUM_THREADS 1 --setenv MKL_NUM_THREADS 1 \
    --chdir "$logical_root/e08-fsrl/formal-source-v1" \
    "$logical_root/e08-fsrl/env/bin/python" "$@"
