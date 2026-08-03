#!/usr/bin/env bash
# Reproduce the STAR P0/P1 Python 3.11 environment on the confirmed data disk.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SAFE_ROOT="${SAFE_ROOT:-/root/autodl-tmp/SAFE}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
PYTHON_ENV="${SAFE_ROOT}/envs/star-p0-p1"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONSTRAINTS="${REPO_ROOT}/scripts/star/constraints-py311.txt"

mkdir -p \
  "${SAFE_ROOT}/code" \
  "${SAFE_ROOT}/envs" \
  "${SAFE_ROOT}/cache/pip" \
  "${SAFE_ROOT}/cache/conda" \
  "${SAFE_ROOT}/cache/xdg" \
  "${SAFE_ROOT}/cache/huggingface" \
  "${SAFE_ROOT}/cache/torch" \
  "${SAFE_ROOT}/cache/torchinductor" \
  "${SAFE_ROOT}/cache/tmp" \
  "${SAFE_ROOT}/runs/active" \
  "${SAFE_ROOT}/runs/completed" \
  "${SAFE_ROOT}/artifacts" \
  "${SAFE_ROOT}/logs"

available_kib="$(df -Pk "${SAFE_ROOT}" | awk 'NR == 2 {print $4}')"
if (( available_kib < 25 * 1024 * 1024 )); then
  echo "Data disk must have at least 25 GiB free before setup" >&2
  exit 1
fi

export CONDA_PKGS_DIRS="${SAFE_ROOT}/cache/conda"
export PIP_CACHE_DIR="${SAFE_ROOT}/cache/pip"
export TMPDIR="${SAFE_ROOT}/cache/tmp"
export XDG_CACHE_HOME="${SAFE_ROOT}/cache/xdg"
export HF_HOME="${SAFE_ROOT}/cache/huggingface"
export TORCH_HOME="${SAFE_ROOT}/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${SAFE_ROOT}/cache/torchinductor"
export WANDB_DIR="${SAFE_ROOT}/runs/active"

if [[ ! -x "${PYTHON_ENV}/bin/python" ]]; then
  "${CONDA_EXE}" create -y -p "${PYTHON_ENV}" python=3.11 pip
fi

PYTHON="${PYTHON_ENV}/bin/python"
"${PYTHON}" -m pip install --index-url "${TORCH_INDEX_URL}" 'torch==2.8.0'
"${PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" -c "${CONSTRAINTS}" \
  numpy scipy matplotlib contourpy pygame gymnasium gymnasium-robotics mujoco \
  xmltodict pyyaml imageio pytest psutil
"${PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" --no-deps \
  'safety-gymnasium==1.0.0'
"${PYTHON}" "${REPO_ROOT}/scripts/star/patch_safety_gymnasium_py311.py"
"${PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" -c "${CONSTRAINTS}" \
  'tensorboard>=2.8.0' 'wandb>=0.13.0' 'pyyaml>=6.0' 'moviepy>=1.0.0' \
  'typing-extensions>=4.0.0' 'typer[all]>=0.7.0' 'seaborn>=0.12.2' \
  'pandas==2.0.3' 'matplotlib>=3.7.1' 'gdown>=4.6.0' \
  'pytorch-lightning>=2.2.2' 'cvxopt==1.3.2' 'gpytorch==1.11' \
  'joblib==1.3.2' 'qpth==0.0.16' 'scikit-learn==1.3.2' \
  pytest pytest-cov pytest-xdist setuptools
"${PYTHON}" -m pip install --index-url "${PIP_INDEX_URL}" --no-deps -e "${REPO_ROOT}"
"${PYTHON}" "${REPO_ROOT}/scripts/star/patch_safety_gymnasium_py311.py"
"${PYTHON}" -m pip check

"${PYTHON}" - <<'PY'
import torch
import omnisafe
import safety_gymnasium

assert torch.__version__ == '2.8.0+cu128'
assert torch.version.cuda == '12.8'
assert torch.cuda.is_available()
assert safety_gymnasium.__version__ == '1.0.0'
print(omnisafe.__version__, torch.__version__, torch.cuda.get_device_name(0))
PY
