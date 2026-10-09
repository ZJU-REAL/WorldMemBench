#!/usr/bin/env bash
# User-run source download and native-extension installation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIPE_SOURCE="${PROJECT_ROOT}/third_party/vipe"

if [[ "${CONDA_DEFAULT_ENV:-}" != "worldmembench-vipe" ]]; then
    echo "Activate worldmembench-vipe before installing ViPE." >&2
    exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ViPE's native extension requires the CUDA toolkit (nvcc)." >&2
    exit 1
fi
if [[ ! -f "${VIPE_SOURCE}/setup.py" ]]; then
    echo "Initialize source submodules first: git submodule update --init --recursive" >&2
    exit 1
fi
if [[ -n "$(git -C "${VIPE_SOURCE}" status --porcelain)" ]]; then
    echo "ViPE source has local changes; preserve them before rerunning setup." >&2
    exit 1
fi
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install --no-build-isolation -c "${PROJECT_ROOT}/scripts/requirements-vipe.txt" -e "${VIPE_SOURCE}"
python -m pip check
