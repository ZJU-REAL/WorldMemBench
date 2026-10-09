#!/usr/bin/env bash
# Prepare the checked-out DA3/SALAD/DINOv2 submodules for offline inference.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
da3_source="${project_root}/third_party/Depth-Anything-3"
dino_source="${project_root}/third_party/dinov2"
salad_source="${da3_source}/da3_streaming/loop_utils/salad"
if [[ ! -f "${da3_source}/da3_streaming/da3_streaming.py" ||
      ! -f "${dino_source}/hubconf.py" ||
      ! -f "${salad_source}/models/backbones/dinov2.py" ]]; then
    echo "Initialize source submodules first: git submodule update --init --recursive" >&2
    exit 1
fi
python "${project_root}/scripts/prepare_da3.py" --patch-salad "${salad_source}"
echo "DA3 source is ready in third_party/. No model weights or Python packages were installed."
