#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
examples_dir="${1:-}"
metric="${2:-}"
case "$metric" in
  all|imaging_quality|aesthetic_quality|human_identity|human_appearance|object_appearance|object_geometry|environment_visible_memory|environment_depth_fidelity|environment_scene_reconstruction|dynamic_memory|camera_rotation|camera_translation) ;;
  *) echo "Usage: $0 <examples-dir> <all|metric> [eval options]"; exit 2 ;;
esac
shift 2
examples_dir="$(cd "$examples_dir" && pwd)"
resources_dir="${WORLDMEMBENCH_DATASETS_DIR:-$repo_dir}"
weights_dir="${WORLDMEMBENCH_WEIGHTS_DIR:-$resources_dir/weights}"
if [[ "$metric" == all ]]; then
  for metric in imaging_quality aesthetic_quality human_identity human_appearance \
      object_appearance object_geometry environment_visible_memory \
      environment_depth_fidelity environment_scene_reconstruction dynamic_memory \
      camera_rotation camera_translation; do
    echo "Evaluating $metric"
    bash "$repo_dir/scripts/eval_demo.sh" "$examples_dir" "$metric" "$@"
  done
  exit 0
fi
output="$examples_dir/eval_results/$metric/$(date -u +%Y%m%d-%H%M%S)-$$"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" -m worldmembench eval "$examples_dir/manifests/$metric.jsonl" \
  --metrics "$metric" --weights-dir "$weights_dir" --output "$output" "$@"
