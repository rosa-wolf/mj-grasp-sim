#!/usr/bin/env bash
# Enhanced parallel rendering: splits all scene dirs into batches and runs as many jobs as cores allow
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"

total_cores=80
cores_per_job="${1:-5}"
min_core_id="${2:-0}"
scene_base_dir="/home/ws/data/outputs/context_clutter_v3/PandaGripper"

if [ "$min_core_id" -ge "$total_cores" ]; then
  echo "Error: min_core_id ($min_core_id) must be less than total_cores ($total_cores)" >&2
  exit 1
fi
if [ "$min_core_id" -lt 0 ]; then
  echo "Error: min_core_id ($min_core_id) must be non-negative" >&2
  exit 1
fi

available_cores=$((total_cores - min_core_id))
max_parallel_jobs=$((available_cores / cores_per_job))
if [ "$max_parallel_jobs" -lt 1 ]; then
  max_parallel_jobs=1
  cores_per_job=$total_cores
fi

echo "Total cores: $total_cores"
echo "Starting from core: $min_core_id"
echo "Available cores: $available_cores"
echo "Cores per job: $cores_per_job"
echo "Max parallel jobs: $max_parallel_jobs"

# Find all scene directories (ignore input_id)
mapfile -t scene_dirs < <(find "$scene_base_dir" -mindepth 1 -maxdepth 1 -type d | sort)
num_scenes=${#scene_dirs[@]}
if [ "$num_scenes" -eq 0 ]; then
  echo "No scene directories found in $scene_base_dir"
  exit 1
fi

# Split into batches
batch_size=$(( (num_scenes + max_parallel_jobs - 1) / max_parallel_jobs ))
batch_start=0
batch_id=0
batch_pids=()
while [ $batch_start -lt $num_scenes ]; do
  batch_end=$((batch_start + batch_size))
  if [ $batch_end -gt $num_scenes ]; then batch_end=$num_scenes; fi
  batch=( "${scene_dirs[@]:batch_start:batch_end-batch_start}" )

  start_core=$((min_core_id + batch_id * cores_per_job))
  end_core=$((start_core + cores_per_job - 1))
  if [ "$end_core" -ge "$total_cores" ]; then end_core=$((total_cores - 1)); fi
  core_range="$start_core-$end_core"
  if [ "$start_core" -eq "$end_core" ]; then core_range="$start_core"; fi

  echo "Batch $((batch_id+1)): ${#batch[@]} scenes on cores $core_range"

  env_vars=(
    "OMP_NUM_THREADS=$cores_per_job"
    "OPENBLAS_NUM_THREADS=$cores_per_job"
    "MKL_NUM_THREADS=$cores_per_job"
    "NUMEXPR_NUM_THREADS=$cores_per_job"
  )

  # Pass scene directories as a proper list for Hydra
  scene_dirs_arg="["
  for dir in "${batch[@]}"; do
    esc_dir=$(printf '%q' "$dir")
    esc_dir=${esc_dir//\'/}
    scene_dirs_arg+="$esc_dir,"
  done
  scene_dirs_arg="${scene_dirs_arg%,}]"
  env "${env_vars[@]}" taskset -c "$core_range" python -m mgs.cli.render_scene_point_label scene_dirs="$scene_dirs_arg" &
  batch_pids+=("$!")

  batch_start=$batch_end
  batch_id=$((batch_id + 1))
  if [ $((batch_id % max_parallel_jobs)) -eq 0 ]; then
    echo "  Waiting for $max_parallel_jobs batches to finish..."
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    batch_pids=()
  fi
done
if [ ${#batch_pids[@]} -gt 0 ]; then
  echo "  Waiting for last ${#batch_pids[@]} batches to finish..."
  for pid in "${batch_pids[@]}"; do wait "$pid"; done
fi

echo "All jobs finished"
