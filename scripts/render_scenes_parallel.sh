#!/usr/bin/env bash
# Run jobs in parallel, allocating multiple cores per Python process
# Usage: ./render_scenes_parallel.sh [CORES_PER_JOB] [MIN_CORE_ID]
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"  # Disable GPU for all jobs

#ids=(0 1 2 3 4 5 6 7 8 9 a b c d e f)
ids=(0 1 2 3 4 5 6 7 8 9 f)
total_cores=$(nproc)

# Cores per job (default: 4, can be overridden by first argument)
cores_per_job="${1:-5}"

# Minimum core ID to start with (default: 0, can be overridden by second argument)
min_core_id="${2:-0}"

# Validate min_core_id
if [ "$min_core_id" -ge "$total_cores" ]; then
  echo "Error: min_core_id ($min_core_id) must be less than total_cores ($total_cores)" >&2
  exit 1
fi

if [ "$min_core_id" -lt 0 ]; then
  echo "Error: min_core_id ($min_core_id) must be non-negative" >&2
  exit 1
fi

# Calculate available cores starting from min_core_id
available_cores=$((total_cores - min_core_id))

# Calculate how many jobs we can run in parallel
max_parallel_jobs=$((available_cores / cores_per_job))

# Ensure we don't run more jobs than we have IDs for
if [ "$max_parallel_jobs" -gt "${#ids[@]}" ]; then
  max_parallel_jobs="${#ids[@]}"
fi

# Ensure we run at least 1 job
if [ "$max_parallel_jobs" -lt 1 ]; then
  max_parallel_jobs=1
  cores_per_job=$total_cores
fi

echo "Total cores: $total_cores"
echo "Starting from core: $min_core_id"
echo "Available cores: $available_cores"
echo "Cores per job: $cores_per_job"
echo "Max parallel jobs: $max_parallel_jobs"

# Process jobs in batches
job_index=0
while [ "$job_index" -lt "${#ids[@]}" ]; do
  # Start a batch of parallel jobs
  batch_pids=()
  
  for ((batch_slot=0; batch_slot<max_parallel_jobs && job_index<${#ids[@]}; batch_slot++)); do
    id=${ids[job_index]}
    
    # Calculate core range for this job (starting from min_core_id)
    start_core=$((min_core_id + batch_slot * cores_per_job))
    end_core=$((start_core + cores_per_job - 1))
    
    # Ensure we don't exceed available cores
    if [ "$end_core" -ge "$total_cores" ]; then
      end_core=$((total_cores - 1))
    fi
    
    core_range="$start_core-$end_core"
    if [ "$start_core" -eq "$end_core" ]; then
      core_range="$start_core"
    fi
    
    echo "Starting input_id=$id on cores $core_range (job $((job_index + 1))/${#ids[@]})"
    
    # Set environment variables for multi-threading
    env_vars=(
      "OMP_NUM_THREADS=$cores_per_job"
      "OPENBLAS_NUM_THREADS=$cores_per_job"
      "MKL_NUM_THREADS=$cores_per_job"
      "NUMEXPR_NUM_THREADS=$cores_per_job"
    )
    
    # Start the job with core affinity and threading environment
    env "${env_vars[@]}" taskset -c "$core_range" python -m mgs.cli.render_scene_point_label input_id="$id" &
    batch_pids+=($!)
    
    job_index=$((job_index + 1))
  done
  
  # Wait for current batch to complete
  echo "Waiting for batch of ${#batch_pids[@]} jobs to complete..."
  for pid in "${batch_pids[@]}"; do
    wait "$pid"
  done
  
  echo "Batch completed. Processed $job_index/${#ids[@]} jobs."
done

echo "All jobs finished"