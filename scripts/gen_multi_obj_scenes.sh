#!/usr/bin/env bash
# Continuously sample random groups of 1-3 objects and generate scenes until
# TARGET_SCENES total successful scenes have been produced.
# Usage: ./gen_multi_obj_scenes.sh [NUM_CPUS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_OBJ_FILE="$PROJECT_ROOT/asset/mj-objects/obj_unsymmetric_train.txt"
NUM_CPUS="${1:-50}"
TARGET_SCENES=5000

# Check if object file exists
if [[ ! -f "$TRAIN_OBJ_FILE" ]]; then
    echo "Error: $TRAIN_OBJ_FILE not found"
    exit 1
fi

# Create a temporary directory for job coordination
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

ALL_OBJECTS_FILE="$TEMP_DIR/all_objects.txt"
SCENE_COUNT_FILE="$TEMP_DIR/scene_count.txt"
LOCK_FILE="$TEMP_DIR/queue.lock"

# Populate all-objects list from the object file (strip \r for CRLF files)
while IFS= read -r object_id || [[ -n "$object_id" ]]; do
    object_id="${object_id//$'\r'/}"
    [[ -z "$object_id" ]] && continue
    [[ "$object_id" =~ ^[[:space:]]*# ]] && continue
    echo "$object_id" >> "$ALL_OBJECTS_FILE"
done < "$TRAIN_OBJ_FILE"

echo "0" > "$SCENE_COUNT_FILE"

# Sample a random element from ALL_OBJECTS_FILE excluding given IDs
# Usage: sample_excluding "id1,id2,id3"
sample_excluding() {
    local exclude_csv="$1"
    local candidate
    while true; do
        candidate=$(shuf -n 1 "$ALL_OBJECTS_FILE")
        local found=0
        local IFS_SAVE="$IFS"
        IFS=','
        for excl in $exclude_csv; do
            if [[ "$candidate" == "$excl" ]]; then
                found=1
                break
            fi
        done
        IFS="$IFS_SAVE"
        if [[ $found -eq 0 ]]; then
            echo "$candidate"
            return
        fi
    done
}

worker_process() {
    local worker_id=$1

    while true; do
        # Check if target has been reached
        exec 200>"$LOCK_FILE"
        flock 200
        current_count=$(cat "$SCENE_COUNT_FILE")
        if [[ $current_count -ge $TARGET_SCENES ]]; then
            flock -u 200
            break
        fi
        flock -u 200
        exec 200>&-

        # ------------------------------------------------------------------ #
        # Sample a random group of 1-4 objects
        # ------------------------------------------------------------------ #
        num_objects=$(( (RANDOM % 3) + 1 ))  # 1, 2, or 3

        first=$(shuf -n 1 "$ALL_OBJECTS_FILE")
        ids_csv="$first"
        for (( e=1; e<num_objects; e++ )); do
            extra=$(sample_excluding "$ids_csv")
            ids_csv="${ids_csv},${extra}"
        done

        # Build Hydra list syntax: ["id1","id2",...]
        hydra_ids="["
        IFS=',' read -ra id_arr <<< "$ids_csv"
        for idx in "${!id_arr[@]}"; do
            [[ $idx -gt 0 ]] && hydra_ids+=","
            hydra_ids+="\"${id_arr[$idx]}\""
        done
        hydra_ids+="]"
        echo "[Worker $worker_id] Object group: $hydra_ids"

        # ------------------------------------------------------------------ #
        # Generate 15 scenes with this group, stop early if target reached
        # ------------------------------------------------------------------ #
        i=1
        retries=0
        while [[ $i -le 10 ]]; do
            # Check global target before each scene
            exec 200>"$LOCK_FILE"
            flock 200
            current_count=$(cat "$SCENE_COUNT_FILE")
            if [[ $current_count -ge $TARGET_SCENES ]]; then
                flock -u 200
                return
            fi
            flock -u 200
            exec 200>&-

            echo "[Worker $worker_id] Scene $i/10 for $hydra_ids (total: $current_count/$TARGET_SCENES)"
            set +e
            output=$(cd "$PROJECT_ROOT" && python -m mgs.cli.gen_scene "object.ids=$hydra_ids" 2>&1)
            exit_code=$?
            set -e

            echo "[Worker $worker_id] Exit code: $exit_code"
            echo "$output" | head -5

            if [[ $exit_code -ne 0 ]] || echo "$output" | grep -qiE "(exception|error|Not enough collision free grasps)"; then
                retries=$((retries + 1))
                echo "[Worker $worker_id] Failed, retrying scene $i/10 for $hydra_ids (retry $retries/10)"
                if [[ $retries -ge 10 ]]; then
                    echo "[Worker $worker_id] Too many retries for $hydra_ids, skipping group"
                    break
                fi
            else
                # Increment shared counter
                exec 200>"$LOCK_FILE"
                flock 200
                new_count=$(( $(cat "$SCENE_COUNT_FILE") + 1 ))
                echo "$new_count" > "$SCENE_COUNT_FILE"
                flock -u 200
                exec 200>&-
                i=$((i + 1))
            fi
        done
    done
}

# Start worker processes
echo "Starting $NUM_CPUS workers to process objects in parallel..."
for ((i = 1; i <= NUM_CPUS; i++)); do
    worker_process $i &
done

# Wait for all workers to finish
wait

echo "All scenes generated successfully!"
