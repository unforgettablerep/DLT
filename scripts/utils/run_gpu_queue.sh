#!/bin/bash
# Run the jobs of a list that have no result yet on GPUs, one worker per listed GPU id:
#   bash scripts/utils/run_gpu_queue.sh <list>          one job at a time on GPU 0
#   bash scripts/utils/run_gpu_queue.sh <list> 0 0 0    three jobs at a time on GPU 0
#   bash scripts/utils/run_gpu_queue.sh <list> 0 1      one job at a time on each of GPU 0 and GPU 1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=$("$PYTHON_BIN" -m src.utils.paths jobs "$1") || exit 1
shift
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || GPUS=(0)
"$PYTHON_BIN" scripts/utils/run_missing.py "$JOBS" > "$JOBS.missing"
echo "$(wc -l < "$JOBS.missing") of $(wc -l < "$JOBS") jobs to run, ${#GPUS[@]} worker(s) on GPU ${GPUS[*]} ($(date))"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
i=0
for g in "${GPUS[@]}"; do
    awk -v n=${#GPUS[@]} -v i=$i '(NR - 1) % n == i' "$JOBS.missing" > "$JOBS.w$i"
    (
        export CUDA_VISIBLE_DEVICES=$g
        while IFS= read -r cmd; do
            bash -c "$cmd" < /dev/null || echo "failed on GPU $g: $(echo "$cmd" | grep -o -- '--des [^ ]*') $(echo "$cmd" | grep -o -- '--seed [^ ]*')"
        done < "$JOBS.w$i"
    ) &
    i=$((i + 1))
done
wait
echo "done $(date); still missing: $("$PYTHON_BIN" scripts/utils/run_missing.py "$JOBS" | wc -l)"
