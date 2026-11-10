#!/bin/bash
# Run the jobs of a list that have no result yet (scripts/run_missing.py) on the GPU.
#   bash scripts/gpu/run_gpu_queue.sh <jobs file>            one job at a time on GPU 0 (single-GPU default)
#   bash scripts/gpu/run_gpu_queue.sh <jobs file> 0 0 0      three jobs at a time on GPU 0 (small models)
#   bash scripts/gpu/run_gpu_queue.sh <jobs file> 0 1        one job at a time on each of GPU 0 and GPU 1
# One worker per listed GPU id; worker i takes every n-th missing job (the lists are ordered longest first) and
# its jobs see only that GPU through CUDA_VISIBLE_DEVICES (TSLib keeps a mask set by the caller).
# Safe to re-run after an interruption: finished jobs are skipped.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
JOBS=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
GPUS=("$@"); [ ${#GPUS[@]} -gt 0 ] || GPUS=(0)
python "$REPO/scripts/run_missing.py" "$JOBS" > "$JOBS.missing"
echo "$(wc -l < "$JOBS.missing") of $(wc -l < "$JOBS") jobs to run, ${#GPUS[@]} worker(s) on GPU ${GPUS[*]} ($(date))"
cd "$REPO"                                         # TSLib finds models/ relative to the working directory
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
i=0
for g in "${GPUS[@]}"; do
  awk -v n=${#GPUS[@]} -v i=$i '(NR - 1) % n == i' "$JOBS.missing" > "$JOBS.w$i"
  ( export CUDA_VISIBLE_DEVICES=$g
    while IFS= read -r cmd; do
      bash -c "$cmd" < /dev/null || echo "failed on GPU $g: $(echo "$cmd" | grep -o -- '--des [^ ]*') $(echo "$cmd" | grep -o -- '--fix_seed [^ ]*')"
    done < "$JOBS.w$i" ) &
  i=$((i + 1))
done
wait
echo "done $(date); still missing: $(python "$REPO/scripts/run_missing.py" "$JOBS" | wc -l)"
