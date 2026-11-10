#!/bin/bash
# Run the jobs of a list that have no result yet (scripts/run_missing.py), one job at a time per GPU.
#   bash scripts/gpu/run_gpu_queue.sh <jobs file> [<gpu id> ...]          (default: GPU 0)
# Each GPU gets every n-th missing job (the lists are ordered longest first); a job sees only its GPU through
# CUDA_VISIBLE_DEVICES (TSLib keeps a mask set by the caller). Safe to re-run after an interruption.
REPO=$(cd "$(dirname "$0")/../.." && pwd)
JOBS=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
GPUS=("$@"); [ ${#GPUS[@]} -gt 0 ] || GPUS=(0)
python "$REPO/scripts/run_missing.py" "$JOBS" > "$JOBS.missing"
echo "$(wc -l < "$JOBS.missing") of $(wc -l < "$JOBS") jobs to run on GPU ${GPUS[*]} ($(date))"
cd "$REPO/tslib"                                   # TSLib finds models/ relative to the working directory
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
i=0
for g in "${GPUS[@]}"; do
  awk -v n=${#GPUS[@]} -v i=$i '(NR - 1) % n == i' "$JOBS.missing" > "$JOBS.gpu$g"
  ( export CUDA_VISIBLE_DEVICES=$g
    while IFS= read -r cmd; do
      bash -c "$cmd" < /dev/null || echo "failed on GPU $g: $(echo "$cmd" | grep -o -- '--des [^ ]*') $(echo "$cmd" | grep -o -- '--fix_seed [^ ]*')"
    done < "$JOBS.gpu$g" ) &
  i=$((i + 1))
done
wait
echo "done $(date); still missing: $(python "$REPO/scripts/run_missing.py" "$JOBS" | wc -l)"
