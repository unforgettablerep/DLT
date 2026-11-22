#!/bin/bash
# Run the jobs of a list that have no result yet on the CPU: bash scripts/utils/run_queue.sh <list> [parallel jobs, default 4]
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=$("$PYTHON_BIN" -m src.utils.paths jobs "$1") || exit 1
"$PYTHON_BIN" scripts/utils/run_missing.py "$JOBS" > "$JOBS.missing"
echo "$(wc -l < "$JOBS.missing") of $(wc -l < "$JOBS") jobs to run, ${2:-4} at a time ($(date))"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
xargs -P "${2:-4}" -d '\n' -I{} bash -c "{}" < "$JOBS.missing"
status=$?
echo "done $(date); still missing: $("$PYTHON_BIN" scripts/utils/run_missing.py "$JOBS" | wc -l)"
exit $status
