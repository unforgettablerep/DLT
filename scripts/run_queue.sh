#!/bin/bash
# Run a job list written by scripts/exps.py:  bash scripts/run_queue.sh <jobs file> [parallel jobs, default 4]
# Jobs run from the repository root (TSLib finds models/ relative to the working directory), 1 CPU thread each.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOBS=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
cd "$REPO"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
xargs -P "${2:-4}" -d '\n' -I{} bash -c "{}" < "$JOBS"
