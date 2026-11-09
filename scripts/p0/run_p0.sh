#!/bin/bash
# Priority-0 grid (195 runs) on the CPU: 4 single-threaded jobs at a time.
#   bash scripts/p0/run_p0.sh <data dir> <work dir>
# <data dir> holds ETTh1.csv, weather.csv, electricity.csv and PEMS08.npz (scripts/download_data.sh).
# Afterwards:  python scripts/p0/aggregate.py <work dir>/results_p0.jsonl results/p0/summary.md
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$(cd "$1" && pwd); WORK=$(mkdir -p "$2" && cd "$2" && pwd)
mkdir -p "$WORK/logs" "$WORK/ckpt"
python "$REPO/scripts/p0/make_jobs.py" "$DATA" "$WORK"
cd "$REPO/tslib"                                   # TSLib looks for models/ relative to the working directory
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P "${JOBS:-4}" -d '\n' -I{} bash -c "{}" < "$WORK/jobs.txt"
