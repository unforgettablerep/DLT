#!/bin/bash
# Priority-0 grid (195 runs) on the CPU: 4 single-threaded jobs at a time.
#   bash scripts/p0/run_p0.sh
# Fixed locations (scripts/paths.py): ETTh1.csv, weather.csv, electricity.csv and PEMS08.npz in dataset/; job list,
# logs, checkpoints and records in work/. Afterwards:  python scripts/collect_results.py
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(python "$REPO/scripts/paths.py" work)
python "$REPO/scripts/p0/make_jobs.py"
cd "$REPO"                                         # TSLib looks for models/ relative to the working directory
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P 4 -d '\n' -I{} bash -c "{}" < "$WORK/jobs_p0.txt"
