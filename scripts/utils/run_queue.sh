#!/bin/bash
# Run a job list on the CPU:  bash scripts/run_queue.sh <list> [parallel jobs, default 4]
# <list> is a suite name (work/jobs_<name>.txt, written by scripts/exps.py) or a file. Jobs that already have a
# result are skipped, so the same command resumes an interrupted queue. Jobs run from the repository root
# (TSLib finds models/ relative to the working directory), 1 CPU thread each.
REPO=$(cd "$(dirname "$0")/.." && pwd)
JOBS=$(python "$REPO/scripts/paths.py" jobs "$1") || exit 1
python "$REPO/scripts/run_missing.py" "$JOBS" > "$JOBS.missing"
echo "$(wc -l < "$JOBS.missing") of $(wc -l < "$JOBS") jobs to run, ${2:-4} at a time ($(date))"
cd "$REPO"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
xargs -P "${2:-4}" -d '\n' -I{} bash -c "{}" < "$JOBS.missing"
status=$?
echo "done $(date); still missing: $(python "$REPO/scripts/run_missing.py" "$JOBS" | wc -l)"
exit $status
