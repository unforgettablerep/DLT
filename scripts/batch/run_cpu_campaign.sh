#!/bin/bash
# Every CPU-feasible experiment of the study, with the protocol of the official loaders (the only one), stage by stage
# (4 cores, no GPU):
#   bash scripts/batch/run_cpu_campaign.sh [first stage, default 1]
# Resumable: each queue runs only the jobs without a CPU row in their store (output/<category>/<suite>.tsv), so the
# same command continues after an interruption.
#   1  stand-alone DLT-dyn and the same model without the branch (CI): ETT, Exchange, Weather, four horizons, 3 seeds
#   2  the guards of the stand-alone model (suite guard)
#   3  P0 kernel and gate study without Electricity, then the learned kernels on PEMS08
#   4  sensitivity (L, R, k) and neighbour selection
#   5  20-trial validation search on ETT, Exchange, Weather, then the searched configurations, 3 seeds
#   6  all variables as neighbours (PEMS04/08 at horizon 12, Weather) and the lead-lag diagnostic
# Electricity, Traffic, Solar, PEMS03/04/07, the baselines and the plug-in studies need a GPU (make_gpu_jobs.sh).
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=$("$PYTHON_BIN" -m src.utils.paths temp)/jobs
LOG=$("$PYTHON_BIN" -m src.utils.paths log)
SMALL="ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather"
FIRST=${1:-1}
run() { bash scripts/utils/run_queue.sh "$1" 4 || echo "some jobs of $1 failed (see log/)"; }

if [ "$FIRST" -le 1 ]; then
  "$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py long_small
  "$PYTHON_BIN" scripts/ablation/make_jobs.py ci_long --datasets $SMALL
  cat "$JOBS"/jobs_long_small.txt "$JOBS"/jobs_ci_long.txt > "$JOBS/jobs_cpu_s1.txt"; run cpu_s1
fi
if [ "$FIRST" -le 2 ]; then
  "$PYTHON_BIN" scripts/ablation/make_jobs.py guard; run guard
  "$PYTHON_BIN" scripts/ablation/guard_table.py > /dev/null
fi
if [ "$FIRST" -le 3 ]; then
  "$PYTHON_BIN" scripts/ablation/make_jobs.py p0 --datasets ETTh1 Weather PEMS08 --list p0_cpu; run p0_cpu
  "$PYTHON_BIN" scripts/ablation/aggregate.py > /dev/null
  OMP_NUM_THREADS=4 "$PYTHON_BIN" scripts/mechanism/kernels.py > "$LOG/kernels.log" 2>&1 || echo "kernel analysis failed (see log/kernels.log)"
fi
if [ "$FIRST" -le 4 ]; then
  for s in sensitivity neighbours; do "$PYTHON_BIN" scripts/hyperparam/make_jobs.py $s; done
  cat "$JOBS"/jobs_sensitivity.txt "$JOBS"/jobs_neighbours.txt > "$JOBS/jobs_cpu_s4.txt"; run cpu_s4
fi
if [ "$FIRST" -le 5 ]; then
  "$PYTHON_BIN" scripts/hyperparam/hpsearch.py make --datasets $SMALL; run hp
  "$PYTHON_BIN" scripts/hyperparam/hpsearch.py select > output/hyperparam/hp_select.md
  "$PYTHON_BIN" scripts/hyperparam/hpsearch.py tuned; run tuned
fi
if [ "$FIRST" -le 6 ]; then
  "$PYTHON_BIN" scripts/ablation/make_jobs.py allnb --datasets Weather --list allnb_cpu; run allnb_cpu
  OMP_NUM_THREADS=4 "$PYTHON_BIN" scripts/plugin/leadlag_diag.py > "$LOG/leadlag_diag.log" 2>&1 || echo "diagnostic failed (see log/leadlag_diag.log)"
fi
echo "tables: output/ablation/guard.md, output/ablation/summary.md, output/hyperparam/hp_select.md"
