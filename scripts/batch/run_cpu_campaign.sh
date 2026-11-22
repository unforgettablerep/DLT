#!/bin/bash
# Every CPU-feasible experiment of the paper, in the order of the finished CPU campaign (4 cores, no GPU).
#   bash scripts/batch/run_cpu_campaign.sh
# Resumable: each queue runs only the jobs without a row in their store (output/<category>/<suite>.tsv), so the same
# command continues after an interruption. With the stores of this repository only the fusion study has runs left
# (seed 2022 was stopped for time).
# Order: P0 kernel and gate study -> learned kernels and CPU efficiency (alone, 4 threads) -> PEMS04/08 horizon 12,
# long-term small datasets, sensitivity, neighbour selection, hyper-parameter search, Electricity -> re-run of the
# searched configurations -> CI linear model and all-neighbour runs -> fusion study (RLinear, DLinear) -> tables.
# Traffic, the plug-in study and the baseline check need a GPU (scripts/batch/make_gpu_jobs.sh); the CPU baseline
# runs with the first protocol are kept in output/benchmark/baselines_v1.tsv.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=$("$PYTHON_BIN" -m src.utils.paths temp)/jobs
LOG=$("$PYTHON_BIN" -m src.utils.paths log)
SMALL="ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather"

"$PYTHON_BIN" scripts/ablation/make_jobs.py p0
bash scripts/utils/run_queue.sh p0 4 || echo "some P0 jobs failed (see log/ablation/p0)"
[ -f output/mechanism/kernels_pems08.json ] || OMP_NUM_THREADS=4 "$PYTHON_BIN" scripts/mechanism/kernels.py \
  > "$LOG/kernels.log" 2>&1 || echo "kernel analysis failed (see log/kernels.log)"
[ -f output/efficiency/efficiency_cpu.jsonl ] || OMP_NUM_THREADS=4 "$PYTHON_BIN" scripts/efficiency/run_efficiency.py \
  --device cpu --threads 4 --variants > "$LOG/efficiency.log" 2>&1 || echo "efficiency measurement failed (see log/efficiency.log)"

for s in pems12 long_small; do "$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py $s; done
"$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py long_large --datasets ECL --list long_large_ecl
for s in sensitivity neighbours; do "$PYTHON_BIN" scripts/hyperparam/make_jobs.py $s; done
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py make
cat "$JOBS"/jobs_pems12.txt "$JOBS"/jobs_long_small.txt "$JOBS"/jobs_sensitivity.txt "$JOBS"/jobs_neighbours.txt \
    "$JOBS"/jobs_hp.txt "$JOBS"/jobs_long_large_ecl.txt > "$JOBS/jobs_cpu_all.txt"
bash scripts/utils/run_queue.sh cpu_all 4 || echo "some jobs of the main queue failed (see log/)"

"$PYTHON_BIN" scripts/hyperparam/hpsearch.py select > output/hyperparam/hp_select.md
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py tuned
bash scripts/utils/run_queue.sh tuned 4 || echo "some tuned jobs failed"

"$PYTHON_BIN" scripts/ablation/make_jobs.py ci_long --datasets $SMALL ECL
"$PYTHON_BIN" scripts/ablation/make_jobs.py allnb
cat "$JOBS"/jobs_ci_long.txt "$JOBS"/jobs_allnb.txt > "$JOBS/jobs_cpu_extra.txt"
bash scripts/utils/run_queue.sh cpu_extra 4 || echo "some jobs of the extra queue failed"

"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_fusion --models RLinear DLinear --seeds 2021 2022
bash scripts/utils/run_queue.sh plugin_fusion 4 || echo "some fusion jobs failed"

"$PYTHON_BIN" scripts/ablation/aggregate.py > /dev/null
"$PYTHON_BIN" scripts/statistics/aggregate_p1.py > /dev/null
"$PYTHON_BIN" scripts/statistics/aggregate_p2.py > /dev/null
echo "tables: output/ablation/summary.md, output/statistics/summary_p1.md, output/statistics/summary_p2.md"
