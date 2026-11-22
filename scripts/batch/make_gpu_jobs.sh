#!/bin/bash
# Job lists of the experiments that need a GPU (README.md, "GPU runs"), written to temp/jobs/jobs_<list>.txt.
#   bash scripts/batch/make_gpu_jobs.sh
# Every job writes its row into the tracked store of its suite (output/<category>/<suite>.tsv), and the queue runner
# (bash scripts/utils/run_gpu_queue.sh <list> [gpu ids]) runs only the jobs without a row there. The records of a
# pre-refactor work/ directory are taken over first (scripts/utils/migrate_work.py), so finished runs are not repeated.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=$("$PYTHON_BIN" -m src.utils.paths temp)/jobs
[ -d work ] && "$PYTHON_BIN" scripts/utils/migrate_work.py work --apply
G="--device gpu"

# 1. the baseline check: the six baselines alone (official settings and protocol), 13 datasets, one seed (312 runs)
"$PYTHON_BIN" scripts/benchmark/make_jobs.py baselines $G --seeds 2021
# 2. how the branch enters the backbone: five arms on four settings, the six backbones, one seed
"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_fusion $G --seeds 2021 --models S_Mamba iTransformer PatchTST RLinear DLinear TiDE
# 3. the plug-in study: every backbone alone and + DLT-dyn on the 13 datasets, one seed (624 runs)
"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_all $G --seeds 2021
# 4. stand-alone DLT-dyn: Electricity and Traffic; Solar and PEMS03/04/07/08 (seed 2021 ran on the CPU)
"$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py long_large $G
"$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py long_extra $G
# 5. the linear model without the cross branch on Traffic (the other long-term sets ran on the CPU)
"$PYTHON_BIN" scripts/ablation/make_jobs.py ci_long $G --datasets Traffic --list ci_long_traffic
# 6. (optional) 20-trial validation search on Electricity, Traffic (H 96) and PEMS04/08 (H 12, TimeMixer protocol)
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py make --datasets ECL Traffic $G
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py make --datasets PEMS04 PEMS08 --H 12 --tag _pems $G
# 7. (optional) all other variables as neighbours on Electricity and Traffic
"$PYTHON_BIN" scripts/ablation/make_jobs.py allnb --datasets ECL Traffic $G --list allnb_large

echo
for t in baselines plugin_fusion plugin_all long_large long_extra ci_long_traffic hp hp_pems allnb_large; do
  echo "$t: $("$PYTHON_BIN" scripts/utils/run_missing.py "$t" | wc -l) of $(wc -l < "$JOBS/jobs_$t.txt") jobs without a result"
done
echo
echo "Run a list:   bash scripts/utils/run_gpu_queue.sh <list> [gpu ids]     (0 0 0: three jobs at a time on GPU 0)"
echo "Efficiency:   python scripts/efficiency/run_efficiency.py --setting smamba --device gpu --tuned"
