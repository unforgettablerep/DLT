#!/bin/bash
# Job lists of the experiments that still need the GPU (README.md, "GPU runs"), written to temp/jobs/jobs_<list>.txt.
#   bash scripts/batch/make_gpu_jobs.sh
# Every job writes its row into the tracked store of its suite (output/<category>/<suite>.tsv), and the queue runner
# (bash scripts/utils/run_gpu_queue.sh <list> [gpu ids]) runs only the jobs without a row from the GPU there
# (scripts/utils/run_missing.py: a CPU row does not count for a GPU job). The tables of the GPU study use GPU runs only:
# the plug-in lists run the + DLT-dyn arms alone (--skip_alone), and the tables take the backbone alone from the
# baseline check on the GPU (plugin_table.py / fusion_table.py --alone baselines --device gpu). Every run uses the
# protocol of the official loaders (src/configs/dlt_profiles.py, protocol()), the only one of the study.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=$("$PYTHON_BIN" -m src.utils.paths temp)/jobs
G="--device gpu"
TWO_STAGE="--dlt_train decoupled --dlt_select variable"     # the plug-in recipe of the paper (Algorithm 1)

# 1. the baselines alone, reproduced once with their official settings (seed 2021; Weather and PEMS04 ran first:
#    baselines_gpu.tsv). No re-tuning: a reproduction more than 10 % behind the published result is marked in the tables.
"$PYTHON_BIN" scripts/benchmark/make_jobs.py baselines $G --seeds 2021 \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange ECL Traffic Solar PEMS03 PEMS07 PEMS08
# 2. stand-alone DLT-dyn and the same model without the branch (CI) on Electricity and Traffic, Solar and PEMS, and the
#    P0 kernel study on Electricity (the CPU runs ETT, Exchange, Weather and PEMS08: run_cpu_campaign.sh)
"$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py long_extra $G
"$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py long_large $G
"$PYTHON_BIN" scripts/ablation/make_jobs.py ci_long $G --datasets ECL Traffic --list ci_long_large
"$PYTHON_BIN" scripts/ablation/make_jobs.py p0 $G --datasets ECL --list p0_ecl
# 3. how the branch enters the backbone: five arms on four settings, the six backbones, + DLT-dyn arms only
"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_fusion $G --seeds 2021 --skip_alone \
    --models S_Mamba iTransformer PatchTST RLinear DLinear TiDE
# 4. the plug-in benchmark: the six baselines + DLT-dyn, two-stage training and per-variable selection, 13 datasets
"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_all $G --seeds 2021 --skip_alone --dlt "$TWO_STAGE" --list plugin_main
# 5. (optional) 20-trial validation search on Electricity, Traffic (H 96) and PEMS04/08 (H 12)
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py make --datasets ECL Traffic $G
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py make --datasets PEMS04 PEMS08 --H 12 --tag _pems $G
# 6. (optional) all other variables as neighbours on Electricity and Traffic
"$PYTHON_BIN" scripts/ablation/make_jobs.py allnb --datasets ECL Traffic $G --list allnb_large

echo
for t in baselines long_extra long_large ci_long_large p0_ecl plugin_fusion plugin_main hp hp_pems allnb_large; do
  echo "$t: $("$PYTHON_BIN" scripts/utils/run_missing.py "$t" | wc -l) of $(wc -l < "$JOBS/jobs_$t.txt") jobs without a GPU result"
done
echo
echo "Run a list:   bash scripts/utils/run_gpu_queue.sh <list> [gpu ids]     (0 0 0: three jobs at a time on GPU 0)"
echo "Tables:       python scripts/benchmark/baseline_check.py"
echo "              python scripts/plugin/fusion_table.py --alone baselines --device gpu"
echo "              python scripts/plugin/plugin_table.py --alone baselines --device gpu --arm dlt_train-decoupled-select-variable"
