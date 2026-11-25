#!/bin/bash
# Job lists of the experiments that still need the GPU (README.md, "GPU runs"), written to temp/jobs/jobs_<list>.txt.
#   bash scripts/batch/make_gpu_jobs.sh
# Every job writes its row into the tracked store of its suite (output/<category>/<suite>.tsv), and the queue runner
# (bash scripts/utils/run_gpu_queue.sh <list> [gpu ids]) runs only the jobs without a row from the GPU there
# (scripts/utils/run_missing.py: a CPU row does not count for a GPU job). The tables of the GPU study use GPU runs only:
# the plug-in lists run the + DLT-dyn arms alone (--skip_alone), and the tables take the backbone alone from the
# baseline check on the GPU (plugin_table.py / fusion_table.py --alone baselines --device gpu). The records of a
# pre-refactor work/ directory are taken over first (scripts/utils/migrate_work.py), so finished runs are not repeated.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=$("$PYTHON_BIN" -m src.utils.paths temp)/jobs
[ -d work ] && "$PYTHON_BIN" scripts/utils/migrate_work.py work --apply
G="--device gpu"
PEMS="PEMS03 PEMS04 PEMS07 PEMS08"
TWO_STAGE="--dlt_train decoupled --dlt_select variable"     # the plug-in recipe of the paper (Algorithm 1)

# 1. the baseline check: the six baselines alone, seed 2021 (Weather and PEMS04 ran first: baselines_gpu.tsv)
"$PYTHON_BIN" scripts/benchmark/make_jobs.py baselines $G --seeds 2021 \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange ECL Traffic Solar PEMS03 PEMS07 PEMS08
# 2. its learning-rate check for iTransformer and PatchTST on PEMS, and a second seed of iTransformer there
"$PYTHON_BIN" scripts/benchmark/make_jobs.py baselines $G --seeds 2021 --models iTransformer PatchTST --datasets $PEMS \
    --backbone_lr 0.0001 0.0005 0.001 --list baselines_lr
"$PYTHON_BIN" scripts/benchmark/make_jobs.py baselines $G --seeds 2022 --models iTransformer --datasets $PEMS --list baselines_itr2022
# 3. stand-alone DLT-dyn on Solar and PEMS (Electricity and Traffic: long_large and tier1_partial)
"$PYTHON_BIN" scripts/multivariate_forecasting/make_jobs.py long_extra $G
# 4. how the branch enters the backbone: five arms on four settings, the six backbones, + DLT-dyn arms only
"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_fusion $G --seeds 2021 --skip_alone \
    --models S_Mamba iTransformer PatchTST RLinear DLinear TiDE
# 5. the plug-in benchmark: the six baselines + DLT-dyn, two-stage training and per-variable selection, 13 datasets
"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_all $G --seeds 2021 --skip_alone --dlt "$TWO_STAGE" --list plugin_main
#    the same at the learning rate chosen by the check of step 2 where it is not the official one (write it after step 2)
"$PYTHON_BIN" scripts/plugin/make_jobs.py plugin_all $G --seeds 2021 --skip_alone --dlt "$TWO_STAGE" --select_lr --list plugin_main_lr
# 6. (optional) 20-trial validation search on Electricity, Traffic (H 96) and PEMS04/08 (H 12, TimeMixer protocol)
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py make --datasets ECL Traffic $G
"$PYTHON_BIN" scripts/hyperparam/hpsearch.py make --datasets PEMS04 PEMS08 --H 12 --tag _pems $G
# 7. (optional) all other variables as neighbours on Electricity and Traffic
"$PYTHON_BIN" scripts/ablation/make_jobs.py allnb --datasets ECL Traffic $G --list allnb_large

echo
for t in baselines baselines_lr baselines_itr2022 long_extra plugin_fusion plugin_main plugin_main_lr hp hp_pems allnb_large; do
  echo "$t: $("$PYTHON_BIN" scripts/utils/run_missing.py "$t" | wc -l) of $(wc -l < "$JOBS/jobs_$t.txt") jobs without a GPU result"
done
echo
echo "Run a list:   bash scripts/utils/run_gpu_queue.sh <list> [gpu ids]     (0 0 0: three jobs at a time on GPU 0)"
echo "Tables:       python scripts/benchmark/baseline_check.py"
echo "              python scripts/plugin/fusion_table.py --alone baselines --device gpu"
echo "              python scripts/plugin/plugin_table.py --alone baselines --device gpu --arm dlt_train-decoupled-select-variable"
