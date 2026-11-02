#!/bin/bash
# Kernel ablation of DLT (all 8 variants) on PEMS08 and Weather.
# Run from tslib/:  bash scripts/long_term_forecast/DLT_script/DLT_ablation.sh
# EXTRA_ARGS is passed through (e.g. "--no_use_gpu --num_workers 0").
for v in ci scalar shift tap3 dlt full scalardyn dltdyn; do
  VARIANT=$v bash scripts/long_term_forecast/DLT_script/DLT_PEMS08.sh
  VARIANT=$v bash scripts/long_term_forecast/DLT_script/DLT_Weather.sh
done
