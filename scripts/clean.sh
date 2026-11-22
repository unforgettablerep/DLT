#!/usr/bin/env bash
# Remove caches and run artefacts under temp/ (checkpoints, raw predictions); output/ (the tracked results) is kept.
#   bash scripts/clean.sh            everything: Python caches, temp/checkpoints, temp/results, temp/test_results
#   bash scripts/clean.sh ETTh1      the checkpoints and raw predictions of one dataset only
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-}"

if [ -z "$DATASET" ]; then
    find . -type d -name "__pycache__" -not -path "./data/*" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -not -path "./data/*" -delete 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
    rm -rf temp/results/* temp/test_results/* temp/checkpoints/*
else
    rm -rf "temp/checkpoints/${DATASET}/"
    for dir in temp/results temp/test_results; do
        if [ -d "$dir" ]; then
            find "$dir" -maxdepth 1 -name "*${DATASET}_*" -exec rm -rf {} +   # <model_id>_<model>_..., model_id holds the dataset
        fi
    done
fi
