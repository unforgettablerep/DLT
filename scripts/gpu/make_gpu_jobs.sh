#!/bin/bash
# Job lists of every experiment that still needs a GPU (GPU_RUNS.md). Copy the CPU results first
# (cp results/p1/results_*.jsonl <work>/): jobs that already have a result are skipped by run_gpu_queue.sh.
#   bash scripts/gpu/make_gpu_jobs.sh <data dir> <work dir>
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
DATA=$(cd "$1" && pwd); WORK=$(mkdir -p "$2" && cd "$2" && pwd)
G="--data $DATA --work $WORK --device gpu"

# tier 1: completes the main tables (Traffic, plug-in on large data, PatchTST on PEMS08)
python scripts/exps.py long_large $G                     # DLT-dyn, Electricity (done on CPU) + Traffic
python scripts/exps.py ci_long $G                        # CI linear, all long-term datasets (all but Traffic on CPU)
python scripts/exps.py plugin $G                         # PatchTST -+ DLT-dyn on PEMS08-12 (rest on CPU)
python scripts/exps.py plugin_large $G                   # DLinear / PatchTST / iTransformer -+ DLT-dyn: ECL, Traffic, PEMS04-12
cat "$WORK"/jobs_long_large.txt "$WORK"/jobs_ci_long.txt "$WORK"/jobs_plugin.txt "$WORK"/jobs_plugin_large.txt \
    > "$WORK/jobs_gpu_tier1.txt"

# tier 2: 20-trial validation search on the large / traffic data (the 6 small datasets were searched on CPU)
python scripts/p1/hpsearch.py make --datasets ECL Traffic $G
python scripts/p1/hpsearch.py make --datasets PEMS04 PEMS08 --H 12 --tag _pems $G
cat "$WORK"/jobs_hp.txt "$WORK"/jobs_hp_pems.txt > "$WORK/jobs_gpu_tier2_search.txt"
# (after the search: hpsearch.py select / tuned, see GPU_RUNS.md, then the list jobs_gpu_tier2_tuned.txt)

# tier 3: baselines re-run with the official TSLib hyper-parameters (the CPU re-ran only the light ones)
python scripts/exps.py baselines $G
cp "$WORK/jobs_baselines.txt" "$WORK/jobs_gpu_tier3.txt"

# tier 4 (optional): all other variables as neighbours on Electricity / Traffic
python scripts/exps.py allnb --datasets ECL Traffic $G
cp "$WORK/jobs_allnb.txt" "$WORK/jobs_gpu_tier4.txt"

for f in "$WORK"/jobs_gpu_tier*.txt; do
  echo "$(basename "$f"): $(python scripts/run_missing.py "$f" | wc -l) of $(wc -l < "$f") jobs without a result"
done
