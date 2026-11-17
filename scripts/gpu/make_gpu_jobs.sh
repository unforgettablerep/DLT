#!/bin/bash
# Job lists of every experiment that still needs a GPU (GPU_RUNS.md).
#   bash scripts/gpu/make_gpu_jobs.sh
# Datasets are read from dataset/ and the lists are written to work/ (fixed, scripts/paths.py). The records of the
# finished runs in results/ are merged into work/ first, so run_gpu_queue.sh skips what is already done.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
WORK=$(python scripts/paths.py work)
python scripts/collect_results.py --to-work
G="--device gpu"

# tier 1: completes the main tables (Traffic, plug-in on large data, PatchTST on PEMS08)
python scripts/exps.py long_large $G                     # DLT-dyn, Electricity (done on CPU) + Traffic
python scripts/exps.py ci_long $G                        # CI linear, all long-term datasets (all but Traffic on CPU)
python scripts/exps.py plugin $G                         # PatchTST -+ DLT-dyn on PEMS08-12 (rest on CPU)
python scripts/exps.py plugin_large $G                   # DLinear / PatchTST / iTransformer -+ DLT-dyn: ECL, Traffic, PEMS04-12
cat "$WORK"/jobs_long_large.txt "$WORK"/jobs_ci_long.txt "$WORK"/jobs_plugin.txt "$WORK"/jobs_plugin_large.txt \
    > "$WORK/jobs_gpu_tier1.txt"

# plug-in diagnosis on Traffic-96 (results/gpu/README.md): six runs
python scripts/exps.py plugin_probe --seeds 2021 $G

# tier 2: 20-trial validation search on the large / traffic data (the 6 small datasets were searched on CPU)
python scripts/p1/hpsearch.py make --datasets ECL Traffic $G
python scripts/p1/hpsearch.py make --datasets PEMS04 PEMS08 --H 12 --tag _pems $G
cat "$WORK"/jobs_hp.txt "$WORK"/jobs_hp_pems.txt > "$WORK/jobs_gpu_tier2_search.txt"
# (after the search: hpsearch.py select / tuned, see GPU_RUNS.md, then the list gpu_tier2_tuned)

# tier 3: the baseline check, the six baselines with their official settings and protocol, one seed
python scripts/exps.py baselines --seeds 2021 $G
cp "$WORK/jobs_baselines.txt" "$WORK/jobs_gpu_tier3.txt"

# tier 4 (optional): all other variables as neighbours on Electricity / Traffic
python scripts/exps.py allnb --datasets ECL Traffic $G
cp "$WORK/jobs_allnb.txt" "$WORK/jobs_gpu_tier4.txt"

echo
for t in gpu_tier1 plugin_probe gpu_tier2_search gpu_tier3 gpu_tier4; do
  echo "$t: $(python scripts/run_missing.py $t | wc -l) of $(wc -l < "$WORK/jobs_$t.txt") jobs without a result"
done
