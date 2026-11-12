#!/bin/bash
# Every CPU-feasible P1 / P2 experiment, in the order of the finished CPU campaign (4 cores, no GPU).
#   bash scripts/run_p1p2_cpu.sh
# Fixed locations (scripts/paths.py): datasets in dataset/, job lists, logs, checkpoints and records in work/.
# Resumable: every queue skips the runs that already have a record, so the same command continues after an
# interruption. The kernel analysis needs the P0 checkpoints in work/ckpt (bash scripts/p0/run_p0.sh).
# Order: kernel analysis and CPU efficiency (alone, 4 threads) -> PEMS04/08 horizon 12 -> long-term small datasets
# -> plug-in -> sensitivity -> neighbour selection -> hyper-parameter search -> Electricity long-term -> re-run of
# the searched configurations -> CI linear model, all-neighbour runs and the light baselines.
# Traffic, the large plug-in suite and the heavy baselines need a GPU (GPU_RUNS.md).
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
WORK=$(python scripts/paths.py work)
mkdir -p results/p1 results/p2
[ -f results/p2/kernels_pems08.json ] || OMP_NUM_THREADS=4 python scripts/p2/kernels.py > "$WORK/kernels.log" 2>&1 \
  || echo "kernel analysis failed (see work/kernels.log)"
[ -f results/p2/efficiency_cpu.jsonl ] || OMP_NUM_THREADS=4 python scripts/p2/efficiency.py --device cpu --threads 4 \
  > "$WORK/efficiency.log" 2>&1 || echo "efficiency measurement failed (see work/efficiency.log)"

for s in pems12 long_small plugin sensitivity neighbours long_large allnb; do python scripts/exps.py $s; done
python scripts/exps.py ci_long --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather ECL
python scripts/p1/hpsearch.py make
cat "$WORK"/jobs_pems12.txt "$WORK"/jobs_long_small.txt "$WORK"/jobs_plugin.txt "$WORK"/jobs_sensitivity.txt \
    "$WORK"/jobs_neighbours.txt "$WORK"/jobs_hp.txt > "$WORK/jobs_cpu_all.txt"
grep -v "traffic.csv" "$WORK/jobs_long_large.txt" >> "$WORK/jobs_cpu_all.txt"     # Electricity only; Traffic on a GPU
bash scripts/run_queue.sh cpu_all 4 || echo "some jobs of the main queue failed (see work/logs)"

python scripts/p1/hpsearch.py select > "$WORK/hp_select.md"
python scripts/p1/hpsearch.py tuned
bash scripts/run_queue.sh tuned 4 || echo "some tuned jobs failed"

python scripts/exps.py baselines --models DLinear --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather ECL
mv "$WORK/jobs_baselines.txt" "$WORK/jobs_baselines_a.txt"
python scripts/exps.py baselines --models iTransformer TimeMixer LagTS --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange
mv "$WORK/jobs_baselines.txt" "$WORK/jobs_baselines_b.txt"
cat "$WORK"/jobs_ci_long.txt "$WORK"/jobs_allnb.txt "$WORK"/jobs_baselines_a.txt "$WORK"/jobs_baselines_b.txt \
    > "$WORK/jobs_cpu_extra.txt"
bash scripts/run_queue.sh cpu_extra 4 || echo "some jobs of the extra queue failed"

python scripts/collect_results.py
