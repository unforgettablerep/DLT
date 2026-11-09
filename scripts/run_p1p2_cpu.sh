#!/bin/bash
# Every CPU-feasible P1 / P2 experiment, in priority order (this machine: 4 cores, no GPU).
#   DLT_EXTRA="<extra --dlt_* args of the proposed model>" bash scripts/run_p1p2_cpu.sh <data dir> <work dir> <P0 checkpoint dir>
# Order: kernel analysis and CPU efficiency (alone, 4 threads) -> PEMS04/08 horizon 12 -> long-term small datasets
# -> plug-in -> sensitivity -> neighbour selection -> hyper-parameter search -> Electricity long-term
# -> re-run of the searched configurations. Traffic, the large plug-in suite and the re-run baselines need a GPU
# (scripts/exps.py ... --device gpu).
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "$1" && pwd); WORK=$(mkdir -p "$2" && cd "$2" && pwd); CKPT=$3
cd "$REPO"
mkdir -p results/p1 results/p2
OMP_NUM_THREADS=4 python scripts/p2/kernels.py --ckpt "$CKPT" --data "$DATA" --out results/p2 > "$WORK/kernels.log" 2>&1 || echo "kernels failed (see $WORK/kernels.log)"
OMP_NUM_THREADS=4 python scripts/p2/efficiency.py --device cpu --threads 4 --out results/p2/efficiency_cpu.jsonl > "$WORK/efficiency.log" 2>&1 || echo "efficiency failed"
for s in pems12 long_small plugin sensitivity neighbours long_large; do
  python scripts/exps.py $s --data "$DATA" --work "$WORK" --dlt "$DLT_EXTRA"
done
python scripts/p1/hpsearch.py make --data "$DATA" --work "$WORK"
cat "$WORK"/jobs_pems12.txt "$WORK"/jobs_long_small.txt "$WORK"/jobs_plugin.txt "$WORK"/jobs_sensitivity.txt \
    "$WORK"/jobs_neighbours.txt "$WORK"/jobs_hp.txt > "$WORK/jobs_cpu_all.txt"
grep -v "traffic.csv" "$WORK/jobs_long_large.txt" >> "$WORK/jobs_cpu_all.txt"     # Electricity only; Traffic on a GPU
echo "queue start $(date)" >> "$WORK/runner.log"
bash scripts/run_queue.sh "$WORK/jobs_cpu_all.txt" 4 || echo "some jobs failed $(date) (see $WORK/logs)" >> "$WORK/runner.log"
python scripts/p1/hpsearch.py select --work "$WORK" > "$WORK/hp_select.md"
python scripts/p1/hpsearch.py tuned --data "$DATA" --work "$WORK"
echo "tuned start $(date)" >> "$WORK/runner.log"
bash scripts/run_queue.sh "$WORK/jobs_tuned.txt" 4 || echo "some tuned jobs failed $(date)" >> "$WORK/runner.log"
echo "P12DONE $(date)" >> "$WORK/runner.log"
