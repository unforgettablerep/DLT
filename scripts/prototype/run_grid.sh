#!/usr/bin/env bash
# Reproduce the 69 runs of the feasibility study (CPU, P parallel workers x 1 thread).
#   bash scripts/prototype/run_grid.sh [out.jsonl] [workers]   # defaults: output/prototype/runs_reproduced.jsonl, 4 workers
# Then: python scripts/prototype/agg_md.py output/prototype/runs_reproduced.jsonl
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:-output/prototype/runs_reproduced.jsonl}
P=${2:-4}
C="python scripts/prototype/proto.py --threads 1 --epochs 6 --patience 2 --out $OUT"   # data: the data/ directory
ALL="ci scalar shift tap3 dlt full scalardyn dltdyn"

jobs() {
  # heavy runs first so they overlap with the cheap ones
  # Electricity (H=96), seed 0; the free kernel is skipped (too slow on CPU)        -> 7 runs
  for v in ci scalar shift tap3 dlt scalardyn dltdyn; do
    echo "$C --data electricity --variant $v --seed 0"
  done
  # PEMS08 (H=96): all variants with seed 0, six variants with seed 1             -> 14 runs
  for v in $ALL; do echo "$C --data PEMS08 --pred_len 96 --variant $v --seed 0"; done
  for v in ci scalar tap3 dlt scalardyn dltdyn; do
    echo "$C --data PEMS08 --pred_len 96 --variant $v --seed 1"
  done
  # ETTh1 (H=96), Weather (H=96), PEMS08 (H=24): all variants, seeds 0 and 1      -> 48 runs
  for s in 0 1; do for v in $ALL; do
    echo "$C --data ETTh1 --variant $v --seed $s"
    echo "$C --data weather --variant $v --seed $s"
    echo "$C --data PEMS08 --pred_len 24 --variant $v --seed $s"
  done; done
}

jobs | xargs -P "$P" -I{} sh -c "{} > /dev/null"
echo "done -> $OUT"
