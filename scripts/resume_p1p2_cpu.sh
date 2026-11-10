#!/bin/bash
# Resume the CPU P1/P2 pipeline of scripts/run_p1p2_cpu.sh after an interruption: runs only the jobs without a
# result, then the hyper-parameter selection and the tuned re-runs, then the extra list (if present).
#   bash scripts/resume_p1p2_cpu.sh <data dir> <work dir> [<code checkout to run the jobs in, default: this repo>]
# The job lines already hold every argument; the checkout only provides the code (use the frozen one of the run).
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$(cd "$1" && pwd); WORK=$(cd "$2" && pwd); CODE=$(cd "${3:-$REPO}" && pwd)
cd "$REPO"
run_missing() {                    # $1: jobs file, $2: label
  [ -f "$1" ] || return 0
  python scripts/run_missing.py "$1" > "$1.missing"
  echo "$2: $(wc -l < "$1.missing") of $(wc -l < "$1") jobs to run $(date)" >> "$WORK/runner.log"
  [ -s "$1.missing" ] && { bash "$CODE/scripts/run_queue.sh" "$1.missing" 4 || echo "$2: some jobs failed $(date)" >> "$WORK/runner.log"; }
  return 0
}
echo "resume $(date) code $(git -C "$CODE" rev-parse --short HEAD)" >> "$WORK/runner.log"
if ! grep -q P12DONE "$WORK/runner.log"; then
  run_missing "$WORK/jobs_cpu_all.txt" "main queue"
  python scripts/p1/hpsearch.py select --work "$WORK" > "$WORK/hp_select.md"
  python scripts/p1/hpsearch.py tuned --data "$DATA" --work "$WORK"
  run_missing "$WORK/jobs_tuned.txt" "tuned"
  echo "P12DONE $(date)" >> "$WORK/runner.log"
fi
run_missing "$WORK/jobs_extra.txt" "extra"
echo "EXTRADONE $(date)" >> "$WORK/runner.log"
