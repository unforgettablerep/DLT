"""Print the job lines of a job list (scripts/exps.py, hpsearch.py) that have no result yet.

    python scripts/run_missing.py <jobs file> [<jobs file> ...] > missing.txt

A job counts as done when its --result_jsonl file holds a record with the job's --des and --fix_seed. This
makes every queue resumable after an interruption (killed jobs, a restarted machine): run only the
missing lines with scripts/run_queue.sh.
"""
import json, os, re, sys

done = {}


def finished(path):
    if path not in done:
        done[path] = set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:              # a record cut off by the interruption
                    continue
                done[path].add((r.get('des'), int(r.get('seed', 2021))))
    return done[path]


for jobs in sys.argv[1:]:
    for line in open(jobs):
        if not line.strip():
            continue
        arg = lambda k, d=None: (re.search(rf'--{k} (\S+)', line) or [None, d])[1]
        key = (arg('des'), int(arg('fix_seed', 2021)))
        if key not in finished(arg('result_jsonl')):
            print(line.rstrip('\n'))
