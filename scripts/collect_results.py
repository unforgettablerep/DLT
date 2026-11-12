"""Merge the result records of the work directory into results/ and rebuild the tables.

    python scripts/collect_results.py                # work/ -> results/, then the tables
    python scripts/collect_results.py --to-work      # results/ -> work/: finished runs are then skipped by the queues

work/results_<suite>.jsonl goes to results/p0 (suite p0), results/p2 (sensitivity, neighbours) or results/p1 (every
other suite); work/tuned*.json and work/hp_select*.md go to results/p1. Records are merged, not overwritten: a run
is identified by its job name (--des) and seed, and a record of the work directory replaces an older one of
the same run. Then the three aggregators write results/p0/summary.md, results/p1/summary.md and
results/p2/summary.md.
"""
import glob, json, os, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

RES = paths.REPO / 'results'
TARGET = {'p0': 'p0', 'sensitivity': 'p2', 'neighbours': 'p2'}
SKIP = {'quickstart', 'smoke'}       # local test runs (scripts/train_dltdyn.sh), not experiment records


def read(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:                       # a record cut off by an interrupted run
                pass
    return rows


TO_WORK = '--to-work' in sys.argv[1:]
os.makedirs(paths.WORK, exist_ok=True)
sources = sorted(glob.glob(str(RES / 'p[012]' / 'results_*.jsonl'))) if TO_WORK else \
    sorted(glob.glob(str(paths.WORK / 'results_*.jsonl')))
for src in sources:
    suite = os.path.basename(src)[len('results_'):-len('.jsonl')]
    if suite in SKIP:
        continue
    dst = paths.WORK / os.path.basename(src) if TO_WORK else RES / TARGET.get(suite, 'p1') / os.path.basename(src)
    # P0 jobs share one --des; they differ by setting (model_id), kernel variant and gate
    key = lambda r: (r.get('model_id'), r.get('variant'), r.get('gate'), int(r.get('seed', 2021))) if suite == 'p0' \
        else (r.get('des'), int(r.get('seed', 2021)))
    merged, old, new = {}, read(dst), read(src)
    os.makedirs(dst.parent, exist_ok=True)
    if TO_WORK:                                                # append only: a running queue may be writing this file
        have = {key(r) for r in old}
        missing = [r for r in new if key(r) not in have]
        with open(dst, 'a') as f:
            f.writelines(json.dumps(r) + '\n' for r in missing)
        print(f'{os.path.relpath(dst, paths.REPO)}: {len(old) + len(missing)} records ({len(missing)} added from results/)')
        continue
    for r in old + new:                                        # the record of the work directory wins
        merged[key(r)] = r
    added = len(merged) - len({key(r) for r in old})
    with open(dst, 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in merged.values())
    print(f'{os.path.relpath(dst, paths.REPO)}: {len(merged)} records ({added} new)')
if TO_WORK:
    for src in sorted(glob.glob(str(RES / 'p1' / 'tuned*.json'))):
        if not (paths.WORK / os.path.basename(src)).exists():
            shutil.copy(src, paths.WORK / os.path.basename(src))
    sys.exit(0)
for src in sorted(glob.glob(str(paths.WORK / 'tuned*.json')) + glob.glob(str(paths.WORK / 'hp_select*.md'))):
    shutil.copy(src, RES / 'p1' / os.path.basename(src))
    print(f'results/p1/{os.path.basename(src)}: copied')

for script in ('p0/aggregate.py', 'p1/aggregate.py', 'p2/aggregate.py'):
    subprocess.run([sys.executable, str(paths.REPO / 'scripts' / script)], check=True, stdout=subprocess.DEVNULL)
print('tables: results/p0/summary.md, results/p1/summary.md, results/p2/summary.md')
