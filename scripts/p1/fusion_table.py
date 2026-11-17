"""How the DLT branch enters the backbone: the arms of the plugin_fusion suite against the backbone alone.

    python scripts/p1/fusion_table.py [--files F ...] [--metric mse|mae] [--out FILE]

Reads results_plugin_fusion.jsonl from results/p1 and the work directory (or --files). Arms (scripts/exps.py,
FUSION_ARMS; models/DLTPlug.py):
  dlt             output fusion (Eq. 2: backbone + sigma * W_c z on the look-back window), joint training
  dlt_out-dec     output fusion, decoupled training, correction kept per variable on validation data
  dlt_refine      refine fusion (distributed-lag correction over the observed past and the backbone forecast), joint
  dlt_refine-dec  refine fusion, decoupled training, per-variable selection
  dlt_input       neighbour signal added to the backbone input, joint
Cells: change of the test metric against the backbone alone, (+ DLT) / alone - 1, averaged over the seeds; bold: the
arm helps. For the decoupled arms, brackets give the change without the per-variable selection (all variables
corrected) and the change of the backbone part of the same run (trained on its own loss, as the backbone alone).
Last rows: per arm, the mean change, the settings it helps, and the worst setting.
"""
import argparse, collections, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
import exps  # noqa: E402

ARMS = ['dlt', 'dlt_out-dec', 'dlt_refine', 'dlt_refine-dec', 'dlt_input']
LABEL = {'dlt': 'output (Eq. 2)', 'dlt_out-dec': 'output, decoupled + select', 'dlt_refine': 'refine',
         'dlt_refine-dec': 'refine, decoupled + select', 'dlt_input': 'input'}


def load(files):
    runs = {}
    for f in files or [d / 'results_plugin_fusion.jsonl' for d in (paths.REPO / 'results' / 'p1', paths.WORK)]:
        f = Path(f)
        if f.exists():
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('model') == 'DLTPlug':
                    runs[(r['des'], r.get('seed'))] = r
    ds_of = {v[1]: k for k, v in exps.DATASETS.items()}
    cells = collections.defaultdict(list)                   # (ds, H, base, arm) -> records
    for r in runs.values():
        ds, H, b = ds_of[r['data_path']], r['pred_len'], r['plug_base']
        stem = f'{ds}_{H}_{b}_'
        if r['des'].startswith(stem):
            cells[(ds, H, b, r['des'][len(stem):])].append(r)
    return cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+')
    p.add_argument('--metric', default='mse', choices=['mse', 'mae'])
    p.add_argument('--out')
    a = p.parse_args()
    k = a.metric
    cells = load(a.files)
    mean = lambda rs, key: statistics.mean(r[key] for r in rs if r.get(key) is not None) \
        if any(r.get(key) is not None for r in rs) else None
    rows = sorted({key[:3] for key in cells if key[3] == 'alone'})
    fmt = lambda d: f'{100 * d:+.1f} %'
    out = ['# How the DLT branch enters the backbone (plugin_fusion)', '',
           f'Change of the test {k.upper()} against the backbone alone; brackets for the decoupled arms: without the '
           f'per-variable selection / the backbone part of the same run.', '',
           '| dataset | H | backbone | alone | ' + ' | '.join(LABEL[x] for x in ARMS) + ' |',
           '|---|---:|---|---:|' + '---:|' * len(ARMS)]
    per_arm = collections.defaultdict(list)
    for ds, H, b in rows:
        alone = mean(cells[(ds, H, b, 'alone')], k)
        if alone is None or alone != alone:
            continue
        line = f'| {ds} | {H} | {b} | {alone:.4f} |'
        for arm in ARMS:
            rs = cells.get((ds, H, b, arm))
            v = mean(rs, k) if rs else None
            if v is None or v != v:
                line += ' |'
                continue
            d = v / alone - 1
            per_arm[arm].append((d, f'{ds}-{H} {b}'))
            cell = fmt(d)
            if arm.endswith('-dec'):
                full, bb = mean(rs, f'{k}_full'), mean(rs, f'{k}_backbone')
                if full is not None and bb is not None:
                    cell += f' ({fmt(full / alone - 1)} / {fmt(bb / alone - 1)})'
            line += f' **{cell}** |' if d < 0 else f' {cell} |'
        out.append(line)
    out += ['', '| arm | settings | mean change | helps | worst |', '|---|---:|---:|---:|---|']
    for arm in ARMS:
        v = per_arm.get(arm)
        if v:
            worst = max(v)
            out.append(f'| {LABEL[arm]} | {len(v)} | {fmt(statistics.mean(d for d, _ in v))} | '
                       f'{sum(d < 0 for d, _ in v)} of {len(v)} | {fmt(worst[0])} ({worst[1]}) |')
    text = '\n'.join(out) + '\n'
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == '__main__':
    main()
