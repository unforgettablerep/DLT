"""Baseline check: every baseline alone against the S-Mamba paper, baseline by baseline.

    python scripts/p1/baseline_check.py [--files F ...] [--models ...] [--tol 5] [--out FILE]

Reads the records of the baselines suite (scripts/exps.py baselines: each baseline alone with its official settings
and the protocol of the official loaders, scripts/baselines.py) from results/p1/results_baselines.jsonl and the
work directory, or --files (e.g. copied from the GPU machine). Records of the backbone alone in the plug-in suites
(DLTPlug, arm "alone") count as well. Only records of the check's protocol (four time features, dropped last
batches; the records name freq and drop_last since 2026-11-17) unless --any_protocol. Several seeds are averaged.
The records of the earlier eight baselines are in results/p1/results_baselines_v1.jsonl.

Reference: scripts/p1/reported.json, block smamba_paper (Tables 2-4 of the S-Mamba paper; for iTransformer, PatchTST,
RLinear, DLinear and TiDE they equal the iTransformer paper). For every baseline: one row per dataset with
ours / paper (change of MSE) per horizon and the change of MAE averaged over the horizons; a cell is marked
when the MSE is more than --tol % worse than published (default 5 %; ** more than 10 %). Below each table: settings
within the tolerance, mean |change|, and the settings that are worse by more than 10 % (to be explained or fixed
before the baseline is used).
Learning-rate check (scripts/exps.py baselines --backbone_lr ...; jobs <name>@lr<x>): where such runs exist, the
setting takes the learning rate with the lowest validation loss among the official one and the checked ones; the
cell then names it ([lr x]). --official_only ignores these runs.
"""
import argparse, collections, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
import baselines  # noqa: E402

DS_OF_FILE = {f: d for d, f in baselines.FILES.items()}


def load(files, any_protocol=False, official_only=False):
    runs = {}
    default = [d / f'results_{s}.jsonl' for d in (paths.REPO / 'results' / 'p1', paths.WORK)
               for s in ('baselines', 'plugin_all', 'plugin_fusion')]
    for f in files or default:
        f = Path(f)
        if not f.exists():
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = r.get('plug_base') if r.get('model') == 'DLTPlug' else r.get('model')
            if r.get('model') == 'DLTPlug' and not r.get('des', '').endswith('_alone'):
                continue
            if model not in baselines.BASELINES or r.get('data_path') not in DS_OF_FILE:
                continue
            if not any_protocol and not (r.get('freq') == 'h' and r.get('drop_last')):
                continue                                         # an earlier protocol (records before 2026-11-17)
            runs[(r['des'], r.get('seed'), r.get('model'))] = (model, r)
    variants = collections.defaultdict(lambda: collections.defaultdict(list))   # (model, ds, H) -> {suffix: records}
    for model, r in runs.values():
        ds, H = DS_OF_FILE[r['data_path']], r['pred_len']
        sfx = r['des'][len(f'{ds}_{H}_{model}'):] if r['des'].startswith(f'{ds}_{H}_{model}') else ''
        sfx = sfx.replace('_alone', '')                         # plug-in suites: arm "alone"
        if sfx and not sfx.startswith('@lr'):
            continue
        variants[(model, ds, H)][sfx].append(r)
    cells, chosen = {}, {}
    for key, v in variants.items():
        vali = lambda rs: statistics.mean(r['best_vali'] for r in rs if r.get('best_vali') is not None) \
            if any(r.get('best_vali') is not None for r in rs) else float('inf')
        sfx = '' if official_only or len(v) == 1 else min(v, key=lambda x: vali(v[x]))
        if sfx not in v:
            continue
        cells[key] = v[sfx]
        if sfx:
            chosen[key] = sfx[3:]
    return cells, chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+')
    p.add_argument('--models', nargs='+', default=list(baselines.BASELINES))
    p.add_argument('--tol', type=float, default=5.0)
    p.add_argument('--any_protocol', action='store_true', help='also records of the earlier protocol')
    p.add_argument('--official_only', action='store_true', help='ignore the runs of the learning-rate check')
    p.add_argument('--out')
    a = p.parse_args()
    R = json.load(open(Path(__file__).with_name('reported.json')))['smamba_paper']
    cells, chosen = load(a.files, a.any_protocol, a.official_only)
    tol = a.tol / 100
    fmt = lambda d: f'{100 * d:+.1f} %'
    out = ['# Baseline check against the S-Mamba paper', '',
           f'Change = ours / published - 1 (MSE per horizon; MAE averaged over the horizons). * more than {a.tol:g} % '
           f'worse than published, ** more than 10 %. Seeds averaged; n = runs per setting.']
    summary = []
    for m in a.models:
        rows, changes, bad = [], [], []
        for ds in baselines.FILES:
            hs = baselines.horizons(ds)
            got = [(H, cells.get((m, ds, H))) for H in hs]
            if not any(rs for _, rs in got):
                continue
            line, mae_ch = [], []
            for H, rs in got:
                ref = R[m][ds][str(H)]
                if not rs:
                    line.append(f'- / {ref[0]:.3f}')
                    continue
                mse = statistics.mean(r['mse'] for r in rs)
                mae = statistics.mean(r['mae'] for r in rs)
                d = mse / ref[0] - 1
                mae_ch.append(mae / ref[1] - 1)
                changes.append(d)
                mark = '**' if d > 0.10 else ('*' if d > tol else '')
                if d > 0.10:
                    bad.append(f'{ds}-{H} {fmt(d)}')
                lr = chosen.get((m, ds, H))
                line.append(f'{mse:.3f} / {ref[0]:.3f} ({fmt(d)}){mark}' + (f' n={len(rs)}' if len(rs) > 1 else '')
                            + (f' [lr {lr}]' if lr else ''))
            rows.append(f'| {ds} | ' + ' | '.join(line) + f' | {fmt(statistics.mean(mae_ch)) if mae_ch else "-"} |')
        name = baselines.NAME.get(m, m)
        if not rows:
            summary.append(f'| {name} | 0 | - | - | - |')
            continue
        out += ['', f'## {name}', '', '| dataset | horizon 1 | horizon 2 | horizon 3 | horizon 4 | MAE |',
                '|---|---|---|---|---|---:|'] + rows
        within = sum(d <= tol for d in changes)
        out += ['', f'{len(changes)} settings; within {a.tol:g} %: {within}; mean |change| '
                    f'{100 * statistics.mean(abs(d) for d in changes):.1f} %; more than 10 % worse: '
                    f'{", ".join(bad) if bad else "none"}.']
        summary.append(f'| {name} | {len(changes)} | {within} | {100 * statistics.mean(abs(d) for d in changes):.1f} % | '
                       f'{len(bad)} |')
    out += ['', '## Summary', '', f'| baseline | settings run | within {a.tol:g} % | mean abs. change | > 10 % worse |',
            '|---|---:|---:|---:|---:|'] + summary
    text = '\n'.join(out) + '\n'
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == '__main__':
    main()
