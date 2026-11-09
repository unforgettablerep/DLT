"""Tables of the P1 experiments (our runs: mean ± std over seeds; published numbers marked as reported).

    python scripts/p1/aggregate.py <work dir with results_*.jsonl> results/p1/summary.md
"""
import json, os, sys, collections
from pathlib import Path
import numpy as np
from scipy import stats

work, out = sys.argv[1], sys.argv[2]
REP = json.load(open(Path(__file__).with_name('reported.json')))
HS = ['96', '192', '336', '720']


FILES = {'ETTh1.csv': 'ETTh1', 'ETTh2.csv': 'ETTh2', 'ETTm1.csv': 'ETTm1', 'ETTm2.csv': 'ETTm2', 'weather.csv': 'Weather',
         'exchange_rate.csv': 'Exchange', 'electricity.csv': 'ECL', 'traffic.csv': 'Traffic', 'PEMS04.npz': 'PEMS04',
         'PEMS08.npz': 'PEMS08'}
dsname = lambda r: FILES[r['data_path']]
LONG = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'ECL', 'Traffic')


def load(suite):
    f = os.path.join(work, f'results_{suite}.jsonl')
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def ms(v, d=3, scale=1.0):
    v = np.asarray(v, float) * scale
    if not len(v):
        return '–'
    s = f'{v.mean():.{d}f}'
    return s + (f' ± {v.std(ddof=1):.{d}f}' if len(v) > 1 else '') + ('' if len(v) >= 3 else f' (n={len(v)})')


L = []
P = L.append
P('# P1: comparison under the standard protocols\n')
P('Our runs: Time-Series-Library, CPU, mean ± std over seeds 2021-2023 (n marks fewer seeds). '
  'Rows marked *reported* are copied from the papers: ' + REP['_sources']['long_term'] + ' ' + REP['_sources']['pems'] + '\n')

# ---------------------------------------------------------------- PEMS, horizon 12
pems = load('pems12')
if pems:
    P('## PEMS04 / PEMS08, input 96, horizon 12 (TimeMixer protocol)\n')
    P('MAE / MAPE (%) / RMSE on the original scale; MAPE as in TimeMixer (ratios > 5 count as 0). '
      'Ours: early stopping on the validation MAE (original scale).\n')
    P('| model | PEMS04 MAE | MAPE | RMSE | PEMS08 MAE | MAPE | RMSE |')
    P('|---|---|---|---|---|---|---|')
    names = {'dltdyn': 'DLT-dyn (L1 loss)', 'dltdyn_mse': 'DLT-dyn (MSE loss)', 'dlt': 'DLT (L1 loss)', 'ci': 'CI linear (L1 loss)'}
    for key, label in names.items():
        cells = []
        for ds in ('PEMS04', 'PEMS08'):
            rs = [r for r in pems if r['des'] == f'{ds}_{key}']
            cells += [ms([r['mae'] for r in rs], 2), ms([r['mape_clip5'] for r in rs], 2, 100), ms([r['rmse'] for r in rs], 2)]
        P(f'| {label} | ' + ' | '.join(cells) + ' |')
    for m, v in REP['pems'].items():
        P(f'| {m} *reported* | ' + ' | '.join(f'{x:.2f}' for ds in ('PEMS04', 'PEMS08') for x in v[ds]) + ' |')
    P('')

# ---------------------------------------------------------------- long-term benchmark
lt = load('long_small') + load('long_large')
tuned = load('tuned')
ci = load('ci_long')
if lt or tuned:
    P('## Long-term forecasting, input 96 (MSE / MAE)\n')
    others = ['MillGNN', 'MoVE', 'TimeMixer', 'iTransformer', 'PatchTST', 'DLinear']
    ours = (('DLT-dyn (default)', lt), ('DLT-dyn (tuned)', tuned), ('CI linear (ours)', ci))
    P('| dataset | H | ' + ' | '.join(o[0] for o in ours) + ' | ' + ' | '.join(f'{m} *rep.*' for m in others) + ' |')
    P('|---|---|' + '---|' * (len(ours) + len(others)))
    for ds in LONG:
        sets = [[r for r in rows if dsname(r) == ds] for _, rows in ours]
        if not any(sets):
            continue
        avg = [[] for _ in ours]
        for H in HS + ['Avg']:
            cells = []
            for i, rows in enumerate(sets):
                if H != 'Avg':
                    d = [r for r in rows if str(r['pred_len']) == H]
                    if d: avg[i].append((np.mean([r['mse'] for r in d]), np.mean([r['mae'] for r in d])))
                    cells.append(f"{ms([r['mse'] for r in d])} / {ms([r['mae'] for r in d])}" if d else '–')
                else:
                    cells.append(f'{np.mean([a[0] for a in avg[i]]):.3f} / {np.mean([a[1] for a in avg[i]]):.3f}'
                                 if len(avg[i]) == 4 else '–')
            rep = []
            for m in others:
                v = REP['long_term'].get(m, {}).get(ds, {}).get(H)
                rep.append(f'{v[0]:.3f} / {v[1]:.3f}' if v else '–')
            P(f'| {ds} | {H} | ' + ' | '.join(cells) + ' | ' + ' | '.join(rep) + ' |')
    P('\nDLT-dyn (tuned): configuration chosen by the validation search (scripts/p1/hpsearch.py, see tuned.json). '
      'CI linear: the same model without the cross-variable branch (RevIN + one linear map per variable).\n')

# ---------------------------------------------------------------- plug-in
plug = load('plugin') + load('plugin_large')
if plug:
    P('## Plug-in: backbone alone vs backbone + DLT-dyn branch\n')
    P('Same backbone hyper-parameters and training budget in both columns; the DLT branch has its own learning '
      'rate (3e-3). PEMS08: horizon 12, TimeMixer protocol (MAE / RMSE on the original scale); otherwise MSE / MAE. '
      'Δ = relative change of the first metric; p = Welch t-test over seeds.\n')
    P('| dataset | H | backbone | alone | + DLT-dyn | Δ | p |')
    P('|---|---|---|---|---|---|---|')
    groups = collections.defaultdict(lambda: {'alone': [], 'dlt': []})
    for r in plug:
        ds = dsname(r)
        groups[(ds, r['pred_len'], r['plug_base'])]['alone' if r['variant'] == 'ci' else 'dlt'].append(r)
    for (ds, H, base), g in sorted(groups.items()):
        k1, k2 = ('mae', 'rmse') if ds.startswith('PEMS') else ('mse', 'mae')
        a, b = [r[k1] for r in g['alone']], [r[k1] for r in g['dlt']]
        dd = 2 if ds.startswith('PEMS') else 3
        delta = f'{100 * (np.mean(b) / np.mean(a) - 1):+.1f}%' if a and b else '–'
        p = f'{stats.ttest_ind(b, a, equal_var=False).pvalue:.3f}' if len(a) > 1 and len(b) > 1 else '–'
        P(f"| {ds} | {H} | {base} | {ms(a, dd)} / {ms([r[k2] for r in g['alone']], dd)} | "
          f"{ms(b, dd)} / {ms([r[k2] for r in g['dlt']], dd)} | {delta} | {p} |")
    P('')

# ---------------------------------------------------------------- re-run baselines
base = load('baselines')
if base:
    P('## Baselines re-run with the official TSLib hyper-parameters (MSE / MAE)\n')
    P('| dataset | H | model | ours (re-run) | reported |')
    P('|---|---|---|---|---|')
    g = collections.defaultdict(list)
    for r in base:
        g[(dsname(r), r['pred_len'], r['model'])].append(r)
    for (ds, H, m), rs in sorted(g.items()):
        v = REP['long_term'].get(m, {}).get(ds, {}).get(str(H))
        P(f"| {ds} | {H} | {m} | {ms([r['mse'] for r in rs])} / {ms([r['mae'] for r in rs])} | "
          + (f'{v[0]:.3f} / {v[1]:.3f}' if v else '–') + ' |')
    P('')
    # averages over the four horizons, every model trained by us with the same pipeline
    models = sorted({m for (_, _, m) in g})
    cols = [('DLT-dyn (default)', lt), ('DLT-dyn (tuned)', tuned), ('CI linear', ci)] + \
           [(m, [r for r in base if r['model'] == m]) for m in models]
    P('### Average MSE / MAE over the four horizons, all trained in this pipeline\n')
    P('| dataset | ' + ' | '.join(c[0] for c in cols) + ' |')
    P('|---|' + '---|' * len(cols))
    for ds in LONG:
        cells, best = [], None
        vals = []
        for _, rows in cols:
            per_h = [[r for r in rows if dsname(r) == ds and str(r['pred_len']) == H] for H in HS]
            if all(per_h):
                vals.append((np.mean([np.mean([r['mse'] for r in h]) for h in per_h]),
                             np.mean([np.mean([r['mae'] for r in h]) for h in per_h])))
            else:
                vals.append(None)
        if not any(vals):
            continue
        bm = min(v[0] for v in vals if v)
        cells = [('–' if v is None else (f'**{v[0]:.3f}** / {v[1]:.3f}' if v[0] == bm else f'{v[0]:.3f} / {v[1]:.3f}'))
                 for v in vals]
        P(f'| {ds} | ' + ' | '.join(cells) + ' |')
    P('\nBold: lowest average MSE among the models trained here.\n')

txt = '\n'.join(L)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
open(out, 'w').write(txt + '\n')
print(txt)
