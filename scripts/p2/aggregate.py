"""Tables of the P2 experiments.

    python scripts/p2/aggregate.py <P0 results_p0.jsonl> <work dir with results_sensitivity / results_neighbours>
                                   <results/p2 dir with efficiency_*.jsonl and kernels_pems08.json> results/p2/summary.md
"""
import json, os, sys, collections, glob
import numpy as np

p0_path, work, p2dir, out = sys.argv[1:5]
FILES = {'ETTh1.csv': 'ETTh1', 'weather.csv': 'Weather', 'PEMS08.npz': 'PEMS08'}
load = lambda f: [json.loads(l) for l in open(f)] if os.path.exists(f) else []
p0 = load(p0_path)
sens = load(os.path.join(work, 'results_sensitivity.jsonl'))
nb = load(os.path.join(work, 'results_neighbours.jsonl'))


def ms(v, d=4):
    v = np.asarray(v, float)
    if not len(v):
        return '–'
    return f'{v.mean():.{d}f}' + (f' ± {v.std(ddof=1):.{d}f}' if len(v) > 1 else '') + ('' if len(v) >= 3 else f' (n={len(v)})')


L = []
P = L.append
P('# P2: sensitivity, neighbour selection, efficiency, learned kernels\n')
settings = [('ETTh1', 96), ('Weather', 96), ('PEMS08', 96)]
default = {(s, H): [r['mse'] for r in p0 if r['model_id'] == f'P0_{s}_{H}' and r['variant'] == 'dltdyn' and r['gate'] == 'none']
           for s, H in settings}

if sens:
    P('## Sensitivity of DLT-dyn (test MSE, one factor at a time; default L = 48, R = 4, k = 8 from the P0 grid)\n')
    for factor, values, dflt in (('L', (12, 24, 48, 96), 48), ('R', (2, 4, 8), 4), ('k', (4, 8, 16), 8)):
        P(f'| {factor} | ' + ' | '.join(f'{s}-{H}' for s, H in settings) + ' |')
        P('|---|' + '---|' * len(settings))
        for v in values:
            cells = []
            for s, H in settings:
                if v == dflt:
                    cells.append(ms(default[(s, H)]) + ' (default)')
                else:
                    cells.append(ms([r['mse'] for r in sens if r['des'] == f'{s}_{H}_{factor}{v}']))
            P(f'| {v} | ' + ' | '.join(cells) + ' |')
        P('')
    P('ETTh1 has 7 variables, so k is capped at 6 (k = 8 and k = 16 coincide).\n')

if nb:
    P('## Neighbour selection (test MSE)\n')
    P('| variant | neighbours | Weather-96 | PEMS08-96 |')
    P('|---|---|---|---|')
    for v in ('dlt', 'dltdyn'):
        for mode in ('ccf', 'random', 'all'):
            cells = []
            for s in ('Weather', 'PEMS08'):
                if mode == 'ccf':
                    cells.append(ms([r['mse'] for r in p0 if r['model_id'] == f'P0_{s}_96' and r['variant'] == v and r['gate'] == 'none']))
                else:
                    cells.append(ms([r['mse'] for r in nb if r['des'] == f'{s}_96_{v}_{mode}']))
            label = {'ccf': 'CCF top-8 (default)', 'random': '8 random', 'all': 'all other variables'}[mode]
            P(f"| {'DLT-dyn' if v == 'dltdyn' else 'DLT'} | {label} | " + ' | '.join(cells) + ' |')
    P('\nCI reference (no neighbours): ' + ', '.join(
        f"{s}-96 {ms([r['mse'] for r in p0 if r['model_id'] == f'P0_{s}_96' and r['variant'] == 'ci'])}" for s in ('Weather', 'PEMS08')) + '\n')

for f in sorted(glob.glob(os.path.join(p2dir, 'efficiency_*.jsonl'))):
    rows = [r for r in load(f) if 'error' not in r]
    if not rows:
        continue
    dev = rows[0]['device'] + (f" ({rows[0]['gpu']})" if rows[0].get('gpu') else f", {rows[0]['threads']} threads")
    dss = [d for d in ('Weather', 'PEMS08', 'ECL', 'Traffic') if any(r['dataset'] == d for r in rows)]
    P(f'## Efficiency on {dev}: parameters / training ms per batch / inference ms per batch / peak memory MB\n')
    P('Batch 32, input 96, horizon 96, random data; baselines with the official TSLib hyper-parameters of each dataset.\n')
    P('| model | ' + ' | '.join(dss) + ' |')
    P('|---|' + '---|' * len(dss))
    models = list(dict.fromkeys(r['model'] for r in rows))
    for m in models:
        cells = []
        for d in dss:
            r = next((x for x in rows if x['model'] == m and x['dataset'] == d), None)
            cells.append(f"{r['params'] / 1e3:.0f}K / {r['train_ms']:.0f} / {r['infer_ms']:.0f} / {r['mem_MB']:.0f}" if r else '–')
        P(f'| {m} | ' + ' | '.join(cells) + ' |')
    P('\nMillGNN (reported, A100, Traffic): 9.7 GB GPU memory; iTransformer 7.5 GB in the same figure.\n')

kj = os.path.join(p2dir, 'kernels_pems08.json')
if os.path.exists(kj):
    K = json.load(open(kj))
    P('## Learned kernels on PEMS08 vs the road network (mean over seeds)\n')
    keys = [('hop_share_selected', '1', 'selected neighbours 1 hop away'),
            ('hop_share_all_pairs', '1', 'all pairs 1 hop away'),
            ('hop_share_selected', '2', 'selected neighbours 2 hops away'),
            ('hop_share_all_pairs', '2', 'all pairs 2 hops away'),
            ('upstream_share', None, 'selected neighbours upstream of the target'),
            ('mass_beyond_lag0', None, 'kernel mass beyond lag 0'),
            ('share_multi_lag', None, 'kernels with >= 2 lags at >= 50 % of the peak'),
            ('peak_lag_mean_upstream_only', None, 'mean peak lag, upstream-only sources (steps)'),
            ('peak_lag_mean_downstream_only', None, 'mean peak lag, downstream-only sources (steps)'),
            ('spearman_distance_peak', 0, 'Spearman rho(road distance, peak lag)'),
            ('spearman_distance_centroid', 0, 'Spearman rho(road distance, lag centroid)')]
    P('| quantity | DLT | DLT-dyn |')
    P('|---|---|---|')
    for key, sub, label in keys:
        cells = []
        for v in ('dlt', 'dltdyn'):
            seeds = K.get(v, {}).get('per_seed', [])
            vals = [s[key][sub] if sub is not None else s[key] for s in seeds if s.get(key) is not None]
            cells.append((f'{np.mean(vals):.3f}' + (f' ± {np.std(vals, ddof=1):.3f}' if len(vals) > 1 else '')) if vals else '–')
        P(f'| {label} | ' + ' | '.join(cells) + ' |')
    P(f"\nCheckpoints: {sum(len(K[v]['per_seed']) for v in K)} (DLT and DLT-dyn, PEMS08-96, P0 grid). Figure: `figures/fig_kernels_pems08.pdf` in the DLTpaper repository (written by scripts/p2/kernels.py).\n")

txt = '\n'.join(L)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
open(out, 'w').write(txt + '\n')
print(txt)
