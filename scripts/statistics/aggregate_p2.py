"""Tables of the P2 experiments: sensitivity, neighbour selection, efficiency, learned kernels.

    python scripts/statistics/aggregate_p2.py [out.md]

Reads output/ablation/p0.tsv (the default recipe of the P0 grid), output/hyperparam/sensitivity.tsv,
output/hyperparam/neighbours.tsv, output/efficiency/efficiency_*.jsonl and output/mechanism/kernels_pems08.json,
and writes output/statistics/summary_p2.md (or the given file).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.configs.dlt_profiles import EFFICIENCY_SETTING  # noqa: E402
from src.utils.paths import OUTPUT  # noqa: E402
from src.utils.result_io import read_results  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else str(OUTPUT / 'statistics' / 'summary_p2.md')
p0 = read_results(OUTPUT / 'ablation' / 'p0.tsv')
sens = read_results(OUTPUT / 'hyperparam' / 'sensitivity.tsv')
nb = read_results(OUTPUT / 'hyperparam' / 'neighbours.tsv')
load = lambda f: [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def p0_mse(s, H, variant, gate='none'):
    return [r['mse'] for r in p0 if r['dataset'] == s and r['pred_len'] == H and r['dlt_variant'] == variant
            and (gate is None or r['dlt_gate'] == gate)]


def ms(v, d=4):
    v = np.asarray(v, float)
    if not len(v):
        return '–'
    return f'{v.mean():.{d}f}' + (f' ± {v.std(ddof=1):.{d}f}' if len(v) > 1 else '') + ('' if len(v) >= 3 else f' (n={len(v)})')


L = []
P = L.append
P('# P2: sensitivity, neighbour selection, efficiency, learned kernels\n')
settings = [('ETTh1', 96), ('Weather', 96), ('PEMS08', 96)]
default = {(s, H): p0_mse(s, H, 'dltdyn') for s, H in settings}

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
                    cells.append(ms(p0_mse(s, 96, v)))
                else:
                    cells.append(ms([r['mse'] for r in nb if r['des'] == f'{s}_96_{v}_{mode}']))
            label = {'ccf': 'CCF top-8 (default)', 'random': '8 random', 'all': 'all other variables'}[mode]
            P(f"| {'DLT-dyn' if v == 'dltdyn' else 'DLT'} | {label} | " + ' | '.join(cells) + ' |')
    P('\nCI reference (no neighbours): ' + ', '.join(
        f"{s}-96 {ms(p0_mse(s, 96, 'ci', None))}" for s in ('Weather', 'PEMS08')) + '\n')


def efficiency_table(rows, dss, main):
    """Rows of one setting: a block of the main models, then a block of the other configurations."""
    note = ''
    seen = list(dict.fromkeys(r['model'] for r in rows))
    for block in ([m for m in main if m in seen], [m for m in seen if m not in main]):
        if not block:
            continue
        P('| model | ' + ' | '.join(dss) + ' |')
        P('|---|' + '---|' * len(dss))
        for m in block:
            cells = []
            for d in dss:
                r = next((x for x in reversed(rows) if x['model'] == m and x['dataset'] == d), None)    # latest record
                cells.append(f"{r['params'] / 1e3:.0f}K / {r['train_ms']:.0f} / {r['infer_ms']:.0f} / {r['mem_MB']:.0f}" if r else '–')
                if r and r.get('mamba_impl') == 'torch':
                    note = '\nS-Mamba was measured with the pure-PyTorch Mamba block (mamba-ssm not installed): its time is an upper bound.\n'
            P(f'| {m} | ' + ' | '.join(cells) + ' |')
        P('')
    if note:
        P(note)


def device(rows):
    return rows[0]['device'] + (f" ({rows[0]['gpu']})" if rows[0].get('gpu') else f", {rows[0]['threads']} threads")


files = sorted(glob.glob(str(OUTPUT / 'efficiency' / 'efficiency_*.jsonl')))
for f in [f for f in files if not os.path.basename(f).startswith('efficiency_smamba_')]:
    rows = [r for r in load(f) if 'error' not in r]
    if not rows:
        continue
    dss = [d for d in ('Weather', 'PEMS08', 'ECL', 'Traffic') if any(r['dataset'] == d for r in rows)]
    P(f'## Efficiency on {device(rows)}: parameters / training ms per batch / inference ms per batch / peak memory MB\n')
    P('Batch 32, input 96, horizon 96, random data; baselines with their official hyper-parameters for each dataset. '
      'First block: DLT-dyn and the eight representative baselines; second block: kernel variants of the controlled '
      'study, the plug-in and further configurations.\n')
    efficiency_table(rows, dss, ['DLT-dyn', 'Informer', 'TimesNet', 'DLinear', 'PatchTST', 'iTransformer', 'TimeMixer',
                                 'TimeXer', 'S-Mamba', 'RLinear', 'TiDE'])
    P('\nMillGNN (reported, A100, Traffic): 9.7 GB GPU memory; iTransformer 7.5 GB in the same figure.\n')

for f in [f for f in files if os.path.basename(f).startswith('efficiency_smamba_')]:
    records = load(f)
    rows = [r for r in records if 'error' not in r]
    if not rows:
        continue
    dss = [d for d in EFFICIENCY_SETTING if any(r['dataset'] == d for r in rows)]
    P(f'## Efficiency in the S-Mamba setting on {device(rows)}: parameters / training ms per batch / inference ms per batch / peak memory MB\n')
    P('Batch 16, input 96; ' + ', '.join(f'{d} horizon {EFFICIENCY_SETTING[d]}' for d in dss) + ' (the efficiency '
      'experiment of the S-Mamba paper, Fig. 5); random data; baselines with their official hyper-parameters for each '
      'dataset. DLT-dyn (tuned): the recipe selected on the validation loss in the eff_tune rounds '
      '(output/efficiency/eff_tune.md), on the basis path as trained and on the equivalent kernel path.\n')
    efficiency_table(rows, dss, ['DLT-dyn', 'DLT-dyn (tuned)', 'DLT-dyn (tuned, kernel path)', 'S-Mamba',
                                 'iTransformer', 'PatchTST', 'RLinear', 'DLinear', 'TiDE'])
    ok = {(r['dataset'], r['model']) for r in rows}
    failed = {(r['dataset'], r['model']): r['error'] for r in records if 'error' in r and (r['dataset'], r['model']) not in ok}
    if failed:
        P('Not measured: ' + '; '.join(f'{m} on {d}: {e}' for (d, m), e in failed.items()) + '.\n')

kj = OUTPUT / 'mechanism' / 'kernels_pems08.json'
if kj.exists():
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
    P(f"\nCheckpoints: {sum(len(K[v]['per_seed']) for v in K)} (DLT and DLT-dyn, PEMS08-96, P0 grid). Figure: `figures/fig_kernels_pems08.pdf` in the DLTpaper repository (written by scripts/mechanism/kernels.py).\n")

txt = '\n'.join(L)
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
open(out, 'w').write(txt + '\n')
print(txt)
