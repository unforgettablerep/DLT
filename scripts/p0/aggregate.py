"""Tables of the priority-0 experiments.

    python scripts/p0/aggregate.py results/p0/results_p0.jsonl results/p0/summary.md
"""
import json, sys, collections
import numpy as np
from scipy import stats

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(l) for l in open(path) if l.strip()]
SETTINGS = ['ETTh1_96', 'Weather_96', 'ECL_96', 'PEMS08_24', 'PEMS08_96']
CONFIGS = [('ci', 'none'), ('scalar', 'none'), ('shift', 'none'), ('tap3', 'none'), ('full', 'none'),
           ('topk', 'none'), ('mill', 'none'), ('milldyn', 'none'), ('scalardyn', 'none'),
           ('dlt', 'none'), ('dltdyn', 'none'), ('dltdyn', 'moe'), ('dltdyn', 'static')]
LABEL = {('ci', 'none'): 'CI (no cross branch)', ('scalar', 'none'): 'scalar (lag 0)',
         ('shift', 'none'): 'shift (CCF peak lag, LIFT-style)', ('tap3', 'none'): 'tap3 (CrossLinear-style)',
         ('full', 'none'): 'full (free, lags 0..48)', ('topk', 'none'): 'top-5 lags, free weights',
         ('mill', 'none'): 'MillGNN-static (top-5 + decay)', ('milldyn', 'none'): 'MillGNN-dyn (top-5 + decay + attention)',
         ('scalardyn', 'none'): 'scalar-dyn', ('dlt', 'none'): '**DLT**', ('dltdyn', 'none'): '**DLT-dyn**',
         ('dltdyn', 'moe'): 'DLT-dyn + MoVE softmax gate', ('dltdyn', 'static'): 'DLT-dyn + static gate (init 0.047)'}
GROUP_BREAK = {('full', 'none'), ('milldyn', 'none'), ('dltdyn', 'none')}   # horizontal separators after these

R = collections.defaultdict(list)
for r in rows:
    s = r['model_id'].replace('P0_', '')
    R[(s, r['variant'], r['gate'])].append(r)

def vals(s, c, key='mse'):
    return np.array([r[key] for r in sorted(R[(s, *c)], key=lambda r: r['seed'])])

lines = []
P = lines.append
n_runs = len(rows)
P(f'# Priority-0 experiments: kernel and gate comparison\n')
P(f'{n_runs} runs. Time-Series-Library, look-back 96, k = 8 neighbours, L = 48, R = 4, top-K = 5, Adam lr 3e-3 '
  f'(halved every epoch), batch 32, 10 epochs, early stopping patience 3 on the validation MSE, '
  f'seeds 2021/2022/2023. Entries: mean ± std over seeds; n < 3 seeds is marked.\n')

def table(key, title):
    P(f'## {title}\n')
    P('| configuration | ' + ' | '.join(SETTINGS) + ' |')
    P('|---|' + '---:|' * len(SETTINGS))
    means = {s: {c: vals(s, c, key).mean() for c in CONFIGS if len(vals(s, c, key))} for s in SETTINGS}
    order = {s: sorted(means[s], key=means[s].get) for s in SETTINGS}
    for c in CONFIGS:
        cells = []
        for s in SETTINGS:
            v = vals(s, c, key)
            if not len(v):
                cells.append('–'); continue
            txt = f'{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}' + ('' if len(v) == 3 else f' (n={len(v)})')
            if order[s][0] == c: txt = f'**{txt}**'
            elif len(order[s]) > 1 and order[s][1] == c: txt = f'<u>{txt}</u>'
            cells.append(txt)
        P(f'| {LABEL[c]} | ' + ' | '.join(cells) + ' |')
        if c in GROUP_BREAK:
            P('| | ' + ' | '.join([''] * len(SETTINGS)) + ' |')
    P('\nBold: best mean; underlined: second best.\n')

table('mse', 'Test MSE')
table('mae', 'Test MAE')

# relative change against CI and against DLT-dyn
P('## Relative test MSE against the channel-independent model (negative = better)\n')
P('| configuration | ' + ' | '.join(SETTINGS) + ' |')
P('|---|' + '---:|' * len(SETTINGS))
for c in CONFIGS[1:]:
    cells = []
    for s in SETTINGS:
        a, b = vals(s, c), vals(s, ('ci', 'none'))
        cells.append(f'{100 * (a.mean() / b.mean() - 1):+.1f}%' if len(a) and len(b) else '–')
    P(f'| {LABEL[c]} | ' + ' | '.join(cells) + ' |')
P('')

P('## DLT-dyn against the other kernels (Welch t-test on the seed means)\n')
P('Δ = MSE(other) / MSE(DLT-dyn) − 1; positive means DLT-dyn is better. p: two-sided Welch t-test, 3 vs 3 seeds.\n')
P('| other kernel | ' + ' | '.join(SETTINGS) + ' |')
P('|---|' + '---|' * len(SETTINGS))
for c in [('full', 'none'), ('topk', 'none'), ('mill', 'none'), ('milldyn', 'none'), ('scalardyn', 'none'), ('dlt', 'none')]:
    cells = []
    for s in SETTINGS:
        a, b = vals(s, c), vals(s, ('dltdyn', 'none'))
        if len(a) >= 2 and len(b) >= 2:
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            cells.append(f'{100 * (a.mean() / b.mean() - 1):+.1f}% (p={p:.3f})')
        else:
            cells.append('–')
    P(f'| {LABEL[c]} | ' + ' | '.join(cells) + ' |')
P('')

P('## Gates on DLT-dyn\n')
P('| gate | ' + ' | '.join(SETTINGS) + ' |')
P('|---|' + '---|' * len(SETTINGS))
for c in [('ci', 'none'), ('dltdyn', 'none'), ('dltdyn', 'moe'), ('dltdyn', 'static')]:
    cells = []
    for s in SETTINGS:
        v, ep = vals(s, c), vals(s, c, 'best_epoch')
        cells.append(f'{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f} (best ep. {ep.mean():.1f})' if len(v) else '–')
    P(f'| {LABEL[c]} | ' + ' | '.join(cells) + ' |')
P('')

P('## Training: epochs run / best epoch / seconds per epoch (1 CPU thread) / parameters\n')
P('| configuration | ' + ' | '.join(SETTINGS) + ' |')
P('|---|' + '---|' * len(SETTINGS))
for c in CONFIGS:
    cells = []
    for s in SETTINGS:
        rs = R[(s, *c)]
        if not rs:
            cells.append('–'); continue
        ep = np.mean([r['epochs'] for r in rs]); be = np.mean([r['best_epoch'] for r in rs])
        spe = np.mean([np.mean(r['epoch_secs']) for r in rs]); npar = rs[0]['n_params']
        cells.append(f'{ep:.1f} / {be:.1f} / {spe:.0f}s / {npar / 1e3:.1f}K')
    P(f'| {LABEL[c]} | ' + ' | '.join(cells) + ' |')
P('')

txt = '\n'.join(lines)
print(txt)
if out:
    open(out, 'w').write(txt + '\n')
