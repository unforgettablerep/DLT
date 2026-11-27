"""Plug-in diagnosis: accuracy and training length of every arm, side by side.

    python scripts/plugin/plugin_diag.py [store.tsv ...]

Reads output/plugin/plugin_all.tsv, output/plugin/plugin_fusion.tsv and output/ablation/allnb.tsv, or the given stores (a run present in several is taken from the last). For every
dataset, horizon and backbone it prints one line per arm (alone, + DLT-dyn, and the probe arms samelr / gate / gate_samelr): test MSE and
MAE, the change against the backbone alone, epochs run, the epoch of the best validation loss, and that loss.
An arm that stops much earlier than the backbone alone points to the training recipe (the fast DLT branch
triggers early stopping before the backbone has converged), not to the branch itself.
"""
import collections, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import OUTPUT  # noqa: E402
from src.utils.result_io import read_results  # noqa: E402

DEFAULT_FILES = [OUTPUT / 'plugin' / 'plugin_all.tsv', OUTPUT / 'plugin' / 'plugin_fusion.tsv', OUTPUT / 'ablation' / 'allnb.tsv']
ARMS = ['alone', 'dlt', 'dlt_samelr', 'dlt_gate', 'dlt_gate_samelr', 'dlt_all']
LABEL = {'alone': 'alone', 'dlt': '+ DLT-dyn (branch lr 3e-3)', 'dlt_samelr': '+ DLT-dyn, backbone lr',
         'dlt_gate': '+ DLT-dyn, static gate', 'dlt_gate_samelr': '+ DLT-dyn, gate + backbone lr',
         'dlt_all': '+ DLT-dyn, all neighbours'}

runs = {}
for f in sys.argv[1:] or DEFAULT_FILES:
    for r in read_results(f):
        runs[(os.path.basename(str(f)), r.get('des'), r.get('seed'))] = r
rows = list(runs.values())
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r.get('model') != 'DLTPlug':
        continue
    stem = f"{r['dataset']}_{r['pred_len']}_{r['plug_base']}_"
    arm = r['des'][len(stem):] if r['des'].startswith(stem) else r['des']
    groups[(r['dataset'], r['pred_len'], r['plug_base'])][arm].append(r)

m = lambda rs, k: np.mean([r[k] for r in rs if r.get(k) is not None]) if rs else float('nan')
for (ds, H, base), arms in sorted(groups.items()):
    pems = ds.startswith('PEMS')
    k1, k2 = ('mae', 'rmse') if pems else ('mse', 'mae')
    print(f'\n{ds} H={H} {base}   ({k1.upper()} / {k2.upper()})')
    ref = m(arms.get('alone', []), k1)
    for arm in [a for a in ARMS if a in arms] + sorted(set(arms) - set(ARMS)):
        rs = arms[arm]
        delta = f'{100 * (m(rs, k1) / ref - 1):+6.1f} %' if arm != 'alone' and ref == ref else '        '
        print(f"  {LABEL.get(arm, arm):32s} n={len(rs)}  {m(rs, k1):9.4f} / {m(rs, k2):8.4f}  {delta}   "
              f"epochs {m(rs, 'epochs'):4.1f}  best epoch {m(rs, 'best_epoch'):4.1f}  best val {m(rs, 'best_vali'):.4f}")
