"""What can a cross-variable branch add to a channel-independent linear forecaster, and why? A data diagnostic.

    python scripts/plugin/leadlag_diag.py [--datasets ETTh1 PEMS08 ...] [--targets 32] [--out output/plugin/leadlag_diag.json]

On the splits and the scaling of the runs (data_provider, input 96, TSLib PEMS split), per dataset:
  - the k = 8 neighbours of every variable, chosen as the DLT branch chooses them on the training split
    (src/layers/DLT_layers.py select_neighbours: max over lags 0..48 of |corr(dx_i(t), dx_j(t - lag))|); the mean |CCF| peak
    of the selected pairs, the share of pairs whose peak is at lag 0 (no lead) and the mean absolute correlation of
    the levels of a variable and its first neighbour (near-copies of a variable have about 1);
  - SHARED: one ridge forecaster from a variable's own input window (96 values) to its next H values, with the
    same weights for every variable (as DLinear), fitted on the training windows of all target variables;
  - three corrections of SHARED per target variable, ridge regressions of its residual on
      OWN   the variable's own window       -> what per-variable weights add (no other variable involved);
      NB    the windows of its 8 neighbours -> what a cross-variable branch adds to a shared channel-independent
                                               backbone, as the DLT branch does to DLinear;
      BOTH  own and neighbour windows       -> BOTH against OWN: what the neighbours' pasts add beyond the
                                               variable's own past (lead-lag information);
    every window minus its own mean, the ridge penalty of each chosen on the validation split, per horizon;
  - the change of the test MSE (ratio of the MSE summed over the target variables) and the same on the training
    split, which shows how much of an in-sample gain does not carry over to the test period.
At most --targets target variables per dataset (a fixed random sample); up to 6000 training and 2000 validation /
test windows per target (an even stride above that). CPU only; about 6 minutes for all 13 datasets on 4 cores.
"""
import argparse, contextlib, io, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
os.chdir(REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from run import parse_args  # noqa: E402
from src.configs.dlt_profiles import DATASET_META, get_pred_lens  # noqa: E402
from src.data_provider.data_factory import data_provider  # noqa: E402
from src.layers.DLT_layers import abs_ccf as _abs_ccf, diff_standardise as _diff_standardise, select_neighbours  # noqa: E402
from src.utils import paths  # noqa: E402

SEQ, K, L = 96, 8, 48
MAX_WIN = {'train': 6000, 'val': 2000, 'test': 2000}
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)                 # times the mean eigenvalue of F'F
FLAGS = ('train', 'val', 'test')


def splits(ds, H):
    m = DATASET_META[ds]
    flag, dfile, C, freq = m['data'], m['data_path'], m['enc_in'], m['freq']
    argv = ['--is_training', '1', '--model_id', 'diag', '--model', 'DLinear',
            '--data', flag, '--root_path', str(paths.root_of(dfile)), '--data_path', dfile, '--features', 'M',
            '--seq_len', str(SEQ), '--label_len', '48', '--pred_len', str(H), '--enc_in', str(C), '--dec_in', str(C),
            '--c_out', str(C), '--freq', freq, '--num_workers', '0']
    args = parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        return {f: np.asarray(data_provider(args, f)[0].data_x, dtype=np.float64) for f in FLAGS}


def windows(A, H, cols, flag):
    """demeaned input windows of the columns cols (n, len(cols), SEQ) and the next H values of cols[0] minus the
    mean of its input window (n, H)"""
    n = A.shape[0] - SEQ - H + 1
    starts = np.arange(0, n, max(1, -(-n // MAX_WIN[flag])))
    W = A[:, cols][starts[:, None] + np.arange(SEQ + H)[None]]             # (n, SEQ + H, len(cols))
    X = W[:, :SEQ].transpose(0, 2, 1)
    mu = X.mean(-1, keepdims=True)
    return X - mu, W[:, SEQ:, 0] - mu[:, 0]


def ridge_eval(F, Y, Fv, Yv, Ft, Yt, horizons):
    """ridge from F to Y; per horizon h (first h columns): training and test MSE (summed over the h steps, mean over
    the windows) at the penalty with the lowest validation MSE"""
    fm, ym = F.mean(0), Y.mean(0)
    F, Fv, Ft = F - fm, Fv - fm, Ft - fm
    e, V = np.linalg.eigh(F.T @ F)
    e = np.clip(e, 0, None)
    Q = V.T @ (F.T @ (Y - ym))
    P = {'train': F @ V, 'val': Fv @ V, 'test': Ft @ V}
    T = {'train': Y, 'val': Yv, 'test': Yt}
    best = {h: (np.inf, None) for h in horizons}
    for lam in LAMBDAS:
        Wl = Q / (e + lam * e.mean())[:, None]
        sv = ((P['val'] @ Wl + ym - Yv) ** 2).cumsum(1).mean(0)            # validation MSE up to every step
        for h in horizons:
            if sv[h - 1] < best[h][0]:
                best[h] = (sv[h - 1], Wl)
    out = {}
    for h in horizons:
        Wl = best[h][1][:, :h]
        out[h] = {f: float(((P[f] @ Wl + ym[:h] - T[f][:, :h]) ** 2).sum(1).mean()) for f in ('train', 'test')}
    return out


def run(ds, n_targets, seed=0):
    t0 = time.time()
    horizons = get_pred_lens(ds)
    Hm = max(horizons)
    S = splits(ds, Hm)
    C = S['train'].shape[1]
    idx, lag = select_neighbours(S['train'], K, L)
    x = _diff_standardise(S['train'], 8000, 'cpu')
    peak = torch.stack([_abs_ccf(x, t)[torch.arange(C)[:, None], idx] for t in range(L + 1)], -1).max(-1).values
    targets = np.arange(C) if C <= n_targets else np.sort(np.random.default_rng(seed).choice(C, n_targets, replace=False))
    tr = S['train']
    level = np.mean([abs(np.corrcoef(tr[:, i], tr[:, idx[i, 0]])[0, 1]) for i in targets
                     if tr[:, i].std() > 0 and tr[:, idx[i, 0]].std() > 0])

    # SHARED: one set of weights for every target variable, penalty chosen on the pooled validation windows
    G, Bm = np.zeros((SEQ, SEQ)), np.zeros((SEQ, Hm))
    for i in targets:
        X, Y = windows(S['train'], Hm, [int(i)], 'train')
        G += X[:, 0].T @ X[:, 0]
        Bm += X[:, 0].T @ Y
    sv = {}
    Ws = {lam: np.linalg.solve(G + lam * np.trace(G) / SEQ * np.eye(SEQ), Bm) for lam in LAMBDAS}
    for i in targets:
        X, Y = windows(S['val'], Hm, [int(i)], 'val')
        for lam, W in Ws.items():
            sv[lam] = sv.get(lam, 0) + ((X[:, 0] @ W - Y) ** 2).sum()
    W = Ws[min(sv, key=sv.get)]

    keys = ('shared', 'own', 'nb', 'both')
    acc = {h: {f'{k}_{f}': 0. for k in keys for f in ('train', 'test')} | {'nb_helps': 0, 'lead_helps': 0}
           for h in horizons}
    for i in targets:
        cols = [int(i)] + idx[i].tolist()
        F, R = {}, {}
        for f in FLAGS:
            X, Y = windows(S[f], Hm, cols, f)
            R[f] = Y - X[:, 0] @ W                                           # residual of SHARED
            F[f] = {'own': X[:, 0], 'nb': X[:, 1:].reshape(len(X), -1), 'both': X.reshape(len(X), -1)}
        res = {k: ridge_eval(F['train'][k], R['train'], F['val'][k], R['val'], F['test'][k], R['test'], horizons)
               for k in ('own', 'nb', 'both')}
        for h in horizons:
            a = acc[h]
            for f in ('train', 'test'):
                a[f'shared_{f}'] += float((R[f][:, :h] ** 2).sum(1).mean())
                for k in ('own', 'nb', 'both'):
                    a[f'{k}_{f}'] += res[k][h][f]
            a['nb_helps'] += int(res['nb'][h]['test'] < (R['test'][:, :h] ** 2).sum(1).mean())
            a['lead_helps'] += int(res['both'][h]['test'] < res['own'][h]['test'])
    out = dict(dataset=ds, variables=C, targets=len(targets), ccf_peak=float(peak.mean()),
               lag0_share=float((lag == 0).float().mean()), level_corr_first=float(level),
               secs=round(time.time() - t0, 1), horizons={})
    for h in horizons:
        a = acc[h]
        out['horizons'][h] = {
            'shared_test_mse': a['shared_test'] / len(targets) / h,
            **{f'{k}_{f}': a[f'{k}_{f}'] / a[f'shared_{f}'] - 1 for k in ('own', 'nb', 'both') for f in ('train', 'test')},
            'lead_train': a['both_train'] / a['own_train'] - 1, 'lead_test': a['both_test'] / a['own_test'] - 1,
            'nb_helps': a['nb_helps'], 'lead_helps': a['lead_helps']}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=list(DATASET_META))
    p.add_argument('--targets', type=int, default=32)
    p.add_argument('--out', default='')
    a = p.parse_args()
    torch.set_grad_enabled(False)
    rows = []
    print('change of the test MSE against SHARED, mean over the horizons (training split in brackets):\n'
          f'{"dataset":9s} {"C":>4s} {"CCF peak":>8s} {"lag 0":>6s} {"level r":>7s}  {"OWN":>15s}  {"NB":>15s}  '
          f'{"BOTH vs OWN":>15s}')
    for ds in a.datasets:
        if not paths.exists(DATASET_META[ds]['data_path']):
            print(f'{ds:9s} data file missing, skipped')
            continue
        r = run(ds, a.targets)
        rows.append(r)
        m = lambda key: 100 * np.mean([v[key] for v in r['horizons'].values()])
        cell = lambda k: f'{m(k + "_test"):+6.1f} ({m(k + "_train"):+5.1f})'
        print(f'{ds:9s} {r["variables"]:4d} {r["ccf_peak"]:8.3f} {100 * r["lag0_share"]:5.0f}% {r["level_corr_first"]:7.3f}  '
              f'{cell("own"):>15s}  {cell("nb"):>15s}  {cell("lead"):>15s}   [{r["targets"]} targets, {r["secs"]} s]',
              flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1) + '\n')
        print(f'written: {a.out}')


if __name__ == '__main__':
    main()
