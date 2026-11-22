"""Mechanism: learned lag kernels on PEMS08 against the road network.

    python scripts/mechanism/kernels.py [--fig_dir <dir>] [--H 96]

Fixed locations (src/utils/paths.py): PEMS08.npz and PEMS08_edges.csv from the data directory; the checkpoints of the
P0 grid (scripts/ablation/make_jobs.py p0, des PEMS08_<H>_<variant>_none) from temp/checkpoints/PEMS08, and those of
the runs before the refactor from work/ckpt if that directory exists; statistics to output/mechanism. The figure goes
to ../DLTpaper/figures if that repository is checked out next to this one, else to output/mechanism (ignored by git).

For every trained DLT / DLT-dyn checkpoint (all seeds) the kernels K_ij(tau) = sum_r alpha_ijr phi_r(tau) of the
selected pairs are compared with the road graph (PEMS08_edges.csv: directed sensor links with length):
  * hop distance of the selected neighbours (undirected) against all candidate pairs,
  * lag centroid sum_tau tau |K| / sum |K| and peak lag of the kernel against the road distance,
  * upstream (a directed path source -> target exists) against downstream sources.
DLT-dyn kernels are the mean input-adapted kernels over the test windows. Writes kernels_pems08.json (statistics)
to output/mechanism and fig_kernels_pems08.pdf / .png to --fig_dir. Figures are kept in the paper repository
(DLTpaper/figures), not in this one.
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils import paths  # noqa: E402
from src.utils.checkpoint import find_checkpoints  # noqa: E402

LEGACY_CKPT = REPO / 'work' / 'ckpt'      # checkpoints of the runs before the refactor (one folder per run)


def road_graph(edges_csv, C):
    e = pd.read_csv(edges_csv)
    A = csr_matrix((e['cost'].values, (e['from'].values, e['to'].values)), shape=(C, C))
    hops = shortest_path(A, directed=False, unweighted=True)
    dist = shortest_path(A, directed=False)                       # metres along the road graph
    down = shortest_path(A, directed=True, unweighted=True)       # down[a, b] < inf: a -> ... -> b along links
    return hops, dist, down


def kernels_from_ckpt(path, data_dir, H):
    sd = torch.load(path, map_location='cpu')
    idx, alpha, phi = sd['idx'].numpy(), sd['alpha'], sd['phi']
    if 'mod.0.weight' in sd:                                       # DLT-dyn: mean adapted coefficients on the test set
        from src.data_provider.data_loader import Dataset_PEMS
        import types
        from src.model.DLT import Model
        C, k = idx.shape
        cfg = types.SimpleNamespace(task_name='long_term_forecast', seq_len=96, pred_len=H, enc_in=C,
                                    dlt_variant='dltdyn', dlt_k=k, dlt_L=phi.shape[1] - 1, dlt_R=phi.shape[0],
                                    dlt_dyn_dim=sd['summ.weight'].shape[0], dlt_impl='basis', dlt_gate='none')
        m = Model(cfg); m.load_state_dict(sd, strict=False); m.eval()
        ds = Dataset_PEMS(types.SimpleNamespace(), data_dir, flag='test', size=[96, 0, H], features='M',
                          data_path='PEMS08.npz', timeenc=1, freq='t')
        X = torch.tensor(np.stack([ds.data_x[s:s + 96] for s in range(0, len(ds), 4)]), dtype=torch.float32)
        acc = torch.zeros_like(alpha)
        with torch.no_grad():
            for b in range(0, len(X), 256):
                x = X[b:b + 256].permute(0, 2, 1)
                xn = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)
                acc += (alpha[None] * (1 + torch.tanh(m.mod(m._pair_features(xn))))).sum(0)
        alpha = acc / len(X)
    K = torch.einsum('ckr,rl->ckl', alpha, phi).numpy()           # (C, k, L+1)
    return idx, sd['lag'].numpy(), K


def stats(idx, lag, K, hops, dist, down):
    C, k, L1 = K.shape
    tau = np.arange(L1)
    absK = np.abs(K)
    centroid = (absK * tau).sum(-1) / absK.sum(-1)
    peak = absK.argmax(-1)
    tgt = np.repeat(np.arange(C)[:, None], k, 1)
    h = hops[idx, tgt]                                             # hops between source j = idx[i, m] and target i
    d = dist[idx, tgt] / 1000.0
    up = np.isfinite(down[idx, tgt])                               # source upstream of the target
    dn = np.isfinite(down[tgt, idx])
    off = ~np.eye(C, dtype=bool)
    ok = np.isfinite(d)
    r_c, p_c = spearmanr(d[ok], centroid[ok])
    r_p, p_p = spearmanr(d[ok], peak[ok])
    near = ok & (h <= 3)
    early = absK[..., :3].sum(-1) / absK.sum(-1)                   # share of |K| on lags 0-2 (10 minutes)
    up_only, dn_only = ok & up & ~dn, ok & dn & ~up

    def rho(mask, y):
        return [float(x) for x in spearmanr(d[mask], y[mask])] if mask.sum() > 10 else None
    return {
        'pairs': int(C * k),
        'hop_share_selected': {f'{n}': float(np.mean(h == n)) for n in (1, 2, 3)} | {'>3 or unreachable': float(np.mean(~(h <= 3)))},
        'hop_share_all_pairs': {f'{n}': float(np.mean(hops[off] == n)) for n in (1, 2, 3)},
        'upstream_share': float(up.mean()), 'downstream_share': float(dn.mean()),
        'centroid_mean_by_hop': {f'{n}': float(centroid[h == n].mean()) for n in (1, 2, 3) if (h == n).any()},
        'centroid_mean_far': float(centroid[~(h <= 3)].mean()) if (~(h <= 3)).any() else None,
        'centroid_upstream_near': float(centroid[near & up].mean()) if (near & up).any() else None,
        'centroid_downstream_near': float(centroid[near & dn & ~up].mean()) if (near & dn & ~up).any() else None,
        'spearman_distance_centroid': [float(r_c), float(p_c)], 'spearman_distance_peak': [float(r_p), float(p_p)],
        'spearman_ccf_lag_centroid': [float(x) for x in spearmanr(lag.ravel(), centroid.ravel())],
        'mass_beyond_lag0': float((absK[..., 1:].sum(-1) / absK.sum(-1)).mean()),
        'early_mass_lag0_2': {'all': float(early.mean()), 'within_3_hops': float(early[near].mean()) if near.any() else None,
                              'beyond_3_hops': float(early[~near].mean()) if (~near).any() else None},
        'upstream_only_pairs': int(up_only.sum()), 'downstream_only_pairs': int(dn_only.sum()),
        'spearman_distance_peak_upstream_only': rho(up_only, peak),
        'spearman_distance_peak_downstream_only': rho(dn_only, peak),
        'peak_lag_mean_upstream_only': float(peak[up_only].mean()) if up_only.any() else None,
        'peak_lag_mean_downstream_only': float(peak[dn_only].mean()) if dn_only.any() else None,
        'share_multi_lag': float(np.mean((absK >= 0.5 * absK.max(-1, keepdims=True)).sum(-1) >= 2)),
    }, centroid, h, d


def figure(path_pdf, examples, idx, K, h, d, centroid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, grid = '#0b0b0b', '#52514e', '#e4e3df'
    col = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}               # categorical slots 1-3 (validated, all pairs)
    style = {1: '-', 2: '--', 3: ':'}
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': ink2, 'axes.labelcolor': ink, 'xtick.color': ink2,
                         'ytick.color': ink2, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})           # embedded TrueType, no Type 3 fonts
    fig, axes = plt.subplots(1, len(examples) + 1, figsize=(7.2, 2.2), gridspec_kw={'width_ratios': [1] * len(examples) + [1.3]})
    tau = np.arange(K.shape[-1])
    for ax, i in zip(axes, examples):
        for m in range(K.shape[1]):
            hh = int(min(h[i, m], 3)) if np.isfinite(h[i, m]) else 3
            ax.plot(tau, K[i, m], color=col[hh], ls=style[hh], lw=1.4)
        ax.axhline(0, color=grid, lw=0.8, zorder=0)
        ax.set_title(f'target sensor {i}', fontsize=8, color=ink)
        ax.set_xlabel('lag τ (5-min steps)')
    axes[0].set_ylabel('kernel $K_{ij}(\\tau)$')
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=col[n], ls=style[n], lw=1.4) for n in (1, 2, 3)],
               ['source 1 hop from the target', '2 hops', '≥ 3 hops or not connected'], frameon=False, fontsize=7,
               loc='upper center', ncol=3, bbox_to_anchor=(0.4, 1.02))
    ax = axes[-1]
    ok = np.isfinite(d)
    ax.scatter(d[ok], centroid[ok], s=6, color='#8c8c8c', alpha=0.45, linewidths=0)
    bins = np.quantile(d[ok], np.linspace(0, 1, 9))
    mid = [(bins[b] + bins[b + 1]) / 2 for b in range(8)]
    med = [np.median(centroid[ok][(d[ok] >= bins[b]) & (d[ok] <= bins[b + 1])]) for b in range(8)]
    ax.plot(mid, med, color='#2a78d6', lw=2, marker='o', ms=4)
    ax.set_xlabel('road-graph distance (edge cost / 1000)')
    ax.set_ylabel('kernel lag centroid (steps)')
    ax.grid(axis='y', color=grid, lw=0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path_pdf); fig.savefig(path_pdf.replace('.pdf', '.png'), dpi=200)


def p0_checkpoints(variant, H):
    """Checkpoints of the ungated P0 runs of a variant on PEMS08 (the latest per seed), then those before the refactor."""
    latest = {}
    for q in find_checkpoints(str(paths.CHECKPOINTS), 'PEMS08', model='DLT', pred_len=H, des=f'PEMS08_{H}_{variant}_none',
                              dlt_k=8, dlt_L=48, dlt_R=4):
        latest[os.path.basename(q).rsplit('_', 1)[0]] = q                   # <prefix>_<seq>.pth: newest last
    legacy = sorted(glob.glob(str(LEGACY_CKPT / f'*_P0_PEMS08_{H}_DLT_*_pl{H}_*_{variant}_k8_L48_R4*' / 'checkpoint.pth')))
    # the P0 grid only: other suites (e.g. long_extra) also train DLT-dyn on PEMS08
    legacy = [q for q in legacy if '_g' not in q.split(f'_{variant}_k8_L48_R4')[1]]   # ungated models only
    return sorted(latest.values()) + legacy


def label(q):
    return os.path.basename(os.path.dirname(q)) if q.endswith('checkpoint.pth') else os.path.basename(q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--H', type=int, default=96)
    p.add_argument('--fig_dir', default='', help='directory of the figure (default: ../DLTpaper/figures if present)')
    a = p.parse_args()
    a.out = str(paths.OUTPUT / 'mechanism')
    a.data = str(paths.root_of('PEMS08.npz'))
    paper = next((d / 'figures' for d in REPO.parent.iterdir()                 # the paper repository next to this one
                  if d.name.lower() == 'dltpaper' and (d / 'figures').is_dir()), None)
    a.fig_dir = os.path.abspath(a.fig_dir) if a.fig_dir else (str(paper) if paper else a.out)
    os.makedirs(a.fig_dir, exist_ok=True)
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    C = np.load(os.path.join(a.data, 'PEMS08.npz'))['data'].shape[1]
    hops, dist, down = road_graph(str(paths.path_of('PEMS08_edges.csv')), C)
    res = {}
    for variant in ('dlt', 'dltdyn'):
        ckpts = p0_checkpoints(variant, a.H)
        per_seed = []
        for q in ckpts:
            idx, lag, K = kernels_from_ckpt(q, a.data, a.H)
            s, centroid, h, d = stats(idx, lag, K, hops, dist, down)
            per_seed.append(s)
            if variant == 'dlt' and len(per_seed) == 1:
                order = np.argsort(-(np.abs(K).sum(-1).sum(-1)))       # targets with the strongest cross input
                figure(os.path.join(a.fig_dir, 'fig_kernels_pems08.pdf'), [int(order[0]), int(order[1])], idx, K, h, d, centroid)
        res[variant] = {'checkpoints': [label(q) for q in ckpts], 'per_seed': per_seed}
        print(variant, json.dumps(per_seed[0] if per_seed else {}, indent=1)[:1500])
    json.dump(res, open(os.path.join(a.out, 'kernels_pems08.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
