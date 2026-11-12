# Lag premise check on the PEMS08 road edges:  python code/pems_lag.py
# (PEMS08.npz and PEMS08_edges.csv from the fixed data directory, scripts/paths.py; or pass the two files)
import os, sys, numpy as np, csv
if len(sys.argv) < 3:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    import paths
    sys.argv[1:] = [str(paths.path_of("PEMS08.npz")), str(paths.path_of("PEMS08_edges.csv"))]
d = np.load(sys.argv[1], allow_pickle=False)["data"][:, :, 0].astype(float)
n = int(0.6 * len(d)); x = np.diff(d[:n], axis=0); x = (x - x.mean(0)) / (x.std(0) + 1e-8)
edges = [(int(r["from"]), int(r["to"]), float(r["cost"])) for r in csv.DictReader(open(sys.argv[2]))]
L = 12; T = len(x)
def ccf(i, j):   # corr(x_i(t), x_j(t - tau)), tau in -L..L ; tau>0 => j leads i
    return np.array([np.mean(x[max(tau,0):T+min(tau,0), i] * x[max(-tau,0):T-max(tau,0), j]) for tau in range(-L, L + 1)])
lags, dist, peak, nsig = [], [], [], []
thr = 2 / np.sqrt(T)
for a, b, c in edges:            # a -> b : upstream a should lead downstream b (lag > 0)
    r = ccf(b, a)
    t = int(np.argmax(np.abs(r))) - L
    lags.append(t); dist.append(c); peak.append(np.abs(r).max())
    pos = np.abs(r[L:]) ; nsig.append(int((pos > max(thr, 0.5 * pos.max())).sum()))
lags, dist, nsig = np.array(lags), np.array(dist), np.array(nsig)
print(f"edges={len(edges)}  5-min steps; train len={T}")
print(f"peak lag distribution (upstream->downstream, +: upstream leads): " + ", ".join(f"{v}:{(lags==v).sum()}" for v in range(-3, 6)))
print(f"share lag>0: {(lags>0).mean():.2f}, lag=0: {(lags==0).mean():.2f}, lag<0: {(lags<0).mean():.2f}")
print(f"Spearman(dist, lag) = {np.corrcoef(np.argsort(np.argsort(dist)), np.argsort(np.argsort(lags)))[0,1]:.2f}")
for lo, hi in [(0, 1000), (1000, 3000), (3000, 1e9)]:
    m = (dist >= lo) & (dist < hi)
    if m.sum(): print(f"dist [{lo:.0f},{hi:.0f}) n={m.sum():3d}  mean lag={lags[m].mean():.2f}  share lag>0={(lags[m]>0).mean():.2f}")
print(f"#non-negative lags with |ccf| >= 50% of peak (spread): median={np.median(nsig):.0f}, share>=2: {(nsig>=2).mean():.2f}")
