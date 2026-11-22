"""Feasibility prototype: does a per-pair *distributed* lag kernel help over
scalar mixing / single-lag shift / short (3-tap) kernels?

All cross-variable variants share: RevIN, the same top-k neighbour sets, the
same self linear head and the same cross linear head. They differ ONLY in how
the per-pair kernel K[i, j, tau] (tau = 0..L) is parameterised:

  ci      : no cross branch (RLinear-style channel-independent baseline)
  scalar  : K = a_ij * [tau == 0]                  (iTransformer/VCformer-like mixing)
  shift   : K = a_ij * [tau == d_ij], d_ij from CCF (LIFT-like single lag)
  tap3    : K free for tau < 3                     (CrossLinear-like short kernel)
  dlt     : K = sum_r alpha_ijr * phi_r(tau)       (proposed: shared lag basis)
  full    : K free for tau <= L                    (unconstrained upper bound)
  scalardyn : scalar + input-adaptive a_ij         (control: dynamics without a lag kernel)
  dltdyn  : dlt + input-adaptive alpha             (proposed dynamic version)

z_i(t) = sum_{j in N(i)} sum_tau K[i,j,tau] * x_j(t - tau)
y_hat_i = W_self x_i + W_cross z_i

Example:
  python code/proto.py --data PEMS08 --pred_len 96 --variant dltdyn --out results/runs.jsonl
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- data
def load(name, root, seq_len):
    """root: a directory with the files, or None for the fixed data directory of the repository (scripts/paths.py)"""
    if root is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        import paths
        locate = lambda f: str(paths.path_of(f))
    else:
        locate = lambda f: os.path.join(root, f)
    if name == "PEMS08":
        d = np.load(locate("PEMS08.npz"), allow_pickle=False)["data"][:, :, 0].astype(np.float64)
        n = len(d)
        b1 = [0, int(0.6 * n) - seq_len, int(0.8 * n) - seq_len]
        b2 = [int(0.6 * n), int(0.8 * n), n]
    else:
        import pandas as pd
        df = pd.read_csv(locate(f"{name}.csv"))
        d = df.iloc[:, 1:].values.astype(np.float64)
        n = len(d)
        if name.startswith("ETTh"):
            m = 30 * 24
            b1 = [0, 12 * m - seq_len, 16 * m - seq_len]
            b2 = [12 * m, 16 * m, 20 * m]
        else:
            ntr, nte = int(n * 0.7), int(n * 0.2)
            nva = n - ntr - nte
            b1 = [0, ntr - seq_len, n - nte - seq_len]
            b2 = [ntr, ntr + nva, n]
    tr = d[b1[0]:b2[0]]
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    d = (d - mu) / sd
    return [torch.tensor(d[b1[k]:b2[k]], dtype=torch.float32) for k in range(3)]


class Windows(torch.utils.data.Dataset):
    def __init__(self, x, seq_len, pred_len):
        self.x, self.s, self.p = x, seq_len, pred_len

    def __len__(self):
        return len(self.x) - self.s - self.p + 1

    def __getitem__(self, i):
        return self.x[i:i + self.s].T, self.x[i + self.s:i + self.s + self.p].T  # (C,T),(C,H)


# ------------------------------------------------- neighbours from train CCF
def neighbours(train, k, L, max_len=8000):
    """top-k sources per target by max |corr(dx_i(t), dx_j(t-tau))|, tau in 0..L,
    computed on first differences of the training split. Returns idx (C,k), lag (C,k)."""
    x = train.numpy()
    x = np.diff(x, axis=0)[-max_len:]
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    n, C = x.shape
    best = np.full((C, C), -1.0)
    blag = np.zeros((C, C), dtype=np.int64)
    for tau in range(L + 1):
        a = x[tau:]            # target at t
        b = x[:n - tau]        # source at t - tau
        c = np.abs(a.T @ b) / (n - tau)  # (C_target, C_source)
        upd = c > best
        best[upd] = c[upd]
        blag[upd] = tau
    np.fill_diagonal(best, -np.inf)
    idx = np.argsort(-best, axis=1)[:, :k]
    lag = np.take_along_axis(blag, idx, axis=1)
    return torch.tensor(idx), torch.tensor(lag), np.take_along_axis(best, idx, axis=1)


# ---------------------------------------------------------------- model
class Model(nn.Module):
    def __init__(self, variant, C, T, H, idx, lag, L, R):
        super().__init__()
        self.v, self.C, self.T, self.L, self.R = variant, C, T, L, R
        self.self_head = nn.Linear(T, H)
        if variant == "ci":
            return
        self.register_buffer("idx", idx)
        self.register_buffer("lag", lag)
        k = idx.shape[1]
        self.k = k
        self.cross_head = nn.Linear(T, H)
        g = lambda *s: nn.Parameter(0.01 * torch.randn(*s))
        if variant in ("scalar", "shift"):
            self.a = g(C, k)
        elif variant == "scalardyn":
            # control: same input-adaptive modulation as dltdyn, but NO lag kernel (tau = 0 only)
            self.a = g(C, k)
            self.summ = nn.Linear(T, 16)
            self.mod = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.mod[-1].weight); nn.init.zeros_(self.mod[-1].bias)
        elif variant == "tap3":
            self.w = g(C, k, 3)
        elif variant == "full":
            self.w = g(C, k, L + 1)
        elif variant in ("dlt", "dltdyn"):
            self.alpha = g(C, k, R)
            tau = torch.arange(L + 1).float()
            # init basis: decaying exponentials with different delays/scales (+ noise)
            delays = torch.linspace(0, L * 0.6, R)
            scales = torch.logspace(0, np.log10(max(L / 4, 2)), R)
            phi = torch.exp(-F.relu(tau[None] - delays[:, None]) / scales[:, None]) * (tau[None] >= delays[:, None])
            phi = phi / phi.norm(dim=1, keepdim=True)
            self.phi = nn.Parameter(phi + 0.01 * torch.randn_like(phi))
            if variant == "dltdyn":
                # alpha_ij(x) = alpha_ij * (1 + tanh(MLP([s_i, s_j])))  ; s = summary of window
                self.summ = nn.Linear(T, 16)
                self.mod = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, R))
                nn.init.zeros_(self.mod[-1].weight); nn.init.zeros_(self.mod[-1].bias)

    def kernel(self, x=None):
        C, k, L = self.C, self.k, self.L
        K = torch.zeros(C, k, L + 1, device=self.self_head.weight.device)
        if self.v == "scalar":
            K = K.clone(); K[:, :, 0] = self.a
        elif self.v == "shift":
            K = K.scatter(2, self.lag.unsqueeze(-1), self.a.unsqueeze(-1))
        elif self.v == "tap3":
            K = torch.cat([self.w, K[:, :, 3:]], dim=2)
        elif self.v == "full":
            K = self.w
        elif self.v == "dlt":
            K = torch.einsum("ikr,rl->ikl", self.alpha, self.phi)
        return K

    def forward(self, x):  # x: (B,C,T)
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True) + 1e-5
        xn = (x - mu) / sd
        y = self.self_head(xn)
        if self.v == "ci":
            return y * sd + mu
        B, C, T = xn.shape
        k, L = self.k, self.L
        z = torch.zeros_like(xn)
        if self.v in ("dlt", "dltdyn"):
            # filter every variable once with the R shared basis kernels: (B,C,R,T)
            xp = F.pad(xn, (L, 0)).reshape(B * C, 1, T + L)
            u = F.conv1d(xp, self.phi.flip(-1).unsqueeze(1)).reshape(B, C, self.R, T)
            if self.v == "dltdyn":
                s = self.summ(xn)                                            # (B,C,16)
                pair = torch.cat([s.unsqueeze(2).expand(B, C, k, 16), s[:, self.idx]], -1)
                alpha = self.alpha * (1 + torch.tanh(self.mod(pair)))        # (B,C,k,R)
            for slot in range(k):
                ug = u[:, self.idx[:, slot]]                                 # (B,C,R,T)
                if self.v == "dlt":
                    z = z + torch.einsum("bcrt,cr->bct", ug, self.alpha[:, slot])
                else:
                    z = z + torch.einsum("bcrt,bcr->bct", ug, alpha[:, :, slot])
        elif self.v == "scalar":
            for slot in range(k):
                z = z + xn[:, self.idx[:, slot]] * self.a[:, slot, None]
        elif self.v == "scalardyn":
            s = self.summ(xn)
            pair = torch.cat([s.unsqueeze(2).expand(B, C, k, 16), s[:, self.idx]], -1)
            a = self.a[None] * (1 + torch.tanh(self.mod(pair).squeeze(-1)))   # (B,C,k)
            for slot in range(k):
                z = z + xn[:, self.idx[:, slot]] * a[:, :, slot, None]
        elif self.v == "shift":
            xp = F.pad(xn, (L, 0))                                           # (B,C,T+L)
            t = torch.arange(T)
            for slot in range(k):
                src = xp[:, self.idx[:, slot]]                               # (B,C,T+L)
                pos = (t[None, :] + L - self.lag[:, slot, None]).expand(C, T)   # (C,T)
                z = z + torch.gather(src, 2, pos.unsqueeze(0).expand(B, C, T)) * self.a[:, slot, None]
        else:  # tap3 / full : explicit per-pair kernels via grouped conv
            K = self.kernel()                                                # (C,k,L+1)
            taps = 3 if self.v == "tap3" else L + 1
            K = K[:, :, :taps]
            xg = F.pad(xn[:, self.idx.reshape(-1), :], (taps - 1, 0))        # (B,C*k,T+taps-1)
            w = K.reshape(C * k, 1, taps).flip(-1)
            z = F.conv1d(xg, w, groups=C * k).reshape(B, C, k, T).sum(2)
        y = y + self.cross_head(z)
        return y * sd + mu


# ---------------------------------------------------------------- train
def evaluate(model, loader):
    model.eval(); se = ae = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            p = model(x)
            se += ((p - y) ** 2).sum().item(); ae += (p - y).abs().sum().item(); n += y.numel()
    return se / n, ae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--root", default=None, help="directory of the data files (default: the fixed dataset/ directory)")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)

    tr, va, te = load(a.data, a.root, a.seq_len)
    C = tr.shape[1]
    k = min(a.k, C - 1)
    idx, lag, strength = neighbours(tr, k, a.L)
    mk = lambda s, sh: torch.utils.data.DataLoader(Windows(s, a.seq_len, a.pred_len), batch_size=a.bs, shuffle=sh, drop_last=False)
    trl, val, tel = mk(tr, True), mk(va, False), mk(te, False)

    model = Model(a.variant, C, a.seq_len, a.pred_len, idx, lag, a.L, a.R)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, bad, best_state, t0 = 1e9, 0, None, time.time()
    for ep in range(a.epochs):
        model.train()
        for x, y in trl:
            opt.zero_grad(); loss = F.mse_loss(model(x), y); loss.backward(); opt.step()
        v = evaluate(model, val)[0]
        if v < best:
            best, bad, best_state = v, 0, {kk: vv.clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
        for g_ in opt.param_groups: g_["lr"] *= 0.5 ** (1 if ep >= 1 else 0)  # TSL 'type1' schedule
    model.load_state_dict(best_state)
    mse, mae = evaluate(model, tel)
    res = dict(data=a.data, variant=a.variant, H=a.pred_len, seed=a.seed, k=k, L=a.L, R=a.R,
               mse=mse, mae=mae, val=best, epochs=ep + 1, secs=round(time.time() - t0, 1),
               params=sum(p.numel() for p in model.parameters()),
               frac_lag0=float((lag == 0).float().mean()), med_lag=float(lag.float().median()),
               mean_strength=float(np.mean(strength)))
    print(json.dumps(res))
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
