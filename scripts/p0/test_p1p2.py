"""Verification of the P1 / P2 additions (run from anywhere; float64):

* --dlt_impl dense == the reference loop for dlt / dltdyn (also with all neighbours)
* --dlt_neighbours random / all: valid neighbour sets (no self, distinct, reproducible per seed)
* DLTPlug == backbone + sigma * W_c z; with --dlt_variant ci it is the backbone alone
* LagTS lag relation loss == an explicit loop over pairs and steps (Eq. 7-9)
"""
import sys, types
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'p0'))
from models import DLT, DLTPlug, LagTS, DLinear
torch.set_default_dtype(torch.float64)
ok_all = True
def ok(name, cond):
    global ok_all
    print(('PASS ' if cond else 'FAIL ') + name); ok_all &= bool(cond)

def cfg(**kw):
    c = dict(task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=24, enc_in=9, dec_in=9, c_out=9,
             dlt_variant='dltdyn', dlt_k=4, dlt_L=10, dlt_R=3, dlt_dyn_dim=8, dlt_ccf_len=8000, dlt_impl='kernel',
             dlt_topk=3, dlt_gate='none', dlt_neighbours='ccf', fix_seed=2021, moving_avg=25, individual=False,
             d_model=16, d_ff=16, n_heads=2, e_layers=1, factor=1, dropout=0.0, activation='gelu', embed='timeF',
             freq='h', lagts_rel_dim=3, lagts_lambda=0.5, plug_base='DLinear')
    c.update(kw); return types.SimpleNamespace(**c)

def reference_z(m, xn):
    """explicit z_i(t) = sum_j sum_tau K_ij(tau) x_j(t - tau) with K_ij = sum_r alpha_ijr phi_r"""
    B, C, T = xn.shape
    al = m.alpha[None].expand(B, -1, -1, -1)
    if m.variant == 'dltdyn':
        al = m.alpha[None] * (1 + torch.tanh(m.mod(m._pair_features(xn))))
    K = torch.einsum('bckr,rl->bckl', al, m.phi)
    z = torch.zeros(B, C, T)
    for i in range(C):
        for jj in range(m.k):
            j = m.idx[i, jj]
            for tau in range(m.L + 1):
                z[:, i, tau:] += K[:, i, jj, tau, None] * xn[:, j, :T - tau]
    return z

torch.manual_seed(0)
Xtr = torch.randn(3000, 9).cumsum(0).numpy()
xn = torch.randn(3, 9, 96)
for nb in ('ccf', 'all'):
    for v in ('dlt', 'dltdyn'):
        torch.manual_seed(1); m = DLT.Model(cfg(dlt_variant=v, dlt_impl='dense', dlt_neighbours=nb)); m.prepare(Xtr)
        with torch.no_grad():
            for p in m.parameters(): p.add_(0.3 * torch.randn_like(p))
            err = (m.cross(xn) - reference_z(m, xn)).abs().max().item()
        ok(f'dense path == reference loop ({v}, neighbours={nb}, k={m.k}; max err {err:.1e})', err < 1e-10)

# neighbour sets
torch.manual_seed(2)
for nb in ('random', 'all'):
    for seed in (2021, 2022):
        m = DLT.Model(cfg(dlt_variant='dlt', dlt_neighbours=nb, fix_seed=seed)); m.prepare(Xtr)
        idx = m.idx
        valid = all(i not in idx[i].tolist() and len(set(idx[i].tolist())) == m.k for i in range(9))
        m2 = DLT.Model(cfg(dlt_variant='dlt', dlt_neighbours=nb, fix_seed=seed)); m2.prepare(Xtr)
        ok(f'neighbours={nb} seed={seed}: k={m.k}, no self, distinct, reproducible', valid and torch.equal(idx, m2.idx))
r1 = DLT.Model(cfg(dlt_variant='dlt', dlt_neighbours='random', fix_seed=2021)); r1.prepare(Xtr)
r2 = DLT.Model(cfg(dlt_variant='dlt', dlt_neighbours='random', fix_seed=2022)); r2.prepare(Xtr)
c1 = DLT.Model(cfg(dlt_variant='dlt')); c1.prepare(Xtr)
ok('random sets differ between seeds and from the CCF top-k', not torch.equal(r1.idx, r2.idx) and not torch.equal(r1.idx, c1.idx))
_, blag = None, None
ok('random neighbours keep the CCF peak lag of each pair',
   all(int(r1.lag[i, j]) == int(DLT.select_neighbours(Xtr, 8, 10, mode='all')[1][i][
       DLT.select_neighbours(Xtr, 8, 10, mode='all')[0][i].tolist().index(int(r1.idx[i, j]))])
       for i in range(9) for j in range(4)))

# DLTPlug
x_enc = torch.randn(4, 96, 9); mark = torch.randn(4, 96, 4)
torch.manual_seed(3); plug = DLTPlug.Model(cfg(dlt_variant='dltdyn')); plug.prepare(Xtr)
with torch.no_grad():
    for p in plug.dlt.parameters(): p.add_(0.3 * torch.randn_like(p))
    x = x_enc.permute(0, 2, 1); mu = x.mean(-1, keepdim=True); sd = x.std(-1, keepdim=True) + 1e-5
    ref = plug.base(x_enc, mark, None, None) + (plug.dlt.cross_head(plug.dlt.cross((x - mu) / sd)) * sd).permute(0, 2, 1)
    ok('DLTPlug = backbone + sigma * W_c z', torch.allclose(plug(x_enc, mark, None, None), ref, atol=1e-12))
torch.manual_seed(4); plug_ci = DLTPlug.Model(cfg(dlt_variant='ci'))
torch.manual_seed(4); base = DLinear.Model(cfg())
ok('DLTPlug with variant ci == the backbone alone (same init, same output)',
   torch.equal(plug_ci(x_enc, mark, None, None), base(x_enc, mark, None, None)))
ok('DLTPlug has no intra-variable DLT head', not hasattr(plug.dlt, 'self_head'))

# LagTS loss against an explicit loop
torch.manual_seed(5); lt = LagTS.Model(cfg(enc_in=4, pred_len=12, seq_len=16))
xe = torch.randn(2, 16, 4); pr = torch.randn(2, 12, 4); tr = torch.randn(2, 12, 4)
loss = lt.extra_loss(xe, pr, tr).item()
mu = xe.mean(1, keepdim=True); sd = xe.std(1, keepdim=True) + 1e-5
xh, yh, yt = (xe - mu) / sd, (pr - mu) / sd, (tr - mu) / sd
S, T, N = 12, 16, 4
acc, cnt = 0.0, 0
for b in range(2):
    for i in range(N):
        for j in range(N):
            R = [sum(yt[b, (t + tau) % S, j] * yt[b, t, i] for t in range(S)) / S for tau in range(S)]   # circular CCF
            cand = R[1:] if i != j else R[1:S // 2 + 1]       # i = j: symmetric, smaller lag of each tie
            d = min(int(np.argmax(cand)) + 1, T)
            z = torch.cat([xh[b, :, i], yh[b, :, i]])
            for t in range(S):
                acc += float((z[T + t - d] - yh[b, t, j]) ** 2); cnt += 1
ref = 0.5 * acc / cnt
ok(f'LagTS lag relation loss == explicit loop ({loss:.6f} vs {ref:.6f})', abs(loss - ref) < 1e-9)
y = lt(xe, None, None, None)
ok('LagTS forward shape (B, S, N)', tuple(y.shape) == (2, 12, 4))
print('ALL PASS' if ok_all else 'SOME CHECKS FAILED')
