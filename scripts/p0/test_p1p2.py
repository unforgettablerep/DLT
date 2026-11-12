"""Verification of the P1 / P2 additions (run from anywhere; float64):

* --dlt_impl dense == the reference loop for dlt / dltdyn (also with all neighbours)
* --dlt_neighbours random / all: valid neighbour sets (no self, distinct, reproducible per seed)
* DLTPlug == backbone + sigma * W_c z; with --dlt_variant ci it is the backbone alone
* LagTS lag relation loss == an explicit loop over pairs and steps (Eq. 7-9)
* S-Mamba: the pure-PyTorch Mamba block == the independent block of models/MambaSimple.py (same parameters), is
  causal and initialised like mamba_ssm; the model is bidirectional over the variates and drops the mark tokens
* every representative baseline has a configuration for every long-term dataset and horizon
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

# ---- S-Mamba
import math
from layers.MambaPy import Mamba
from models import SMamba
from models.MambaSimple import MambaBlock
torch.manual_seed(5)
worst = 0.0
for d_model, d_state, d_conv, expand, Lq in ((32, 16, 2, 1, 11), (48, 2, 2, 1, 25), (64, 32, 4, 2, 9)):
    blk = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    ref = MambaBlock(types.SimpleNamespace(d_model=d_model, d_ff=d_state, d_conv=d_conv, expand=expand),
                     d_inner=d_model * expand, dt_rank=math.ceil(d_model / 16))
    ref.load_state_dict(blk.state_dict(), strict=True)          # same parameter names and shapes
    hq = torch.randn(3, Lq, d_model)
    with torch.no_grad():
        worst = max(worst, (blk(hq) - ref(hq)).abs().max().item())
ok(f'Mamba block (PyTorch) == MambaSimple block with the same parameters (max err {worst:.1e})', worst < 1e-10)
blk = Mamba(32, d_state=8, d_conv=2, expand=1)
hq = torch.randn(2, 12, 32); h2 = hq.clone(); h2[:, 7:] += 1.0
with torch.no_grad():
    moved = (blk(hq) - blk(h2)).abs().amax(-1)[0].nonzero().flatten().tolist()
ok('Mamba block is causal (a change at token 7 moves outputs 7.. only)', moved == list(range(7, 12)))
sp = torch.nn.functional.softplus(blk.dt_proj.bias)
ok('Mamba block initialisation as in mamba_ssm (dt in [1e-3, 1e-1], A = -(1..d_state), D = 1)',
   bool(sp.min() >= 1e-3 - 1e-9 and sp.max() <= 0.1 + 1e-9) and torch.allclose(-torch.exp(blk.A_log[0]).double(), -torch.arange(1., 9.).double(), atol=1e-6)
   and bool((blk.D == 1).all()))
c = cfg(enc_in=6, dec_in=6, c_out=6, d_model=32, d_ff=32, e_layers=2, d_state=4, mamba_impl='torch', use_norm=1)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    sm = SMamba.Model(c).eval()
xq, mq = torch.randn(2, 96, 6), torch.randn(2, 96, 4)
with torch.no_grad():
    yq = sm(xq, mq, None, None)
    x3 = xq.clone(); x3[:, :, 3] += torch.randn(2, 96)          # change variate 3 only
    dy = (sm(x3, mq, None, None) - yq).abs().amax((0, 1))
ok('S-Mamba forward shape (B, H, N) without the mark tokens', tuple(yq.shape) == (2, 24, 6))
ok('S-Mamba is bidirectional over the variates (variate 3 reaches variates before and after it)', bool((dy > 1e-9).all()))
with torch.no_grad():
    ys = sm(xq * 3.0 + 5.0, mq, None, None)                     # instance normalisation: scale and shift pass through
err = (ys - (yq * 3.0 + 5.0)).abs().max().item()               # exact up to the 1e-5 inside sqrt(var + 1e-5)
ok(f'S-Mamba instance normalisation (output follows scale and shift of the input, max err {err:.1e})', err < 1e-4)

# ---- configurations of the representative baselines
sys.path.insert(0, str(REPO / 'scripts'))
import exps
missing = [(m, ds, H) for m in exps.BASELINES for ds in exps.LONG_SETS for H in exps.HORIZONS
           if exps.backbone_args(m, ds, H)[1] == 'fallback' and m != 'DLinear' and not exps.FALLBACK.get(m, {}).get(ds)]
ok(f'official or documented configuration for every baseline, dataset and horizon ({len(exps.BASELINES)} models)', not missing)
print('ALL PASS' if ok_all else 'SOME CHECKS FAILED')
