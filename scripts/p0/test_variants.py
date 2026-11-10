"""Verification of the DLT kernels and gates in models/DLT.py.

    python scripts/p0/test_variants.py <dir with ETTh1.csv>

Checks (float64): every variant's cross() against an explicit loop over pairs and lags of
z_i(t) = sum_j sum_tau K_ij(tau) x_j(t - tau); the top-K lag sets against a reference CCF loop;
topk(K=1) == shift; the 8 original variants against the version of commit af398bd (same
parameters for the same seed, identical outputs); both gates against their formulas;
finite forward / backward and checkpoint round trips for every variant x gate.
"""
import sys, types, importlib.util, io, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models import DLT as new
old_src = subprocess.run(['git', '-C', str(REPO), 'show', 'af398bd:tslib/models/DLT.py'],
                         capture_output=True, text=True, check=True).stdout
tmp = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False); tmp.write(old_src); tmp.close()
spec = importlib.util.spec_from_file_location('DLT_old', tmp.name); old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old); os.unlink(tmp.name)
torch.set_default_dtype(torch.float64)          # exact comparisons
data = sys.argv[1]

def cfg(**kw):
    c = dict(task_name='long_term_forecast', seq_len=96, pred_len=24, enc_in=12, dlt_variant='dltdyn', dlt_k=4,
             dlt_L=10, dlt_R=3, dlt_dyn_dim=8, dlt_ccf_len=8000, dlt_impl='kernel', dlt_topk=3, dlt_gate='none')
    c.update(kw); return types.SimpleNamespace(**c)

ok = lambda name, cond: print(('PASS ' if cond else 'FAIL ') + name) or cond
allok = True

# ---- 1. neighbour selection is unchanged (real data)
ett = pd.read_csv(f'{data}/ETTh1.csv').values[:, 1:].astype(float)[:8545]
ett = (ett - ett.mean(0)) / ett.std(0)
i0, l0 = old.select_neighbours(ett, 6, 48)
i1, l1 = new.select_neighbours(ett, 6, 48)
allok &= ok('select_neighbours identical to previous version', torch.equal(i0, i1) and torch.equal(l0, l1))
tl = new.select_lags(ett, i1, 5, 48)
# reference: per pair, the 5 lags with largest |CCF|
x = np.diff(ett, axis=0)[-8000:]; x = (x - x.mean(0)) / x.std(0); n = len(x)
good = True
for i in range(7):
    for j in range(6):
        s = i1[i, j].item()
        r = np.array([abs((x[t:, i] * x[:n - t, s]).sum()) / (n - t) for t in range(49)])
        good &= sorted(np.argsort(-r)[:5].tolist()) == tl[i, j].tolist()
        good &= tl[i, j, :].tolist().count(l1[i, j].item()) == 1          # the peak lag is in the set
allok &= ok('select_lags = top-5 |CCF| lags per pair (reference loop), contains the peak lag', good)

# ---- 2. reference operator: z_i(t) = sum_j sum_tau K_ij(tau) x_j(t - tau)
def dense_kernel(m, xn):
    """per-sample dense kernels (B, C, k, L+1) for every variant, built independently of cross()"""
    B = xn.shape[0]; C, k, L = m.C, m.k, m.L
    K = torch.zeros(B, C, k, L + 1)
    v = m.variant
    if v == 'scalar': K[..., 0] = m.a
    elif v == 'scalardyn':
        K[..., 0] = m.a[None] * (1 + torch.tanh(m.mod(m._pair_features(xn)).squeeze(-1)))
    elif v == 'shift':
        for i in range(C):
            for j in range(k): K[:, i, j, m.lag[i, j]] = m.a[i, j]
    elif v in ('tap3', 'full'): K[..., :m.w.shape[-1]] = m.w
    elif v in ('dlt', 'dltdyn'):
        al = m.alpha[None].expand(B, -1, -1, -1)
        if v == 'dltdyn': al = m.alpha[None] * (1 + torch.tanh(m.mod(m._pair_features(xn))))
        K = torch.einsum('bckr,rl->bckl', al, m.phi)
    elif v in ('topk', 'mill', 'milldyn'):
        if v == 'topk': w = m.w[None].expand(B, -1, -1, -1)
        else:
            # Lambda written out from MillGNN Eq. (9) with explicit loops
            w = torch.zeros(B, C, k, m.K)
            for b in range(B):
                if v == 'milldyn':
                    s = m.summ(xn[b]); q = m.q_proj(s); kk = m.k_proj(s)
                for i in range(C):
                    if v == 'milldyn':
                        sc = torch.stack([q[i] @ kk[m.idx[i, j]] for j in range(k)]) / np.sqrt(q.shape[-1])
                        att = torch.softmax(sc, 0)
                    for j in range(k):
                        a = torch.sigmoid(m.b_att[i, j]) if v == 'mill' else att[j]
                        be, bi = torch.sigmoid(m.b_exc[i, j]), torch.sigmoid(m.b_inh[i, j])
                        for mm in range(m.K):
                            t = m.tlag[i, j, mm].double()
                            w[b, i, j, mm] = m.a[i, j] * torch.relu(torch.exp(-be * t) - (1 - a) * torch.exp(-bi * t))
        for i in range(C):
            for j in range(k):
                for mm in range(m.K): K[:, i, j, m.tlag[i, j, mm]] += w[:, i, j, mm]
    return K

def reference_z(m, xn):
    K = dense_kernel(m, xn); B, C, T = xn.shape
    z = torch.zeros(B, C, T)
    for i in range(C):
        for jj in range(m.k):
            j = m.idx[i, jj]
            for tau in range(m.L + 1):
                if K[:, i, jj, tau].abs().max() == 0: continue
                z[:, i, tau:] += K[:, i, jj, tau, None] * xn[:, j, :T - tau]
    return z

torch.manual_seed(0)
C = 12; Xtr = torch.randn(3000, C).cumsum(0).numpy()
xn = torch.randn(3, C, 96)
for v in new.VARIANTS[1:]:
    for impl in (('kernel', 'basis') if v in ('dlt', 'dltdyn') else ('kernel',)):
        torch.manual_seed(1); m = new.Model(cfg(dlt_variant=v, dlt_impl=impl)); m.prepare(Xtr)
        with torch.no_grad():
            for p in m.parameters(): p.add_(0.3 * torch.randn_like(p))   # move away from init (dyn layers non-zero)
            err = (m.cross(xn) - reference_z(m, xn)).abs().max().item()
        allok &= ok(f'{v:9s} [{impl}] cross() == reference loop (max err {err:.1e})', err < 1e-10)

# ---- 3. topk with K = 1 is the single-lag shift; old variants unchanged vs previous version
torch.manual_seed(2); ms = new.Model(cfg(dlt_variant='shift')); ms.prepare(Xtr)
mt = new.Model(cfg(dlt_variant='topk', dlt_topk=1)); mt.prepare(Xtr)
with torch.no_grad(): mt.w.copy_(ms.a[..., None])
allok &= ok('topk(K=1) == shift (same weights, top-1 lag == peak lag)',
            torch.equal(mt.tlag[..., 0], ms.lag) and torch.allclose(mt.cross(xn), ms.cross(xn), atol=1e-12))
xin = torch.randn(4, 96, C)
for v in old.VARIANTS:
    torch.manual_seed(3); mo = old.Model(cfg(dlt_variant=v)); mo.prepare(Xtr)
    torch.manual_seed(3); mn = new.Model(cfg(dlt_variant=v)); mn.prepare(Xtr)
    same_init = all(torch.equal(a, b) for a, b in zip(mo.state_dict().values(), mn.state_dict().values())) \
        and list(mo.state_dict()) == list(mn.state_dict())
    allok &= ok(f'{v:9s} same parameters for the same seed and identical forward as before',
                same_init and torch.equal(mo(xin, None, None, None), mn(xin, None, None, None)))

# ---- 4. gates
torch.manual_seed(4); base = new.Model(cfg(dlt_variant='dltdyn')); base.prepare(Xtr)
for gate in ('moe', 'static'):
    torch.manual_seed(4); mg = new.Model(cfg(dlt_variant='dltdyn', dlt_gate=gate, dlt_gate_init=-3.0)); mg.prepare(Xtr)
    with torch.no_grad():
        x = xin.permute(0, 2, 1); mu = x.mean(-1, keepdim=True); sd = x.std(-1, keepdim=True) + 1e-5; xx = (x - mu) / sd
        ys, yc = mg.self_head(xx), mg.cross_head(mg.cross(xx))
        if gate == 'moe':
            gw = torch.softmax(xx @ mg.gate_net.weight.T + mg.gate_net.bias, -1); y = gw[..., :1] * ys + gw[..., 1:] * yc
        else:
            y = ys + 1 / (1 + np.exp(3.0)) * yc
        ref = (y * sd + mu).permute(0, 2, 1)
    # same seed -> all non-gate parameters identical to the ungated model (gate params are created last)
    shared = all(torch.equal(base.state_dict()[k], v) for k, v in mg.state_dict().items() if k in base.state_dict())
    allok &= ok(f'gate {gate:6s} forward matches formula; non-gate init unchanged',
                shared and torch.allclose(mg(xin, None, None, None), ref, atol=1e-12))

# ---- 5. every variant x gate: forward / backward finite, checkpoint round trip keeps tlag
for v in new.VARIANTS:
    for gate in (('none',) if v == 'ci' else new.GATES):
        torch.manual_seed(5); m = new.Model(cfg(dlt_variant=v, dlt_gate=gate)); m.prepare(Xtr)
        y = m(xin, None, None, None); y.pow(2).mean().backward()
        fin = torch.isfinite(y).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        buf = io.BytesIO(); torch.save(m.state_dict(), buf); buf.seek(0)
        m2 = new.Model(cfg(dlt_variant=v, dlt_gate=gate)); m2.load_state_dict(torch.load(buf))
        rt = torch.equal(m2(xin, None, None, None), y.detach())
        if not (fin and rt): allok &= ok(f'{v}/{gate} finite + checkpoint round trip', False)
print('PASS all variants x gates: finite forward/backward, checkpoint round trip' if allok else '')
# gradients reach the gate / mill parameters
torch.manual_seed(6); m = new.Model(cfg(dlt_variant='milldyn', dlt_gate='static')); m.prepare(Xtr)
m(xin, None, None, None).pow(2).mean().backward()
allok &= ok('gradients reach b_exc, b_inh, q_proj, gate_logit',
            all(p.grad is not None and p.grad.abs().sum() > 0 for p in (m.b_exc, m.b_inh, m.q_proj.weight, m.gate_logit)))
print('ALL PASS' if allok else 'SOME CHECKS FAILED')
