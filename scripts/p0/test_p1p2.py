"""Verification of the P1 / P2 additions (run from anywhere; float64):

* --dlt_impl dense == the reference loop for dlt / dltdyn (also with all neighbours)
* --dlt_neighbours random / all: valid neighbour sets (no self, distinct, reproducible per seed)
* DLTPlug == backbone + sigma * W_c z; with --dlt_variant ci it is the backbone alone
* DLTPlug --dlt_fusion refine / input start as the backbone alone; refine == the explicit distributed-lag sum over
  the observed past and the backbone forecast, causal in time; --dlt_train decoupled gives the backbone exactly the
  gradient of its own loss; --dlt_select applies the per-variable mask
* S-Mamba: the pure-PyTorch Mamba block == the independent block of Time-Series-Library's MambaSimple (kept in
  scripts/p0/reference/), is causal and initialised like mamba_ssm; the model is bidirectional over the variates
  and drops the mark tokens
* RLinear == RevIN (learnable affine) + one linear map + RevIN^-1, written out; output follows scale and shift
* the six baselines (scripts/baselines.py) have a setting for every dataset and horizon, taken from the official
  scripts (spot checks against the files), the protocol of the official loaders, and every job line of the
  baseline check parses with run.py
* TiDE --tide_checkpoint: identical outputs and gradients (dropout masks replayed)
* iTransformer: --use_norm 1 (default) is instance normalisation, --use_norm 0 turns it off; the official
  per-horizon PEMS / Solar settings of iTransformer and S-Mamba
"""
import sys, types
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'p0'))
from models import DLT, DLTPlug, DLinear
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

# ---- DLTPlug fusions (--dlt_fusion refine / input) and decoupled training
torch.manual_seed(4); base0 = DLinear.Model(cfg())
with torch.no_grad():
    yb0 = base0(x_enc, mark, None, None)
for fu in ('refine', 'input'):
    torch.manual_seed(4); pf = DLTPlug.Model(cfg(dlt_variant='dltdyn', dlt_fusion=fu)); pf.prepare(Xtr)
    pf.base.load_state_dict(base0.state_dict())
    with torch.no_grad():
        same = torch.allclose(pf(x_enc, mark, None, None), yb0, atol=1e-12)
    ok(f'DLTPlug --dlt_fusion {fu}: starts as the backbone alone (zero coefficients / gains), no free T->H head',
       same and not hasattr(pf.dlt, 'cross_head'))
# refine: explicit distributed-lag sum over [observed past ; backbone forecast], future part only
torch.manual_seed(5); pr = DLTPlug.Model(cfg(dlt_variant='dlt', dlt_fusion='refine', dlt_impl='kernel')); pr.prepare(Xtr)
with torch.no_grad():
    pr.dlt.alpha.normal_(0, 0.3)
    yb = pr._backbone(x_enc, mark, None, None)
    c = pr.refine(x_enc, yb)
    xx = x_enc.permute(0, 2, 1); mu = xx.mean(-1, keepdim=True)
    sn = torch.cat([xx - mu, yb.permute(0, 2, 1) - mu], -1)            # centred, not scaled
    ref = reference_z(pr.dlt, sn)[..., -24:].permute(0, 2, 1)
    err = (c - ref).abs().max().item()
ok(f'refine == explicit sum_j sum_tau K_ij(tau) s_j(t - tau) over centred past + forecast (max err {err:.1e})', err < 1e-10)
with torch.no_grad():                                  # a change of the forecast at step h0 moves steps >= h0 only
    yb2 = yb.clone(); yb2[:, 10:, :] += 1.0
    moved = (pr.refine(x_enc, yb2) - c).abs().amax((0, 2)).gt(1e-12).nonzero().flatten().tolist()
ok('refine is causal in time: a change of the backbone forecast at step 10 moves the correction at steps >= 10 only',
   moved == list(range(10, 24)))
xf = x_enc.clone(); xf[..., 0] = 0.3                                      # variable 0 flat over the window
with torch.no_grad():
    cf = pr.refine(xf, pr._backbone(xf, mark, None, None) + 5.0)
ok(f'refine stays bounded with a flat neighbour window and a far-off forecast (max |c| {cf.abs().max().item():.2f})',
   bool(torch.isfinite(cf).all()) and cf.abs().max().item() < 100)
# decoupled training: the backbone gets exactly the gradient of its own loss; the branch learns the residual
yt = torch.randn(4, 24, 9)
for fu in ('output', 'refine'):
    torch.manual_seed(6); pd_ = DLTPlug.Model(cfg(dlt_variant='dltdyn', dlt_fusion=fu, dlt_train='decoupled')); pd_.prepare(Xtr)
    torch.manual_seed(6); b_alone = DLinear.Model(cfg()); b_alone.load_state_dict(pd_.base.state_dict())
    with torch.no_grad():
        for q in pd_.dlt.parameters():
            q.add_(0.05 * torch.randn_like(q))
    pd_.train()
    out = pd_(x_enc, mark, None, None)
    loss = torch.nn.functional.mse_loss(out, yt) + pd_.extra_loss(x_enc, out, yt)
    loss.backward()
    torch.nn.functional.mse_loss(b_alone(x_enc, mark, None, None), yt).backward()
    gb = max((p1.grad - p2.grad).abs().max().item() for p1, p2 in zip(pd_.base.parameters(), b_alone.parameters()))
    gd = sum(q.grad.abs().sum().item() for q in pd_.dlt.parameters() if q.grad is not None)
    ok(f'--dlt_train decoupled ({fu}): backbone gradient == that of the backbone alone (max diff {gb:.1e}), '
       f'branch gradient non-zero', gb < 1e-12 and gd > 0)
# per-variable selection mask
torch.manual_seed(7); ps_ = DLTPlug.Model(cfg(dlt_variant='dltdyn')); ps_.prepare(Xtr); ps_.eval()
with torch.no_grad():
    yb_, c_ = ps_.parts(x_enc, mark, None, None)
    ps_.select_mask = torch.tensor([True, False] * 4 + [True])
    y_sel = ps_(x_enc, mark, None, None)
ok('--dlt_select: the correction is added for the kept variables only',
   torch.allclose(y_sel, yb_ + c_ * ps_.select_mask.to(c_.dtype), atol=1e-12)
   and torch.allclose(y_sel[..., 1], yb_[..., 1], atol=1e-12))

# ---- S-Mamba
import math
from layers.MambaPy import Mamba
from models import SMamba
from reference.mamba_simple_block import MambaBlock
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
ok(f'Mamba block (PyTorch) == TSLib MambaSimple block with the same parameters (max err {worst:.1e})', worst < 1e-10)
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

# ---- RLinear
from models import RLinear
cr = cfg(enc_in=5, seq_len=16, pred_len=8, dropout=0.0, use_norm=1)
torch.manual_seed(3); rl = RLinear.Model(cr).eval()
with torch.no_grad():
    rl.rev.affine_weight.uniform_(0.5, 1.5); rl.rev.affine_bias.uniform_(-0.3, 0.3)
xr = torch.randn(3, 16, 5) * 2 + 1
with torch.no_grad():
    yr = rl(xr, None, None, None)
    mu = xr.mean(1, keepdim=True); sd = torch.sqrt(xr.var(1, keepdim=True, unbiased=False) + 1e-5)
    w, b = rl.rev.affine_weight, rl.rev.affine_bias
    z = ((xr - mu) / sd) * w + b                                 # RevIN, affine
    p = rl.Linear(z.transpose(1, 2)).transpose(1, 2)             # one map seq_len -> pred_len for every variate
    ref = (p - b) / (w + 1e-10) * sd + mu                        # RevIN^-1 (eps^2 in the official code)
    err = (yr - ref).abs().max().item()
    err2 = (rl(xr * 3.0 + 5.0, None, None, None) - (yr * 3.0 + 5.0)).abs().max().item()
ok(f'RLinear == RevIN + linear map + RevIN^-1, written out (max err {err:.1e})', err < 1e-10)
ok(f'RLinear output follows scale and shift of the input (max err {err2:.1e})', err2 < 1e-4)

# ---- settings of the six baselines (scripts/baselines.py) and the baseline check
sys.path.insert(0, str(REPO / 'scripts'))
import exps, baselines, shlex, paths
from run import get_parser
A = {(m, ds, H): baselines.args(m, ds, H) for m in baselines.BASELINES for ds in baselines.FILES
     for H in baselines.horizons(ds)}
ok(f'a setting for every baseline, dataset and horizon ({len(A)} = 6 x 52)', len(A) == 6 * 52)
a = lambda m, ds, H: A[(m, ds, H)][0]
ok('official values (iTransformer ETTh1-336: d_model 512, Exchange-336: 1 epoch; S-Mamba ETTh1-96: d_state 2, '
   'lr 7e-5; TiDE: decoder output 8; RLinear ETTh2: batch 256, 15 epochs; PatchTST ETTm1-192: batch 128)',
   '--d_model 512' in a('iTransformer', 'ETTh1', 336) and '--train_epochs 1' in a('iTransformer', 'Exchange', 336)
   and '--d_state 2' in a('SMamba', 'ETTh1', 96) and '--learning_rate 0.00007' in a('SMamba', 'ETTh1', 96)
   and all('--c_out 8' in a('TiDE', ds, 96) for ds in ('ETTh1', 'ECL', 'Weather'))
   and '--batch_size 256' in a('RLinear', 'ETTh2', 96) and '--train_epochs 15' in a('RLinear', 'ETTh2', 96)
   and '--batch_size 128' in a('PatchTST', 'ETTm1', 192))
ok('documented fallbacks (PatchTST Solar / PEMS: Electricity; TiDE: ETTh1; RLinear Exchange: ETTh1, PEMS: '
   'Electricity; DLinear without a script: library defaults)',
   'ECL configuration' in A[('PatchTST', 'PEMS08', 12)][1] and 'ETTh1 configuration' in A[('TiDE', 'Weather', 96)][1]
   and 'ETTh1 configuration' in A[('RLinear', 'Exchange', 96)][1] and 'ECL configuration' in A[('RLinear', 'PEMS04', 12)][1]
   and a('DLinear', 'Weather', 96) == '' and 'c_out' not in a('iTransformer', 'ECL', 96))
ok('official PEMS settings per horizon (iTransformer PEMS08-48: 4 layers, use_norm 0; S-Mamba PEMS08-24: lr 7e-4)',
   '--e_layers 4' in a('iTransformer', 'PEMS08', 48) and '--use_norm 0' in a('iTransformer', 'PEMS08', 48)
   and '--use_norm 1' in a('iTransformer', 'PEMS08', 12) and '--learning_rate 0.0007' in a('SMamba', 'PEMS08', 24)
   and all('--use_norm 0' in a(m, 'PEMS04', H) for m in ('iTransformer', 'SMamba') for H in (12, 24, 48, 96)))
ok('official PEMS03/07 and Solar settings (S-Mamba PEMS03-12: 5 epochs, PEMS03-48: 5 layers lr 1e-4; '
   'iTransformer PEMS07-96: batch 16, Solar: lr 5e-4)',
   '--train_epochs 5' in a('SMamba', 'PEMS03', 12) and '--e_layers 5' in a('SMamba', 'PEMS03', 48)
   and '--learning_rate 0.0001' in a('SMamba', 'PEMS03', 48) and '--batch_size 16' in a('iTransformer', 'PEMS07', 96)
   and '--use_norm 0' in a('SMamba', 'PEMS07', 24) and '--learning_rate 0.0005' in a('iTransformer', 'Solar', 336))
ok('protocol of the official loaders (freq h, drop_last, test batch 1; PEMS / Solar: no time marks; PEMS: their split)',
   baselines.protocol('ETTm1') == '--freq h --drop_last --test_batch_size 1'
   and baselines.protocol('Solar').endswith('--no_time_marks') and '--pems_split itransformer' in baselines.protocol('PEMS07'))
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ['exps.py', 'baselines', '--seeds', '2021', '--list', '_test_baselines']
    exps.main()
P, bad, n = get_parser(), [], 0
jf = paths.jobs_file('_test_baselines')
for line in open(jf):
    toks = shlex.split(line.split(' > ')[0]); args = P.parse_args(toks[toks.index('run.py') + 1:]); n += 1
    if args.freq != 'h' or not args.drop_last or args.test_batch_size != 1:
        bad.append(args.des)
import os; os.remove(jf)
ok(f'baseline check: every job line parses with run.py and carries the protocol ({n} lines)', n == 312 and not bad)

# ---- TiDE --tide_checkpoint: same outputs and gradients (dropout on), far less memory kept for backward
from models import TiDE
ct = cfg(enc_in=12, dec_in=12, c_out=8, seq_len=96, pred_len=24, d_model=32, d_ff=16, e_layers=2, d_layers=2,
         dropout=0.3, freq='h', embed='timeF')
torch.manual_seed(1); t0 = TiDE.Model(ct).train()
ct.tide_checkpoint = True
torch.manual_seed(1); t1 = TiDE.Model(ct).train()
xt, ytt = torch.randn(4, 96, 12), torch.randn(4, 24, 12)
torch.manual_seed(7); o0 = t0(xt, None, None, None); ((o0 - ytt) ** 2).mean().backward()
torch.manual_seed(7); o1 = t1(xt, None, None, None); ((o1 - ytt) ** 2).mean().backward()
gdiff = max((p.grad - q.grad).abs().max().item() for p, q in zip(t0.parameters(), t1.parameters()))
ok(f'TiDE --tide_checkpoint: identical outputs and gradients with dropout (max gradient difference {gdiff:.1e})',
   torch.equal(o0, o1) and gdiff == 0.0)

# ---- iTransformer --use_norm
from models import iTransformer
ci = cfg(enc_in=6, d_model=16, d_ff=16, e_layers=1, n_heads=2, output_attention=False)
torch.manual_seed(0); it1 = iTransformer.Model(ci).eval()
ci.use_norm = 0
torch.manual_seed(0); it0 = iTransformer.Model(ci).eval()
xq, mq = torch.randn(2, 96, 6), torch.randn(2, 96, 4)
with torch.no_grad():
    y1, y0 = it1(xq, mq, None, None), it0(xq, mq, None, None)
    err = (it1(xq * 3.0 + 5.0, mq, None, None) - (y1 * 3.0 + 5.0)).abs().max().item()
ok(f'iTransformer --use_norm 1 (default): output follows scale and shift of the input (max err {err:.1e})', err < 1e-4)
ok('iTransformer --use_norm 0: no instance normalisation (same weights, different output)', (y1 - y0).abs().max() > 1e-3)
print('ALL PASS' if ok_all else 'SOME CHECKS FAILED')
