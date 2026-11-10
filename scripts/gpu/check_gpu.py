"""Check the GPU code paths against the CPU before the GPU experiments (the CPU runs are the reference).

    python scripts/gpu/check_gpu.py [--data <dir with PEMS08.npz / weather.csv>] [--device cuda]

1. Every DLT variant / gate / implementation path, DLTPlug with DLinear / PatchTST / iTransformer and LagTS:
   the model is built and prepared on the CPU, copied to the GPU, and the forward output, the loss gradient of
   every parameter and LagTS's relation loss are compared (float32). Strict mode switches TF32 off; the default
   mode (TF32 for cuDNN convolutions on Ampere+, as in training) is reported with a looser tolerance.
2. Neighbour / lag selection (prepare) run on the GPU gives the same sets as on the CPU (real data if --data
   holds PEMS08.npz or weather.csv, else synthetic data).
Exit code 1 if a strict check fails. With --device cpu it compares the CPU with itself (a dry run of the script).
"""
import argparse, copy, os, sys, types
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tslib'))
os.chdir(REPO / 'tslib')
from models import DLT, DLTPlug, LagTS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--data', default='')
p.add_argument('--device', default='cuda')
a = p.parse_args()
dev = torch.device(a.device)
if dev.type == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA is not available')


def cfg(**kw):
    c = dict(task_name='long_term_forecast', seq_len=96, label_len=48, pred_len=24, enc_in=21, dec_in=21, c_out=21,
             dlt_variant='dltdyn', dlt_k=8, dlt_L=48, dlt_R=4, dlt_dyn_dim=16, dlt_ccf_len=8000, dlt_impl='kernel',
             dlt_topk=5, dlt_gate='none', dlt_gate_init=-3.0, dlt_neighbours='ccf', fix_seed=2021, moving_avg=25,
             individual=False, d_model=64, d_ff=64, n_heads=4, e_layers=2, d_layers=1, factor=3, dropout=0.0,
             activation='gelu', embed='timeF', freq='h', output_attention=False, use_norm=1, patch_len=16, stride=8,
             lagts_rel_dim=8, lagts_lambda=0.1, plug_base='DLinear', down_sampling_layers=0, down_sampling_window=1,
             down_sampling_method=None, channel_independence=1, decomp_method='moving_avg', top_k=5, num_kernels=6)
    c.update(kw)
    return types.SimpleNamespace(**c)


SPECS = [(f'DLT {v}', DLT.Model, dict(dlt_variant=v)) for v in
         ('ci', 'scalar', 'shift', 'tap3', 'full', 'topk', 'mill', 'milldyn', 'scalardyn', 'dlt', 'dltdyn')]
SPECS += [(f'DLT {v} impl={i}', DLT.Model, dict(dlt_variant=v, dlt_impl=i)) for v in ('dlt', 'dltdyn') for i in ('basis', 'dense')]
SPECS += [(f'DLT dltdyn gate={g}', DLT.Model, dict(dlt_gate=g)) for g in ('moe', 'static')]
SPECS += [(f'DLT {v} all neighbours (dense)', DLT.Model, dict(dlt_variant=v, dlt_impl='dense', dlt_neighbours='all'))
          for v in ('dlt', 'dltdyn')]
SPECS += [(f'DLTPlug {b} + dltdyn', DLTPlug.Model, dict(plug_base=b, dlt_impl='basis')) for b in ('DLinear', 'PatchTST', 'iTransformer')]
SPECS += [('LagTS', LagTS.Model, dict(enc_in=7, dec_in=7, c_out=7))]

torch.manual_seed(0)
Xtr = torch.randn(4000, 21).cumsum(0).numpy()
fails = []


def rel(a_, b_):
    return ((a_ - b_).abs().max() / (b_.abs().max() + 1e-12)).item()


for strict in (True, False):
    if dev.type == 'cuda':
        torch.backends.cudnn.allow_tf32 = not strict
        torch.backends.cuda.matmul.allow_tf32 = not strict
    tol = 1e-4 if strict else 5e-3
    print(f"\n== {'strict float32 (TF32 off)' if strict else 'default math (TF32 as in training)'}: tolerance {tol:g}")
    for name, cls, kw in SPECS:
        c = cfg(**kw)
        torch.manual_seed(1)
        m = cls(c).float()
        if hasattr(m, 'prepare'):
            m.prepare(Xtr[:, :c.enc_in])
        with torch.no_grad():                                   # move every parameter away from its initial value
            for q in m.parameters():
                q.add_(0.05 * torch.randn_like(q))
        g = copy.deepcopy(m).to(dev)
        m.eval(); g.eval()
        torch.manual_seed(2)
        x, xm = torch.randn(8, c.seq_len, c.enc_in), torch.randn(8, c.seq_len, 4)
        xd, xmd = torch.zeros(8, c.label_len + c.pred_len, c.enc_in), torch.randn(8, c.label_len + c.pred_len, 4)
        outs = []
        for model, d in ((m, torch.device('cpu')), (g, dev)):
            model.zero_grad()
            y = model(x.to(d), xm.to(d), xd.to(d), xmd.to(d))
            y = y[0] if isinstance(y, tuple) else y
            loss = (y ** 2).mean()
            if name == 'LagTS':
                loss = loss + model.extra_loss(x.to(d), y, torch.randn(8, c.pred_len, c.enc_in, generator=torch.Generator().manual_seed(3)).to(d))
            loss.backward()
            outs.append((y.detach().cpu(), loss.item(), {n: q.grad.detach().cpu() for n, q in model.named_parameters() if q.grad is not None}))
        (yc, lc, gc), (yg, lg, gg) = outs
        e_out = rel(yg, yc)
        e_grad = max((rel(gg[n], gc[n]) for n in gc if gc[n].abs().max() > 0), default=0.0)
        ok = e_out < tol and e_grad < max(tol, 10 * tol if not strict else tol) and set(gc) == set(gg)
        print(f"{'PASS' if ok else 'FAIL'}  {name:38s} output {e_out:.1e}  grads {e_grad:.1e}  loss {lc:.5f} / {lg:.5f}")
        if not ok and strict:
            fails.append(name)

# ---- neighbour / lag selection on the device
values, src = None, 'synthetic'
if a.data:
    if os.path.exists(os.path.join(a.data, 'PEMS08.npz')):
        d = np.load(os.path.join(a.data, 'PEMS08.npz'))['data']
        values, src = (d[..., 0] if d.ndim == 3 else d)[:int(0.6 * len(d))].astype(np.float64), 'PEMS08 (train split)'
    elif os.path.exists(os.path.join(a.data, 'weather.csv')):
        import pandas as pd
        d = pd.read_csv(os.path.join(a.data, 'weather.csv')).iloc[:, 1:].values
        values, src = d[:int(0.7 * len(d))].astype(np.float64), 'Weather (train split)'
if values is None:
    values = Xtr
print(f'\n== neighbour / lag selection on {dev} vs cpu ({src}, C = {values.shape[1]})')
for mode in ('ccf', 'random'):
    ic, lc_ = DLT.select_neighbours(values, 8, 48, device='cpu', mode=mode, seed=2021)
    ig, lg_ = DLT.select_neighbours(values, 8, 48, device=dev, mode=mode, seed=2021)
    same_i = (ic.cpu() == ig.cpu()).float().mean().item()
    same_l = (lc_.cpu() == lg_.cpu()).float().mean().item()
    ok = same_i == 1.0 and same_l == 1.0
    print(f"{'PASS' if ok else 'FAIL'}  neighbours ({mode}): identical {same_i:.2%}, peak lags identical {same_l:.2%}")
    if not ok:
        fails.append(f'selection {mode}')
tc = DLT.select_lags(values, ic.cpu(), 5, 48, device='cpu')
tg = DLT.select_lags(values, ic.to(dev), 5, 48, device=dev)
ok = bool((tc.cpu() == tg.cpu()).all())
print(f"{'PASS' if ok else 'FAIL'}  top-5 lag sets (MillGNN-style variants)")
if not ok:
    fails.append('lag sets')

print('\nall strict checks passed' if not fails else f'\nFAILED: {fails}')
sys.exit(1 if fails else 0)
