"""Check the GPU code paths against the CPU (the CPU runs are the reference of all reported results).

    python scripts/gpu/check_gpu.py [--data <dataset file or directory>] [--device cuda] [--all]

Default: the stand-alone DLT-dyn model (both implementation paths). --all adds every other DLT variant and gate,
DLTPlug with DLinear / PatchTST / iTransformer and LagTS.

1. Each model is built and prepared on the CPU, copied to the GPU, and the forward output and the gradient of
   every parameter are compared in three settings:
     float64            logic check: only rounding at ~1e-15 separates the devices          tolerance 1e-9
     float32, TF32 off  the training dtype without TensorFloat-32                           tolerance 1e-4
     float32, default   as in training (PyTorch enables TF32 for cuDNN convolutions)        reported only
   Gradient error of a parameter = max |g_gpu - g_cpu| / max(max |g_cpu|, 1e-3 * G), G = the largest gradient entry
   of the model. The floor matters: some gradients are exactly zero in exact arithmetic (the bias of an attention
   key projection, by the shift invariance of the softmax) and hold nothing but rounding noise, which must not be
   compared with itself. Such parameters are listed as "zero-gradient".
2. Neighbour selection (prepare) on the GPU picks the same sources as on the CPU. Ties (e.g. a dead sensor,
   whose correlation with every source is 0) may be broken differently and are accepted if the scores agree.

Exit code 1 if a float64 or float32/TF32-off check fails. --device cpu compares the CPU with itself (dry run).
"""
import argparse, contextlib, copy, io, os, sys, types
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from models import DLT  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--data', default='', help='dataset file (.csv / .npz) or a directory that holds one, for the selection check')
p.add_argument('--device', default='cuda')
p.add_argument('--all', action='store_true', help='also the other DLT variants, gates, DLTPlug and LagTS')
a = p.parse_args()
dev = torch.device(a.device)
if dev.type == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA is not available (torch %s)' % torch.__version__)
print(f'torch {torch.__version__}, device {dev}' + (
    f' ({torch.cuda.get_device_name(dev)}, CUDA runtime {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})'
    if dev.type == 'cuda' else ''))
TF32_DEFAULT = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
FLOOR, ZERO = 1e-3, 1e-6


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


SPECS = [(f'DLT dltdyn impl={i}', DLT.Model, dict(dlt_variant='dltdyn', dlt_impl=i)) for i in ('kernel', 'basis')]
if a.all:
    from models import DLTPlug, LagTS  # noqa: E402
    SPECS += [(f'DLT {v}', DLT.Model, dict(dlt_variant=v)) for v in
              ('ci', 'scalar', 'shift', 'tap3', 'full', 'topk', 'mill', 'milldyn', 'scalardyn', 'dlt')]
    SPECS += [('DLT dlt impl=basis', DLT.Model, dict(dlt_variant='dlt', dlt_impl='basis'))]
    SPECS += [(f'DLT {v} impl=dense', DLT.Model, dict(dlt_variant=v, dlt_impl='dense')) for v in ('dlt', 'dltdyn')]
    SPECS += [(f'DLT dltdyn gate={g}', DLT.Model, dict(dlt_gate=g)) for g in ('moe', 'static')]
    SPECS += [(f'DLT {v} all neighbours (dense)', DLT.Model, dict(dlt_variant=v, dlt_impl='dense', dlt_neighbours='all'))
              for v in ('dlt', 'dltdyn')]
    SPECS += [(f'DLTPlug {b} + dltdyn', DLTPlug.Model, dict(plug_base=b, dlt_impl='basis'))
              for b in ('DLinear', 'PatchTST', 'iTransformer')]
    SPECS += [('LagTS', LagTS.Model, dict(enc_in=7, dec_in=7, c_out=7))]

MODES = [('float64', torch.float64, None, 1e-9, True),
         ('float32, TF32 off', torch.float32, (False, False), 1e-4, True),
         ('float32, default math as in training (reported only)', torch.float32, TF32_DEFAULT, 5e-3, False)]

torch.manual_seed(0)
Xtr = torch.randn(4000, 21).cumsum(0).numpy()
fails, zero_note = [], {}


def build(name, cls, kw):
    """model on the CPU (float32) with prepared neighbour sets and every parameter moved off its initial value"""
    c = cfg(**kw)
    torch.manual_seed(1)
    m = cls(c).float()
    with contextlib.redirect_stdout(io.StringIO()):
        if hasattr(m, 'prepare'):
            m.prepare(Xtr[:, :c.enc_in])
    with torch.no_grad():
        for q in m.parameters():
            q.add_(0.05 * torch.randn_like(q))
    return c, m.eval()


def run(model, c, name, d, dt):
    g = torch.Generator().manual_seed(2)
    x, xm = torch.randn(8, c.seq_len, c.enc_in, generator=g), torch.randn(8, c.seq_len, 4, generator=g)
    xd, xmd = torch.zeros(8, c.label_len + c.pred_len, c.enc_in), torch.randn(8, c.label_len + c.pred_len, 4, generator=g)
    tr = torch.randn(8, c.pred_len, c.enc_in, generator=g)
    to = lambda t: t.to(device=d, dtype=dt)
    model.zero_grad()
    y = model(to(x), to(xm), to(xd), to(xmd))
    y = y[0] if isinstance(y, tuple) else y
    loss = (y ** 2).mean()
    if name == 'LagTS':
        loss = loss + model.extra_loss(to(x), y, to(tr))
    loss.backward()
    return y.detach().cpu().double(), {n: q.grad.detach().cpu().double() for n, q in model.named_parameters() if q.grad is not None}


models = [(name,) + build(name, cls, kw) for name, cls, kw in SPECS]
for label, dt, tf32, tol, counts in MODES:
    if tf32 is not None:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    print(f'\n== {label}: tolerance {tol:g} (outputs), {tol:g} (gradients)' if counts else f'\n== {label}')
    for name, c, m in models:
        mc = copy.deepcopy(m).to(dt)
        md = copy.deepcopy(mc).to(dev)
        yc, gc = run(mc, c, name, torch.device('cpu'), dt)
        yd, gd = run(md, c, name, dev, dt)
        e_out = ((yd - yc).abs().max() / yc.abs().max()).item()
        G = max(g.abs().max().item() for g in gc.values())
        errs = {n: (gd[n] - gc[n]).abs().max().item() / max(gc[n].abs().max().item(), FLOOR * G) for n in gc}
        worst = max(errs, key=errs.get)
        zero = [n for n in gc if gc[n].abs().max().item() < ZERO * G]
        if zero:
            zero_note[name] = zero
        ok = e_out < tol and errs[worst] < tol and set(gc) == set(gd)
        tag = ('PASS' if ok else 'FAIL') if counts else ('ok  ' if ok else 'NOTE')
        print(f'{tag}  {name:36s} output {e_out:.1e}   gradients {errs[worst]:.1e} (largest at {worst})')
        if not ok and counts:
            fails.append(f'{name} [{label}]')
torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = TF32_DEFAULT
if zero_note:
    print('\nzero-gradient parameters (below 1e-6 of the largest gradient; compared on the absolute scale):')
    for name, zs in zero_note.items():
        print(f'  {name}: ' + ', '.join(zs))


# ---- neighbour selection on the device
def load_values(path):
    """(train split of a dataset, label) from a .csv / .npz file or the first known file in a directory"""
    if os.path.isdir(path):
        cand = [os.path.join(path, f) for f in ('weather.csv', 'PEMS08.npz', 'PEMS04.npz', 'electricity.csv', 'ETTh1.csv')]
        cand += sorted(str(q) for q in Path(path).glob('*.csv')) + sorted(str(q) for q in Path(path).glob('*.npz'))
        path = next((q for q in cand if os.path.exists(q)), '')
    if path.endswith('.npz'):
        d = np.load(path)['data']
        d = d[..., 0] if d.ndim == 3 else d
        return d[:int(0.6 * len(d))].astype(np.float64), f'{os.path.basename(path)} (first 60 %)'
    if path.endswith('.csv'):
        import pandas as pd
        d = pd.read_csv(path).iloc[:, 1:].values
        return d[:int(0.7 * len(d))].astype(np.float64), f'{os.path.basename(path)} (first 70 %)'
    return Xtr, 'synthetic data (pass --data for a real dataset)'


values, src = load_values(a.data) if a.data else load_values('')
print(f'\n== neighbour selection on {dev} vs cpu: {src}, {values.shape[1]} variables')
x = DLT._diff_standardise(values, 8000, 'cpu')
score = torch.stack([DLT._abs_ccf(x, tau) for tau in range(49)]).max(0).values       # (target, source)
for mode in ('ccf', 'random'):
    ic, lc = DLT.select_neighbours(values, 8, 48, device='cpu', mode=mode, seed=2021)
    ig, lg = DLT.select_neighbours(values, 8, 48, device=dev, mode=mode, seed=2021)
    ic, ig, lc, lg = ic.cpu(), ig.cpu(), lc.cpu(), lg.cpu()
    same = (ic == ig).all(1)
    sc, sg = score.gather(1, ic).sort(1).values, score.gather(1, ig).sort(1).values
    tied = (sc - sg).abs().max(1).values < 1e-9                    # different sources with the same scores
    ok = bool((same | tied).all()) and bool(((lc == lg) | ~same[:, None]).all())
    print(f"{'PASS' if ok else 'FAIL'}  neighbours ({mode}): identical for {int(same.sum())} of {len(same)} targets"
          + (f', {int((~same & tied).sum())} with ties broken differently' if not same.all() else ''))
    if not ok:
        fails.append(f'neighbour selection ({mode})')
if a.all:
    ic = DLT.select_neighbours(values, 8, 48, device='cpu')[0]
    tc = DLT.select_lags(values, ic, 5, 48, device='cpu').cpu()
    tg = DLT.select_lags(values, ic.to(dev), 5, 48, device=dev).cpu()
    share = (tc == tg).all(-1).float().mean().item()
    ok = share > 0.999
    print(f"{'PASS' if ok else 'FAIL'}  top-5 lag sets (MillGNN-style variants): identical for {share:.2%} of the pairs")
    if not ok:
        fails.append('lag sets')

print('\nall checks passed' if not fails else '\nFAILED: ' + '; '.join(fails))
sys.exit(1 if fails else 0)
