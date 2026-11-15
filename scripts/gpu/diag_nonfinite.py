"""Where do NaN / inf forecasts come from? Re-tests a finished run from its saved checkpoint.

    python scripts/gpu/diag_nonfinite.py <job name> [--list plugin_all] [--seed 2021] [--split test]

<job name> is the --des of the job (e.g. Traffic_96_TimesNet_alone); the job line is taken from the job list
(work/jobs_<list>.txt) and the checkpoint from work/ckpt (kept after every run). The script runs the split
batch by batch as the test of run.py does and reports
  - the batches and windows with non-finite forecasts, and the variables affected,
  - what is special about their input windows (variables that are constant over the window, largest value),
  - whether the same batch is finite on the GPU with TensorFloat-32 off and on the CPU (float32): finite there
    means a numerical problem of the GPU kernels, not of the model or the data,
  - the layer where the first non-finite value appears (forward hooks), on the device of the run.
Nothing is written.
"""
import argparse, contextlib, glob, io, os, re, shlex, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
os.chdir(REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import paths  # noqa: E402
from run import get_parser  # noqa: E402
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # noqa: E402


def forward(model, batch, args, dev):
    bx, by, bxm, bym = (t.float().to(dev) for t in batch)
    dec = torch.cat([by[:, :args.label_len, :], torch.zeros_like(by[:, -args.pred_len:, :])], dim=1)
    y = model(bx, bxm, dec, bym)
    return (y[0] if isinstance(y, tuple) else y)[:, -args.pred_len:, :]


def first_bad_layer(model, batch, args, dev):
    """name of the first module (in execution order) whose output contains a non-finite value"""
    found, hooks = [], []

    def hook(name):
        def f(_, __, out):
            outs = out if isinstance(out, (tuple, list)) else (out,)
            if not found and any(torch.is_tensor(o) and o.is_floating_point() and not torch.isfinite(o).all() for o in outs):
                found.append(name)
        return f
    for name, m in model.named_modules():
        if name:
            hooks.append(m.register_forward_hook(hook(name)))
    with torch.no_grad():
        forward(model, batch, args, dev)
    for h in hooks:
        h.remove()
    return found[0] if found else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('des')
    p.add_argument('--list', default='plugin_all')
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    a = p.parse_args()

    line = next((l for l in open(paths.jobs_file(a.list)) if f'--des {a.des} ' in l and f'--fix_seed {a.seed} ' in l), None)
    if line is None:
        raise SystemExit(f'no job {a.des} (seed {a.seed}) in {paths.jobs_file(a.list)}')
    toks = shlex.split(line.split(' > ')[0])
    args = get_parser().parse_args(toks[toks.index('run.py') + 1:])
    ckpts = sorted(glob.glob(os.path.join(args.checkpoints, f'*_{a.des}_0*', 'checkpoint.pth')))
    if not ckpts:
        raise SystemExit(f'no checkpoint of {a.des} under {args.checkpoints}')
    if len(ckpts) > 1:
        print(f'note: {len(ckpts)} checkpoints match, using the newest')
        ckpts.sort(key=os.path.getmtime)
    gpu = torch.cuda.is_available()
    dev = torch.device('cuda:0' if gpu else 'cpu')
    args.use_gpu, args.gpu_type, args.device, args.num_workers = gpu, 'cuda', dev, 0
    with contextlib.redirect_stdout(io.StringIO()):
        exp = Exp_Long_Term_Forecast(args)
        data, loader = exp._get_data(flag=a.split)
        train_data = exp._get_data(flag='train')[0]
        net = exp.model.module if isinstance(exp.model, torch.nn.DataParallel) else exp.model
        if hasattr(net, 'prepare'):                     # buffers (neighbours) come from the checkpoint anyway
            net.prepare(train_data)
    state = torch.load(ckpts[-1], map_location=dev)
    exp.model.load_state_dict(state)
    exp.model.eval()
    print(f'{a.des}: checkpoint {os.path.relpath(ckpts[-1], REPO)}; {a.split} split, {len(data)} windows, '
          f'batch {args.batch_size}, device {torch.cuda.get_device_name(0) if gpu else "cpu"}, '
          f'TF32 matmul {torch.backends.cuda.matmul.allow_tf32} / cuDNN {torch.backends.cudnn.allow_tf32}')

    bad, n_bad_values, first = [], 0, None
    with torch.no_grad():
        for i, batch in enumerate(loader):
            y = forward(exp.model, batch, args, dev)
            ok = torch.isfinite(y)
            if not ok.all():
                nb = int((~ok).sum())
                n_bad_values += nb
                win = (~ok).flatten(1).any(1).nonzero().flatten().tolist()
                bad.append((i, nb, [i * args.batch_size + w for w in win]))
                if first is None:
                    first = (i, batch, (~ok).any(1).any(0).nonzero().flatten().tolist())
    total = len(data) * args.pred_len * args.c_out
    print(f'non-finite forecasts: {n_bad_values} of {total} values, in {len(bad)} of {i + 1} batches')
    if not bad:
        print('all forecasts are finite on this device: the NaN of the run does not come back from the checkpoint')
        return
    for i, nb, wins in bad[:10]:
        print(f'  batch {i}: {nb} values, windows {wins[:8]}{" ..." if len(wins) > 8 else ""}')
    i, batch, varis = first
    x = batch[0].double()
    const = (x.std(1) == 0)                              # (B, C): variable constant over the input window
    print(f'first bad batch {i}: variables with non-finite forecasts {varis[:10]}{" ..." if len(varis) > 10 else ""}; '
          f'input windows with a constant variable: {int(const.any(1).sum())} of {x.shape[0]}; '
          f'largest |input| {x.abs().max().item():.3g}')
    print(f'first module with a non-finite output ({dev.type}): {first_bad_layer(exp.model, batch, args, dev)}')

    def finite_with(model, d):
        with torch.no_grad():
            return bool(torch.isfinite(forward(model, batch, args, d)).all())
    if gpu:
        t32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        print(f'same batch on the GPU, TF32 off: {"finite" if finite_with(exp.model, dev) else "NOT finite"}')
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = t32
        cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        print(f'same batch on the GPU, cuDNN off: {"finite" if finite_with(exp.model, dev) else "NOT finite"}')
        torch.backends.cudnn.enabled = cudnn
        cpu_model = exp.model.to('cpu')
        print(f'same batch on the CPU (float32): {"finite" if finite_with(cpu_model, torch.device("cpu")) else "NOT finite"}')
        if not finite_with(cpu_model, torch.device('cpu')):
            print(f'first module with a non-finite output (cpu): {first_bad_layer(cpu_model, batch, args, torch.device("cpu"))}')


if __name__ == '__main__':
    main()
