"""Pre-flight check of a job list: every configuration builds and trains one step.

    python scripts/smoke_jobs.py <list> [--device cpu|gpu] [--horizons ends|all] [--batch N] [--match TEXT]

<list> is a suite name (work/jobs_<name>.txt, e.g. plugin_all) or a file. A long queue should not find out after
hours that a configuration fails (an option the model does not know, a shape that does not fit, a missing data
file). For every distinct configuration of the list (model, backbone, dataset and arm; with --horizons ends at the
shortest and the longest horizon of each, with all at every horizon) the script parses the run.py arguments of
the job, builds the model as run.py does, prepares a DLT branch on the training split, and trains it for one step
(forward, backward, optimizer step) on one batch of the training split. On the GPU (the default when there is one)
the batch has the job's own size, and the peak GPU memory of the step is printed: what the job will need, and the
sum to keep below the card's memory when several jobs share it. On the CPU the batch has 2 samples, so that the
check stays quick; --batch N sets another size (at most the job's). --match keeps the jobs whose
--des matches the regular expression TEXT (e.g. 'ETTh1_' or '_(PatchTST|iTransformer)_'). Nothing is written to work/ or results/. One line per configuration, then the number of
failures; exit code 1 if there is one.
"""
import argparse, collections, contextlib, gc, io, os, re, shlex, sys, tempfile, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
os.chdir(REPO)                                        # TSLib finds models/ relative to the working directory
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import paths  # noqa: E402
from run import get_parser  # noqa: E402
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # noqa: E402


def job_argv(line):
    toks = shlex.split(line.split(' > ')[0])
    return toks[toks.index('run.py') + 1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('list')
    p.add_argument('--device', default='gpu' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'gpu'])
    p.add_argument('--horizons', default='ends', choices=['ends', 'all'])
    p.add_argument('--batch', type=int, help="batch size (default: the job's own on the GPU, 2 on the CPU)")
    p.add_argument('--match', default='')
    a = p.parse_args()
    parser = get_parser()

    groups = collections.defaultdict(dict)            # configuration -> {horizon: argv}
    for line in open(paths.jobs_file(a.list)):
        if 'run.py' not in line:
            continue
        argv = job_argv(line)
        args = parser.parse_args(argv)
        if not re.search(a.match, args.des):
            continue
        H = args.pred_len
        key = (args.model, args.plug_base if args.model == 'DLTPlug' else '', args.data_path,
               re.sub(rf'_{H}(?=_|$)', '', args.des, count=1))
        groups[key].setdefault(H, argv)
    todo = []
    for key, by_h in sorted(groups.items()):
        hs = sorted(by_h) if a.horizons == 'all' else sorted({min(by_h), max(by_h)})
        todo += [(key, H, by_h[H]) for H in hs]
    gpu = a.device == 'gpu'
    if gpu and not torch.cuda.is_available():
        raise SystemExit('--device gpu, but PyTorch sees no GPU')
    batch = a.batch or (None if gpu else 2)
    print(f'{len(todo)} checks: {len(groups)} configurations of {a.list}; device '
          f'{torch.cuda.get_device_name(0) if gpu else "cpu"}; batch {batch or "of each job"}')

    dev = torch.device('cuda:0' if gpu else 'cpu')
    cache, fails, peaks, tmp = {}, [], [], tempfile.mkdtemp(prefix='smoke_')
    for (model, base, data_path, name), H, argv in todo:
        args = parser.parse_args(argv)
        args.use_gpu, args.gpu_type, args.device = gpu, 'cuda', dev
        args.batch_size, args.num_workers = min(args.batch_size, batch or args.batch_size), 0
        args.result_jsonl, args.checkpoints = '', tmp
        label = f'{name} H={H}'
        t0, out, mem = time.time(), io.StringIO(), ''
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        try:
            with contextlib.redirect_stdout(out):
                torch.manual_seed(args.fix_seed)
                exp = Exp_Long_Term_Forecast(args)
                dkey = (args.data, args.root_path, args.data_path, args.seq_len, args.label_len, args.pred_len,
                        args.features, args.target, args.freq, args.embed)
                if dkey not in cache:
                    cache.clear()                      # one dataset at a time keeps the memory low
                    cache[dkey] = exp._get_data(flag='train')[0]
                data = cache[dkey]
                loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0)
                net = exp.model.module if isinstance(exp.model, nn.DataParallel) else exp.model
                if hasattr(net, 'prepare'):
                    net.prepare(data)
                bx, by, bxm, bym = (t.float().to(exp.device) for t in next(iter(loader)))
                dec = torch.cat([by[:, :args.label_len, :], torch.zeros_like(by[:, -args.pred_len:, :])], dim=1)
                y = exp.model(bx, bxm, dec, bym)
                y = y[0] if isinstance(y, tuple) else y
                f = -1 if args.features == 'MS' else 0
                y, target = y[:, -args.pred_len:, f:], by[:, -args.pred_len:, f:]
                if y.shape != target.shape:
                    raise RuntimeError(f'output {tuple(y.shape)}, target {tuple(target.shape)}')
                loss = exp._select_criterion()(y, target)
                loss.backward()
                bad = [n for n, q in exp.model.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
                if not torch.isfinite(loss) or bad:
                    raise RuntimeError(f'loss {loss.item()}, non-finite gradients in {bad[:3]}')
                exp._select_optimizer().step()         # Adam allocates its state here, as in training
                n = sum(q.numel() for q in exp.model.parameters())
            if gpu:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                peaks.append((peak, label, args.batch_size))
                mem = f'  batch {args.batch_size:3d}  peak {peak:5.2f} GB'
            print(f'OK    {label:42s} {n / 1e6:8.2f} M params {time.time() - t0:6.1f} s{mem}', flush=True)
        except Exception as e:                         # report and go on with the next configuration
            fails.append(label)
            last = [ln for ln in out.getvalue().splitlines() if ln.strip()][-3:]
            print(f'FAIL  {label:42s} {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ""}', flush=True)
            for ln in last + traceback.format_exc().strip().splitlines()[-3:-1]:
                print(f'      | {ln[:200]}')
        finally:
            exp = net = y = loss = bx = by = bxm = bym = dec = target = None    # free the GPU before the next check
            gc.collect()
            if a.device == 'gpu':
                torch.cuda.empty_cache()
    if peaks:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f'\nlargest peak GPU memory of one training step (card: {total:.0f} GB); jobs that run at the same time '
              f'on one card need the sum of theirs:')
        for peak, label, bs in sorted(peaks, reverse=True)[:8]:
            print(f'  {peak:5.2f} GB  {label}  (batch {bs})')
    print(f'\n{len(todo) - len(fails)} of {len(todo)} checks passed' + (
        '' if not fails else '; failed: ' + ', '.join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
