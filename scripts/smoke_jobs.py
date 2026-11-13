"""Pre-flight check of a job list: every configuration builds and trains one step.

    python scripts/smoke_jobs.py <list> [--device cpu|gpu] [--horizons ends|all] [--batch 2] [--match TEXT]

<list> is a suite name (work/jobs_<name>.txt, e.g. plugin_all) or a file. A long queue should not find out after
hours that a configuration fails (an option the model does not know, a shape that does not fit, a missing data
file). For every distinct configuration of the list (model, backbone, dataset and arm; with --horizons ends at the
shortest and the longest horizon of each, with all at every horizon) the script parses the run.py arguments of
the job, builds the model as run.py does, prepares a DLT branch on the training split, and sends one batch of the
training split forward and backward with batch size --batch (quick on the CPU too). --match keeps the jobs whose
--des contains TEXT. Nothing is written to work/ or results/. One line per configuration, then the number of
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
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--match', default='')
    a = p.parse_args()
    parser = get_parser()

    groups = collections.defaultdict(dict)            # configuration -> {horizon: argv}
    for line in open(paths.jobs_file(a.list)):
        if 'run.py' not in line:
            continue
        argv = job_argv(line)
        args = parser.parse_args(argv)
        if a.match not in args.des:
            continue
        H = args.pred_len
        key = (args.model, args.plug_base if args.model == 'DLTPlug' else '', args.data_path,
               re.sub(rf'_{H}(?=_|$)', '', args.des, count=1))
        groups[key].setdefault(H, argv)
    todo = []
    for key, by_h in sorted(groups.items()):
        hs = sorted(by_h) if a.horizons == 'all' else sorted({min(by_h), max(by_h)})
        todo += [(key, H, by_h[H]) for H in hs]
    print(f'{len(todo)} checks: {len(groups)} configurations of {a.list}, device {a.device}, batch {a.batch}')

    dev = torch.device('cuda:0' if a.device == 'gpu' else 'cpu')
    cache, fails, tmp = {}, [], tempfile.mkdtemp(prefix='smoke_')
    for (model, base, data_path, name), H, argv in todo:
        args = parser.parse_args(argv)
        args.use_gpu, args.gpu_type, args.device = a.device == 'gpu', 'cuda', dev
        args.batch_size, args.num_workers = min(args.batch_size, a.batch), 0
        args.result_jsonl, args.checkpoints = '', tmp
        label = f'{name} H={H}'
        t0, out = time.time(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                torch.manual_seed(args.fix_seed)
                exp = Exp_Long_Term_Forecast(args)
                dkey = (args.data, args.root_path, args.data_path, args.seq_len, args.label_len, args.pred_len,
                        args.features, args.target, args.freq, args.embed)
                if dkey not in cache:
                    cache.clear()                      # one dataset at a time keeps the memory low
                    cache[dkey] = exp._get_data(flag='train')
                data, loader = cache[dkey]
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
                n = sum(q.numel() for q in exp.model.parameters())
            print(f'OK    {label:42s} {n / 1e6:8.2f} M params {time.time() - t0:6.1f} s', flush=True)
        except Exception as e:                         # report and go on with the next configuration
            fails.append(label)
            last = [ln for ln in out.getvalue().splitlines() if ln.strip()][-3:]
            print(f'FAIL  {label:42s} {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ""}', flush=True)
            for ln in last + traceback.format_exc().strip().splitlines()[-3:-1]:
                print(f'      | {ln[:200]}')
        finally:
            exp = net = y = loss = None
            gc.collect()
            if a.device == 'gpu':
                torch.cuda.empty_cache()
    print(f'\n{len(todo) - len(fails)} of {len(todo)} checks passed' + (
        '' if not fails else '; failed: ' + ', '.join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
