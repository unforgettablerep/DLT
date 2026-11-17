"""P2: parameters, training / inference time per batch and peak memory of DLT and the baselines.

    python scripts/p2/efficiency.py --device gpu          # on the CUDA machine -> results/p2/efficiency_gpu.jsonl
    python scripts/p2/efficiency.py --device cpu --threads 4                 # -> results/p2/efficiency_cpu.jsonl

Default models: DLT-dyn and the six baselines (S-Mamba, iTransformer, PatchTST, RLinear, DLinear, TiDE).
--variants adds the kernel variants of the controlled study and the plug-in.
Every (dataset, model) is measured in a fresh subprocess (clean peak-memory counter) on random data with the
dataset's shape: batch 32, input 96, horizon 96. Baselines use the official hyper-parameters of that dataset
(scripts/baselines.py). Train time = forward + MSE + backward + Adam step. Memory: GPU = peak allocated by
PyTorch; CPU = growth of the peak resident set size of the process during model creation and training.
"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

DATASETS = {'Weather': 21, 'PEMS08': 170, 'ECL': 321, 'Traffic': 862}
# name: (TSLib model, extra args). Baselines also get their official arguments for the dataset (scripts/baselines.py).
OURS = {'DLT-dyn': ('DLT', '--dlt_variant dltdyn --dlt_impl basis')}
BASELINES = {'S-Mamba': ('SMamba', ''), 'iTransformer': ('iTransformer', ''), 'PatchTST': ('PatchTST', ''),
             'RLinear': ('RLinear', ''), 'DLinear': ('DLinear', ''), 'TiDE': ('TiDE', '')}
VARIANTS = {   # --variants: the kernels of the controlled study, the plug-in and further configurations
    'CI (RLinear-style)': ('DLT', '--dlt_variant ci'),
    'scalar': ('DLT', '--dlt_variant scalar'),
    'full kernel': ('DLT', '--dlt_variant full'),
    'top-5 lags': ('DLT', '--dlt_variant topk'),
    'MillGNN-dyn kernel': ('DLT', '--dlt_variant milldyn'),
    'DLT': ('DLT', '--dlt_variant dlt --dlt_impl kernel'),
    'DLT-dyn (kernel path)': ('DLT', '--dlt_variant dltdyn --dlt_impl kernel'),
    'iTransformer + DLT-dyn': ('DLTPlug', '--plug_base iTransformer --dlt_variant dltdyn --dlt_impl basis'),
    'PatchTST (paper config)': ('PatchTST', '--e_layers 3 --d_model 128 --d_ff 256 --n_heads 16'),
}
MODELS = {**OURS, **BASELINES, **VARIANTS}
# too slow or too large for the 13 GB CPU machine at batch 32 (they are measured on the GPU)
SKIP_CPU = {('PatchTST', 'Traffic'), ('PatchTST', 'ECL'), ('TiDE', 'Traffic')}


def one(spec):
    import numpy as np, torch
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
    os.chdir(REPO)
    import run, baselines
    from importlib import import_module
    dev = torch.device('cuda' if spec['device'] == 'gpu' else 'cpu')
    torch.set_num_threads(spec['threads'])
    C, H, B, T = DATASETS[spec['dataset']], 96, 32, 96
    model_name, extra = MODELS[spec['name']]
    base = model_name if model_name != 'DLTPlug' else extra.split()[1]
    bargs = baselines.args(base, spec['dataset'], H)[0] if base not in ('DLT',) else ''
    argv = (f'--task_name long_term_forecast --is_training 1 --model_id eff --model {model_name} --data custom '
            f'--seq_len {T} --label_len 48 --pred_len {H} --enc_in {C} --dec_in {C} --c_out {C} '
            f'--dlt_k 8 --dlt_L 48 --dlt_R 4 {bargs} {extra}').split()
    args = run.get_parser().parse_args(argv)
    args.batch_size = B
    x = torch.randn(B, T, C, device=dev); y = torch.randn(B, H, C, device=dev)
    xm = torch.randn(B, T, 4, device=dev); dec = torch.randn(B, 48 + H, C, device=dev); dm = torch.randn(B, 48 + H, 4, device=dev)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    if model_name == 'SMamba' and dev.type == 'cpu':
        args.mamba_impl = 'torch'                     # mamba_ssm needs CUDA
    model = import_module(f'models.{model_name}').Model(args).float().to(dev)
    if hasattr(model, 'prepare'):
        model.prepare(np.random.randn(4000, C).cumsum(0))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)

    def step():
        opt.zero_grad(); out = model(x, xm, dec, dm)
        out = out[0] if isinstance(out, tuple) else out
        loss = ((out[:, -H:] - y) ** 2).mean()
        if hasattr(model, 'extra_loss'):
            loss = loss + model.extra_loss(x, out[:, -H:], y)
        loss.backward(); opt.step()
    n_warm, n = (5, 30) if dev.type == 'cuda' else (2, spec['steps'])
    for _ in range(n_warm): step()
    sync(); t = time.time()
    for _ in range(n): step()
    sync(); train_ms = (time.time() - t) / n * 1e3
    model.eval()
    with torch.no_grad():
        for _ in range(n_warm): model(x, xm, dec, dm)
        sync(); t = time.time()
        for _ in range(n): model(x, xm, dec, dm)
        sync(); infer_ms = (time.time() - t) / n * 1e3
    mem = (torch.cuda.max_memory_allocated() - m0) / 2 ** 20 if dev.type == 'cuda' else \
        (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024
    return {'dataset': spec['dataset'], 'C': C, 'model': spec['name'], 'device': spec['device'],
            'threads': spec['threads'], 'batch': B, 'H': H, 'params': int(sum(p.numel() for p in model.parameters())),
            'train_ms': round(train_ms, 1), 'infer_ms': round(infer_ms, 1), 'mem_MB': round(mem, 1),
            'gpu': torch.cuda.get_device_name() if dev.type == 'cuda' else None, 'torch': torch.__version__,
            **({'mamba_impl': model.mamba_impl} if model_name == 'SMamba' else {})}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--steps', type=int, default=8, help='measured training steps on the CPU')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--models', nargs='+', help='default: DLT-dyn and the eight representative baselines')
    p.add_argument('--variants', action='store_true', help='also the kernel variants and the plug-in')
    p.add_argument('--out', help='default: results/p2/efficiency_<device>.jsonl')
    p.add_argument('--one', help=argparse.SUPPRESS)
    a = p.parse_args()
    a.models = a.models or list(OURS) + list(BASELINES) + (list(VARIANTS) if a.variants else [])
    a.out = a.out or str(REPO / 'results' / 'p2' / f'efficiency_{a.device}.jsonl')
    if a.one:
        print('RESULT ' + json.dumps(one(json.loads(a.one))))
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ds in a.datasets:
        for name in a.models:
            if a.device == 'cpu' and ((MODELS[name][0], ds) in SKIP_CPU or MODELS[name][0] == 'SMamba'):
                continue                              # S-Mamba: the CPU block is a slow fallback, not a measurement
            spec = {'dataset': ds, 'name': name, 'device': a.device, 'threads': a.threads, 'steps': a.steps}
            r = subprocess.run([sys.executable, __file__, '--out', a.out, '--one', json.dumps(spec)],
                               capture_output=True, text=True, env={**os.environ, 'OMP_NUM_THREADS': str(a.threads)})
            line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
            err = (r.stderr.strip().splitlines() or [''])[-1]
            if not err and r.returncode == -9:                  # SIGKILL without a traceback: the OOM killer
                err = 'killed (SIGKILL, out of memory)'
            rec = json.loads(line[0][7:]) if line else {'dataset': ds, 'model': name, 'device': a.device,
                                                         'error': err or f'exit code {r.returncode}'}
            print(json.dumps(rec), flush=True)
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
