"""Efficiency: parameters, training / inference time per batch and peak memory of DLT and the baselines.

    python scripts/efficiency/run_efficiency.py --device gpu      # on the CUDA machine -> output/efficiency/efficiency_gpu.jsonl
    python scripts/efficiency/run_efficiency.py --device cpu --threads 4        # -> output/efficiency/efficiency_cpu.jsonl
    python scripts/efficiency/run_efficiency.py --setting smamba --device gpu   # the efficiency setting of the S-Mamba
                                                                  # paper -> output/efficiency/efficiency_smamba_gpu.jsonl

Default models: DLT-dyn and the six baselines (S-Mamba, iTransformer, PatchTST, RLinear, DLinear, TiDE).
--variants adds the kernel variants of the controlled study and the plug-in.
Every (dataset, model) is measured in a fresh subprocess (clean peak-memory counter) on random data with the
dataset's shape: batch 32, input 96, horizon 96. --setting smamba follows the efficiency experiment of the S-Mamba
paper (Wang et al., Neurocomputing 2025, Fig. 5): PEMS07 (horizon 12), Electricity, Traffic and ETTm1 (horizon 96),
batch size 16. --dlt "<args>" / --name replace the DLT-dyn arguments (e.g. a tuned recipe); --tuned adds, per dataset,
DLT-dyn with the recipe chosen on the validation loss in the eff_tune rounds (scripts/efficiency/eff_tune.py), on the basis
path as trained and on the equivalent kernel path; --amp measures with torch.autocast (GPU). Baselines use the
official hyper-parameters of that dataset
(src/configs/baseline_profiles.py). Train time = forward + MSE + backward + Adam step. Memory: GPU = peak allocated by
PyTorch; CPU = growth of the peak resident set size of the process during model creation and training.
"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.paths import OUTPUT  # noqa: E402

DATASETS = {'Weather': 21, 'PEMS08': 170, 'ECL': 321, 'Traffic': 862, 'PEMS07': 883, 'ETTm1': 7}
SMAMBA_SETTING = {'PEMS07': 12, 'ECL': 96, 'Traffic': 96, 'ETTm1': 96}   # dataset: horizon (S-Mamba, Fig. 5), batch 16
# name: (model, extra args). Baselines also get their official arguments for the dataset (src/configs/baseline_profiles.py).
OURS = {'DLT-dyn': ('DLT', '--dlt_variant dltdyn --dlt_impl basis')}
BASELINES = {'S-Mamba': ('S_Mamba', ''), 'iTransformer': ('iTransformer', ''), 'PatchTST': ('PatchTST', ''),
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
TUNED = {'DLT-dyn (tuned)': 'basis', 'DLT-dyn (tuned, kernel path)': 'kernel'}   # --tuned: name -> implementation
# too slow or too large for the 13 GB CPU machine at batch 32 (they are measured on the GPU); with the batch size 16
# of --setting smamba they are attempted (a run killed for lack of memory is recorded as an error)
SKIP_CPU = {('PatchTST', 'Traffic'), ('PatchTST', 'ECL'), ('TiDE', 'Traffic')}


def one(spec):
    import numpy as np, torch
    sys.path.insert(0, str(REPO / 'scripts'))
    os.chdir(REPO)
    import run
    from src.configs import baseline_profiles
    from importlib import import_module
    dev = torch.device('cuda' if spec['device'] == 'gpu' else 'cpu')
    torch.set_num_threads(spec['threads'])
    C, H, B, T = DATASETS[spec['dataset']], spec.get('H', 96), spec.get('batch', 32), 96
    model_name, extra = MODELS.get(spec['name'], ('DLT', ''))
    if spec.get('dlt') is not None and model_name == 'DLT':
        extra = spec['dlt']
    base = model_name if model_name != 'DLTPlug' else extra.split()[1]
    bargs = baseline_profiles.args(base, spec['dataset'], H)[0] if base not in ('DLT',) else ''
    dlt_defaults = '--dlt_k 8 --dlt_L 48 --dlt_R 4' if model_name in ('DLT', 'DLTPlug') else ''   # the DLT parser only
    argv = (f'--is_training 1 --model_id eff --model {model_name} --data custom '
            f'--seq_len {T} --label_len 48 --pred_len {H} --enc_in {C} --dec_in {C} --c_out {C} '
            f'{dlt_defaults} {bargs} {extra}').split()
    args = run.parse_args(argv)
    args.batch_size = B
    x = torch.randn(B, T, C, device=dev); y = torch.randn(B, H, C, device=dev)
    xm = torch.randn(B, T, 4, device=dev); dec = torch.randn(B, 48 + H, C, device=dev); dm = torch.randn(B, 48 + H, 4, device=dev)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
    if model_name == 'S_Mamba' and dev.type == 'cpu':
        args.mamba_impl = 'torch'                     # mamba_ssm needs CUDA
    model = import_module(f'src.model.{model_name}').Model(args).float().to(dev)
    if hasattr(model, 'prepare'):
        model.prepare(np.random.randn(4000, C).cumsum(0))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)

    amp = bool(spec.get('amp')) and dev.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if amp else None

    def step():
        opt.zero_grad()
        with torch.autocast('cuda', enabled=amp):
            out = model(x, xm, dec, dm)
            out = out[0] if isinstance(out, tuple) else out
            loss = ((out[:, -H:] - y) ** 2).mean()
            if hasattr(model, 'extra_loss'):
                loss = loss + model.extra_loss(x, out[:, -H:], y)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
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
            'amp': amp, **({'dlt': extra} if model_name == 'DLT' else {}),
            'train_ms': round(train_ms, 1), 'infer_ms': round(infer_ms, 1), 'mem_MB': round(mem, 1),
            'gpu': torch.cuda.get_device_name() if dev.type == 'cuda' else None, 'torch': torch.__version__,
            **({'mamba_impl': model.mamba_impl} if model_name == 'S_Mamba' else {})}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--steps', type=int, default=8, help='measured training steps on the CPU')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS))
    p.add_argument('--models', nargs='+', help='default: DLT-dyn and the eight representative baselines')
    p.add_argument('--variants', action='store_true', help='also the kernel variants and the plug-in')
    p.add_argument('--out', help='default: output/efficiency/efficiency_<device>.jsonl (efficiency_smamba_<device>.jsonl)')
    p.add_argument('--setting', default='p2', choices=['p2', 'smamba'],
                   help='p2: batch 32, horizon 96 on --datasets; smamba: the setting of the S-Mamba paper (Fig. 5)')
    p.add_argument('--batch', type=int, help='batch size (default 32; 16 with --setting smamba)')
    p.add_argument('--dlt', help='DLT arguments of the measured DLT model (default: DLT-dyn, basis path)')
    p.add_argument('--name', help='record name of the model measured with --dlt (default DLT-dyn)')
    p.add_argument('--amp', action='store_true', help='torch.autocast mixed precision (GPU only)')
    p.add_argument('--tuned', action='store_true',
                   help='--setting smamba: add DLT-dyn with the recipe selected in the eff_tune rounds, per dataset '
                        '(basis and kernel path); with --models, list the names "DLT-dyn (tuned)" / '
                        '"DLT-dyn (tuned, kernel path)" to measure them')
    p.add_argument('--one', help=argparse.SUPPRESS)
    a = p.parse_args()
    explicit = bool(a.models)
    a.models = a.models or list(OURS) + list(BASELINES) + (list(VARIANTS) if a.variants else [])
    tuned = {}
    if a.tuned:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import eff_tune
        tuned = {ds: eff_tune.recipe(r) for ds, (t, r) in eff_tune.selected().items()}
        print('tuned recipes:', json.dumps(tuned), flush=True)
        if not explicit:
            a.models = a.models + list(TUNED)
    if a.dlt is not None:
        a.models = [a.name or 'DLT-dyn' if m == 'DLT-dyn' else m for m in a.models]
    known = set(MODELS) | set(TUNED) | ({a.name or 'DLT-dyn'} if a.dlt is not None else set())
    if any(m not in known for m in a.models):
        p.error('unknown models {}; choose from {}'.format([m for m in a.models if m not in known], sorted(known)))
    smamba = a.setting == 'smamba'
    a.batch = a.batch or (16 if smamba else 32)
    if smamba and a.datasets == list(DATASETS):
        a.datasets = list(SMAMBA_SETTING)
    a.out = a.out or str(OUTPUT / 'efficiency' / f'efficiency_{"smamba_" if smamba else ""}{a.device}.jsonl')
    if a.one:
        print('RESULT ' + json.dumps(one(json.loads(a.one))))
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ds in a.datasets:
        for name in a.models:
            if name in TUNED and ds not in tuned:
                continue                              # no tuning record for this dataset (or --tuned not given)
            base_model = MODELS.get(name, ('DLT', ''))[0]
            if a.device == 'cpu' and (((base_model, ds) in SKIP_CPU and a.batch > 16) or base_model == 'S_Mamba'):
                continue                              # S-Mamba: the CPU block is a slow fallback, not a measurement
            spec = {'dataset': ds, 'name': name, 'device': a.device, 'threads': a.threads, 'steps': a.steps,
                    'batch': a.batch, 'H': SMAMBA_SETTING.get(ds, 96) if smamba else 96, 'amp': a.amp,
                    'dlt': a.dlt if a.dlt is not None and name == (a.name or 'DLT-dyn') else
                    tuned[ds].replace('--dlt_impl basis', f'--dlt_impl {TUNED[name]}') if name in TUNED else None}
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
