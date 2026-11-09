"""Job lists for the P1 / P2 experiments (Time-Series-Library, one JSON line per run).

    python scripts/exps.py <suite> --data <dir> --work <dir> [--device cpu|gpu] [--seeds 2021 2022 2023]
                           [--dlt "<extra --dlt_* args for the proposed model>"]

<dir> must contain the dataset files (ETTh1.csv ... traffic.csv, PEMS04.npz, PEMS08.npz). The job list is
written to <work>/jobs_<suite>.txt (one shell command per line, longest first); every job appends its JSON
record to <work>/results_<suite>.jsonl and logs to <work>/logs/<suite>/. Run a list with
    bash scripts/run_queue.sh <work>/jobs_<suite>.txt [parallel jobs]
from anywhere (the runner changes into tslib/). On a GPU use --device gpu and run one job per GPU.

Suites
  pems12       P1  PEMS04/08, input 96, horizon 12, TimeMixer protocol (L1 training loss, validation MAE and
                   test MAE / MAPE / RMSE on the original scale): DLT-dyn, DLT, CI (+ DLT-dyn trained with MSE)
  long_small   P1  DLT-dyn on ETTh1/2, ETTm1/2, Exchange, Weather, horizons 96/192/336/720
  long_large   P1  DLT-dyn on Electricity and Traffic, horizons 96/192/336/720 (GPU recommended)
  plugin       P1  DLinear / PatchTST / iTransformer, alone and + DLT-dyn (DLTPlug) on ETTh1, Weather, PEMS08-12
                   (PatchTST on PEMS08 only with --device gpu; PatchTST uses the configurations of its paper)
  plugin_large P1  the same on Electricity / Traffic (all horizons) and PEMS04-12 (GPU)
  baselines    P1  iTransformer, PatchTST, DLinear, TimeMixer, LagTS (small data only) on the long-term
                   benchmark with the official TSLib hyper-parameters (GPU recommended)
  sensitivity  P2  DLT-dyn, one factor at a time: L in {12,24,96}, R in {2,8}, k in {4,16} (default 48/4/8)
                   on ETTh1-96, Weather-96, PEMS08-96
  neighbours   P2  DLT-dyn / DLT with random or all neighbours instead of the CCF top-k (Weather-96, PEMS08-96)
"""
import argparse, os, re, json, itertools
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSLIB = REPO / 'tslib'
HORIZONS = (96, 192, 336, 720)

# name: (--data, file, variables, --freq, train windows (for cost estimates))
DATASETS = {
    'ETTh1':    ('ETTh1', 'ETTh1.csv', 7, 'h', 8545),
    'ETTh2':    ('ETTh2', 'ETTh2.csv', 7, 'h', 8545),
    'ETTm1':    ('ETTm1', 'ETTm1.csv', 7, 't', 34465),
    'ETTm2':    ('ETTm2', 'ETTm2.csv', 7, 't', 34465),
    'Weather':  ('custom', 'weather.csv', 21, 't', 36792),
    'Exchange': ('custom', 'exchange_rate.csv', 8, 'd', 5120),
    'ECL':      ('custom', 'electricity.csv', 321, 'h', 18317),
    'Traffic':  ('custom', 'traffic.csv', 862, 'h', 12185),
    'PEMS04':   ('PEMS', 'PEMS04.npz', 307, 't', 10088),
    'PEMS08':   ('PEMS', 'PEMS08.npz', 170, 't', 10606),
}
# the proposed model and its training set-up (identical to the controlled study, P0)
DLT_DEFAULT = '--dlt_variant dltdyn --dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl basis'
DLT_TRAIN = '--learning_rate 0.003 --batch_size 32 --train_epochs 10 --patience 3'
PEMS_PROTOCOL = '--loss MAE --inverse --vali_inverse'          # TimeMixer / MillGNN short-term protocol


def official_args(model, data_file, H):
    """Hyper-parameters of a baseline from the official TSLib script (scripts/long_term_forecast/*_script)."""
    keys = ('e_layers', 'd_layers', 'factor', 'd_model', 'd_ff', 'n_heads', 'batch_size', 'learning_rate',
            'train_epochs', 'patience', 'dropout', 'down_sampling_layers', 'down_sampling_window',
            'down_sampling_method', 'channel_independence', 'top_k', 'moving_avg', 'label_len', 'use_norm')
    for path in sorted((TSLIB / 'scripts' / 'long_term_forecast').glob(f'*_script/{model}*.sh')):
        txt = re.sub(r'\\\s*\n', ' ', path.read_text())
        var = dict(re.findall(r'^(\w+)=(\S+)', txt, re.M))
        for line in txt.splitlines():
            if 'run.py' not in line or line.lstrip().startswith('#'):
                continue
            toks, d, i = line.split(), {}, 0
            while i < len(toks):
                if toks[i].startswith('--'):
                    val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith('--') else ''
                    if val.startswith('$'):
                        val = var.get(val[1:].strip('{}'), val)
                    d[toks[i][2:]] = val
                    i += 2 if val else 1
                else:
                    i += 1
            if d.get('model', '').strip('$') in (model, 'model_name') and d.get('data_path') == data_file \
                    and str(d.get('pred_len')) == str(H):
                return ' '.join(f'--{k} {d[k]}' for k in keys if k in d and d[k]), path.name
    return None, None


# where TSLib has no official script: settings of the original papers' repositories / TSLib defaults
FALLBACK = {
    'iTransformer': {'ETTh1': '--e_layers 2 --d_model 256 --d_ff 256', 'ETTm1': '--e_layers 2 --d_model 128 --d_ff 128',
                     'ETTm2': '--e_layers 2 --d_model 128 --d_ff 128', 'Exchange': '--e_layers 2 --d_model 128 --d_ff 128',
                     'Weather': '--e_layers 3 --d_model 512 --d_ff 512',
                     'PEMS04': '--e_layers 4 --d_model 1024 --d_ff 1024 --learning_rate 0.0005',
                     'PEMS08': '--e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.001 --batch_size 16'},
    'PatchTST': {'PEMS04': '--e_layers 2 --d_model 128 --d_ff 256 --n_heads 16', 'PEMS08': '--e_layers 2 --d_model 128 --d_ff 256 --n_heads 16',
                 'ETTh1': '--e_layers 1 --n_heads 2'},
    'DLinear': {},
    'TimeMixer': {'Exchange': '--e_layers 2 --d_model 16 --d_ff 32 --learning_rate 0.01 --down_sampling_layers 3 --down_sampling_window 2',
                  'PEMS04': '--e_layers 5 --d_model 128 --d_ff 256 --learning_rate 0.003 --down_sampling_layers 1 --down_sampling_window 2 --channel_independence 0 --use_norm 0',
                  'PEMS08': '--e_layers 5 --d_model 128 --d_ff 256 --learning_rate 0.003 --down_sampling_layers 1 --down_sampling_window 2 --channel_independence 0 --use_norm 0'},
    'LagTS': {'ETTh1': '--e_layers 2 --d_model 128 --d_ff 128 --learning_rate 0.0001',
              'ETTh2': '--e_layers 2 --d_model 128 --d_ff 128 --learning_rate 0.0001',
              'Exchange': '--e_layers 2 --d_model 128 --d_ff 128 --learning_rate 0.0001',
              'Weather': '--e_layers 3 --d_model 512 --d_ff 512 --learning_rate 0.0001'},
}
MODEL_EXTRA = {'TimeMixer': '--down_sampling_method avg', 'PatchTST': '', 'iTransformer': '', 'DLinear': '', 'LagTS': ''}


# PatchTST in the plug-in study: the configurations of the PatchTST paper (TSLib's scripts use d_model 512 /
# d_ff 2048, which is ~8x more expensive and not what the paper reports); the training budget stays TSLib's
PATCHTST_PAPER = {'small': '--e_layers 3 --d_model 16 --d_ff 128 --n_heads 4 --dropout 0.3',
                  'large': '--e_layers 3 --d_model 128 --d_ff 256 --n_heads 16 --dropout 0.2'}


def plugin_backbone_args(model, ds, H):
    args, src = backbone_args(model, ds, H)
    if model == 'PatchTST':
        keep = ' '.join(t for t in re.findall(r'--(?:batch_size|train_epochs|patience|learning_rate) \S+', args))
        size = 'small' if ds.startswith('ETT') or ds == 'Exchange' else 'large'
        args, src = f'{PATCHTST_PAPER[size]} {keep}'.strip(), 'PatchTST paper'
    return args, src


def backbone_args(model, ds, H):
    args, src = official_args(model, DATASETS[ds][1], H)
    if args is None:
        args, src = FALLBACK.get(model, {}).get(ds, ''), 'fallback'
    return (args + ' ' + MODEL_EXTRA.get(model, '')).strip(), src


class Gen:
    def __init__(self, a, suite):
        self.a, self.suite, self.jobs = a, suite, []
        self.work = os.path.abspath(a.work)
        os.makedirs(f'{self.work}/logs/{suite}', exist_ok=True)

    def add(self, name, model, ds, H, seed, args, cost):
        dflag, dfile, C, freq, _ = DATASETS[ds]
        dev = '--no_use_gpu --num_workers 0' if self.a.device == 'cpu' else '--num_workers 4'
        label = '0' if ds.startswith('PEMS') else '48'
        cmd = (f'python -u run.py --task_name long_term_forecast --is_training 1 --model {model} '
               f'--model_id {self.suite}_{ds}_{H} --root_path {os.path.abspath(self.a.data)} --data {dflag} '
               f'--data_path {dfile} --freq {freq} --features M --seq_len 96 --label_len {label} --pred_len {H} '
               f'--enc_in {C} --dec_in {C} --c_out {C} --des {name} --fix_seed {seed} {args} {dev} '
               f'--checkpoints {self.work}/ckpt --no_epoch_test --no_save_preds '
               f'--result_jsonl {self.work}/results_{self.suite}.jsonl '
               f'> {self.work}/logs/{self.suite}/{name}_s{seed}.log 2>&1')
        self.jobs.append((cost, f'{name}_s{seed}', cmd))

    def write(self):
        # seed-major order (every configuration gets its first seed early), longest first inside a seed
        by_seed = {}
        for j in self.jobs:
            by_seed.setdefault(j[1].rsplit('_s', 1)[1], []).append(j)
        ordered = [j for s in sorted(by_seed) for j in sorted(by_seed[s], key=lambda j: -j[0])]
        path = f'{self.work}/jobs_{self.suite}.txt'
        open(path, 'w').write('\n'.join(j[2] for j in ordered) + '\n')
        hours = sum(j[0] for j in ordered) / 3600
        print(f'{self.suite}: {len(ordered)} jobs -> {path}; rough CPU estimate {hours:.1f} core-hours')


def cost(ds, rel=1.0):
    """very rough single-thread CPU seconds of a 10-epoch DLT-dyn run (only used to order the jobs)"""
    _, _, C, _, n = DATASETS[ds]
    return rel * 10 * n / 32 * (0.0007 * min(C, 400) + 0.002) * 2.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('suite')
    p.add_argument('--data', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    p.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023])
    p.add_argument('--dlt', default='', help='extra --dlt_* arguments of the proposed model (e.g. a gate)')
    a = p.parse_args()
    g = Gen(a, a.suite)
    dlt = f'{DLT_DEFAULT} {a.dlt}'.strip()
    S = a.seeds
    if a.suite == 'pems12':
        for ds, seed in itertools.product(('PEMS04', 'PEMS08'), S):
            for name, args, rel in (('dltdyn', f'{dlt} {DLT_TRAIN} {PEMS_PROTOCOL}', 1.0),
                                    ('dlt', f'--dlt_variant dlt --dlt_k 8 --dlt_L 48 --dlt_R 4 {DLT_TRAIN} {PEMS_PROTOCOL}', 0.9),
                                    ('ci', f'--dlt_variant ci {DLT_TRAIN} {PEMS_PROTOCOL}', 0.1),
                                    ('dltdyn_mse', f'{dlt} {DLT_TRAIN} --inverse --vali_inverse', 1.0)):
                g.add(f'{ds}_{name}', 'DLT', ds, 12, seed, args, cost(ds, rel))
    elif a.suite in ('long_small', 'long_large'):
        sets = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather') if a.suite == 'long_small' else ('ECL', 'Traffic')
        for ds, H, seed in itertools.product(sets, HORIZONS, S):
            g.add(f'{ds}_{H}_dltdyn', 'DLT', ds, H, seed, f'{dlt} {DLT_TRAIN}', cost(ds))
    elif a.suite in ('plugin', 'plugin_large'):
        sets = (('ETTh1', 96), ('Weather', 96), ('PEMS08', 12)) if a.suite == 'plugin' else \
               tuple((d, H) for d in ('ECL', 'Traffic', 'PEMS04') for H in (HORIZONS if d != 'PEMS04' else (12,)))
        for (ds, H), base, seed in itertools.product(sets, ('DLinear', 'PatchTST', 'iTransformer'), S):
            if a.suite == 'plugin' and a.device == 'cpu' and (base, ds) == ('PatchTST', 'PEMS08'):
                continue                                       # ~5 h per run on one CPU thread: GPU only
            bargs, _ = plugin_backbone_args(base, ds, H)
            proto = PEMS_PROTOCOL if ds.startswith('PEMS') else ''
            rel = {'DLinear': 0.3, 'PatchTST': 3.0, 'iTransformer': 1.5}[base]
            g.add(f'{ds}_{H}_{base}_alone', 'DLTPlug', ds, H, seed,
                  f'--plug_base {base} --dlt_variant ci {bargs} {proto}', cost(ds, rel))
            g.add(f'{ds}_{H}_{base}_dlt', 'DLTPlug', ds, H, seed,
                  f'--plug_base {base} {dlt} --dlt_lr 0.003 {bargs} {proto}', cost(ds, rel + 1))
    elif a.suite == 'baselines':
        for ds, H, seed in itertools.product(('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'ECL', 'Traffic'),
                                             HORIZONS, S):
            for base in ('iTransformer', 'PatchTST', 'DLinear', 'TimeMixer', 'LagTS'):
                if base == 'LagTS' and DATASETS[ds][2] > 21:
                    continue                                   # O(N^2) pairs: the paper uses N <= 21
                bargs, _ = backbone_args(base, ds, H)
                g.add(f'{ds}_{H}_{base}', base, ds, H, seed, bargs, cost(ds, 2.0))
    elif a.suite == 'sensitivity':
        for ds, H in (('ETTh1', 96), ('Weather', 96), ('PEMS08', 96)):
            for name, args, rel in (('L12', '--dlt_L 12', 0.6), ('L24', '--dlt_L 24', 0.7), ('L96', '--dlt_L 96', 1.8),
                                    ('R2', '--dlt_R 2', 0.9), ('R8', '--dlt_R 8', 1.3),
                                    ('k4', '--dlt_k 4', 0.6), ('k16', '--dlt_k 16', 1.9)):
                for seed in S:
                    g.add(f'{ds}_{H}_{name}', 'DLT', ds, H, seed, f'{dlt} {DLT_TRAIN} {args}', cost(ds, rel))
    elif a.suite == 'neighbours':
        for ds, H in (('Weather', 96), ('PEMS08', 96)):
            for v, nb, impl, rel in (('dltdyn', 'random', 'basis', 1.0), ('dltdyn', 'all', 'dense', 4.0),
                                     ('dlt', 'random', 'kernel', 0.9), ('dlt', 'all', 'dense', 2.0)):
                for seed in S:
                    g.add(f'{ds}_{H}_{v}_{nb}', 'DLT', ds, H, seed,
                          f'--dlt_variant {v} --dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl {impl} '
                          f'--dlt_neighbours {nb} {DLT_TRAIN}', cost(ds, rel))
    else:
        raise SystemExit(f'unknown suite {a.suite}')
    g.write()


if __name__ == '__main__':
    main()
