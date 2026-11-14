"""Job lists for the P1 / P2 experiments (Time-Series-Library, one JSON line per run).

    python scripts/exps.py <suite> [--device cpu|gpu] [--seeds 2021 2022 2023]
                           [--dlt "<extra --dlt_* args for the proposed model>"]

Datasets are read from dataset/ and everything is written to work/ (fixed, see scripts/paths.py): the job list
work/jobs_<suite>.txt (one shell command per line, longest first), one JSON record per run in
work/results_<suite>.jsonl, logs in work/logs/<suite>/ and checkpoints in work/ckpt/. Run a list with
    bash scripts/run_queue.sh <suite> [parallel jobs]          CPU
    bash scripts/gpu/run_gpu_queue.sh <suite> [gpu ids]        GPU (--device gpu when generating)

Suites
  pems12       P1  PEMS04/08, input 96, horizon 12, TimeMixer protocol (L1 training loss, validation MAE and
                   test MAE / MAPE / RMSE on the original scale): DLT-dyn, DLT, CI (+ DLT-dyn trained with MSE)
  long_small   P1  DLT-dyn on ETTh1/2, ETTm1/2, Exchange, Weather, horizons 96/192/336/720
  long_large   P1  DLT-dyn on Electricity and Traffic, horizons 96/192/336/720 (GPU recommended)
  plugin       P1  DLinear / PatchTST / iTransformer, alone and + DLT-dyn (DLTPlug) on ETTh1, Weather, PEMS08-12
                   (PatchTST on PEMS08 only with --device gpu; PatchTST uses the configurations of its paper)
  plugin_large P1  the same on Electricity / Traffic (all horizons) and PEMS04-12 (GPU)
  baselines    P1  the eight representative baselines (BASELINES below: Informer, TimesNet, DLinear, PatchTST,
                   iTransformer, TimeMixer, TimeXer, S-Mamba) on the long-term benchmark with their official
                   hyper-parameters (GPU recommended). --models LagTS adds our LagTS re-implementation (<= 21 variables)
  sensitivity  P2  DLT-dyn, one factor at a time: L in {12,24,96}, R in {2,8}, k in {4,16} (default 48/4/8)
                   on ETTh1-96, Weather-96, PEMS08-96
  neighbours   P2  DLT-dyn / DLT with random or all neighbours instead of the CCF top-k (Weather-96, PEMS08-96)
  ci_long      P1  the channel-independent linear model (DLT without the cross branch) on the long-term benchmark
  allnb        P1  DLT-dyn / DLT with all other variables as neighbours (dense path): PEMS04/08-12, Weather and
                   Electricity (DLT-dyn there only with --device gpu), iTransformer + DLT-dyn on PEMS08-12
  plugin_probe P1  diagnosis of the plug-in on Traffic-96: + DLT-dyn with the backbone's learning rate, with a
                   static gate, and with both (PatchTST, iTransformer; use --seeds 2021 for six runs)
  plugin_all   P1  every representative backbone (BASELINES) alone and + DLT-dyn (DLTPlug, branch lr 3e-3) on the
                   long-term benchmark and Solar (horizons 96/192/336/720) and on PEMS03/04/07/08 (horizons
                   12/24/48/96, protocol of iTransformer and S-Mamba: MSE loss, metrics on the normalised scale).
                   Settings: the official ones (TSLib scripts; for Solar and PEMS those of the iTransformer, S-Mamba
                   and TimeMixer repositories, REPO_OFFICIAL), else the documented fallbacks (FALLBACK_FROM);
                   PatchTST: its paper's. GPU. The table: python scripts/p1/plugin_table.py

--models / --datasets restrict the baselines, ci_long and plugin_all suites, e.g. the CPU-feasible subset
    python scripts/exps.py baselines --models DLinear iTransformer TimeMixer --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange
--dlt "<args>" adds to or overrides the DLT-dyn arguments, e.g. --dlt "--dlt_gate static" or --dlt "--dlt_lr 0"
(plugin_all: the branch then trains with the backbone's learning rate). --list NAME writes the jobs to
work/jobs_NAME.txt instead of work/jobs_<suite>.txt, so that a second list can run next to a running queue.
"""
import argparse, os, re, json, itertools, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

REPO = paths.REPO
TSLIB = REPO
# The representative baselines of every comparison (accuracy and efficiency): one model per family.
#   Informer (classic Transformer), TimesNet (CNN), DLinear (linear), PatchTST (channel-independent Transformer),
#   iTransformer (variate tokens), TimeMixer (multi-scale MLP), TimeXer (patch + variate tokens), S-Mamba (state space)
BASELINES = ('Informer', 'TimesNet', 'DLinear', 'PatchTST', 'iTransformer', 'TimeMixer', 'TimeXer', 'SMamba')
HORIZONS = (96, 192, 336, 720)
LONG_SETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'ECL', 'Traffic')
PEMS_SETS = ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')
PEMS_HORIZONS = (12, 24, 48, 96)                               # iTransformer / S-Mamba protocol, input 96
PLUGIN_ALL_SETS = LONG_SETS + ('Solar',) + PEMS_SETS           # the datasets of plugin_all

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
    'Solar':    ('Solar', 'solar_AL.txt', 137, 't', 36601),
    'PEMS03':   ('PEMS', 'PEMS03.npz', 358, 't', 15617),
    'PEMS04':   ('PEMS', 'PEMS04.npz', 307, 't', 10088),
    'PEMS07':   ('PEMS', 'PEMS07.npz', 883, 't', 16827),
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
            'down_sampling_method', 'channel_independence', 'top_k', 'moving_avg', 'label_len', 'use_norm',
            'd_state')
    for path in sorted((TSLIB / 'scripts' / 'long_term_forecast').glob(f'*_script/{model}*.sh')):
        txt = re.sub(r'\\[ \t]*\n', ' ', path.read_text())      # a backslash continues one line only
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
              'ETTm1': '--e_layers 2 --d_model 128 --d_ff 128 --learning_rate 0.0001',
              'ETTm2': '--e_layers 2 --d_model 128 --d_ff 128 --learning_rate 0.0001',
              'Exchange': '--e_layers 2 --d_model 128 --d_ff 128 --learning_rate 0.0001',
              'Weather': '--e_layers 3 --d_model 512 --d_ff 512 --learning_rate 0.0001'},
    # official S-Mamba scripts for PEMS at horizon 12 (scripts/multivariate_forecasting/PEMS/S_Mamba_0x.sh)
    'SMamba': {'PEMS04': '--e_layers 4 --d_model 1024 --d_ff 1024 --learning_rate 0.0003 --use_norm 0',
               'PEMS08': '--e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.001 --use_norm 1'},
}
# no official configuration for this dataset: the configuration of another dataset at the same horizon.
# Informer has one generic setting in every TSLib script; TimeXer was not tuned on Exchange (small-data setting of ETTh1).
# Solar and PEMS (many variables) take the Electricity configuration where a model has none of its own.
FALLBACK_FROM = {'Informer': {d: 'ETTh1' for d in ('ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'Traffic',
                                                    'Solar', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')},
                 'TimeXer': {'Exchange': 'ETTh1', **{d: 'ECL' for d in ('Solar', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')}},
                 'TimesNet': {d: 'ECL' for d in ('Solar', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')}}
MODEL_EXTRA = {'TimeMixer': '--down_sampling_method avg'}

# Official settings of the models' own repositories for the datasets that TSLib has no script for, per horizon:
# scripts/multivariate_forecasting/{PEMS/*_0x.sh, SolarEnergy/*.sh} of iTransformer and S-Mamba, and of TimeMixer
# scripts/long_term_forecast/Solar_script/TimeMixer_unify.sh and scripts/short_term_forecast/PEMS/TimeMixer.sh (its
# only PEMS setting, horizon 12, used here for every horizon). Unset values are the defaults there and here
# (lr 1e-4, batch 32, 10 epochs, patience 3, use_norm 1, d_state 32). Used by plugin_all. The older PEMS-12 suites
# (pems12 protocol) keep FALLBACK above, which for iTransformer differs: PEMS04 without --use_norm 0, PEMS08 with
# lr 1e-3 / batch 16 (the settings of its horizons 48 and 96).
def _per_h(short, long):                                       # one setting for 12 / 24, another for 48 / 96
    return {12: short, 24: short, 48: long, 96: long}


_TM_PEMS = ('--e_layers 5 --d_model 128 --d_ff 256 --batch_size 32 --learning_rate 0.003 --train_epochs 10 '
            '--patience 10 --use_norm 0 --down_sampling_layers 1 --down_sampling_window 2 --channel_independence 0')
REPO_OFFICIAL = {
    'iTransformer': {
        'PEMS03': dict.fromkeys(PEMS_HORIZONS, '--e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.001'),
        'PEMS04': dict.fromkeys(PEMS_HORIZONS, '--e_layers 4 --d_model 1024 --d_ff 1024 --learning_rate 0.0005 --use_norm 0'),
        'PEMS07': _per_h('--e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.001 --use_norm 0',
                         '--e_layers 4 --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.001 --use_norm 0'),
        'PEMS08': _per_h('--e_layers 2 --d_model 512 --d_ff 512 --use_norm 1',
                         '--e_layers 4 --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.001 --use_norm 0'),
        'Solar': dict.fromkeys(HORIZONS, '--e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.0005')},
    'SMamba': {
        'PEMS03': {12: '--e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.001 --train_epochs 5',
                   24: '--e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.001',
                   48: '--e_layers 5 --d_model 512 --d_ff 512 --learning_rate 0.0001',
                   96: '--e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.0015'},
        'PEMS04': dict.fromkeys(PEMS_HORIZONS, '--e_layers 4 --d_model 1024 --d_ff 1024 --learning_rate 0.0003 --use_norm 0'),
        'PEMS07': {12: '--e_layers 2 --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.0007 --use_norm 0',
                   24: '--e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.0007 --use_norm 0',
                   48: '--e_layers 4 --d_model 512 --d_ff 512 --learning_rate 0.0005 --use_norm 0',
                   96: '--e_layers 4 --d_model 512 --d_ff 512 --batch_size 16 --learning_rate 0.0005 --use_norm 0'},
        'PEMS08': {12: '--e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.001 --use_norm 1',
                   24: '--e_layers 2 --d_model 512 --d_ff 512 --learning_rate 0.0007 --use_norm 1',
                   **dict.fromkeys((48, 96), '--e_layers 4 --d_model 256 --d_ff 256 --batch_size 16 '
                                             '--learning_rate 0.001 --use_norm 1')},
        'Solar': dict.fromkeys(HORIZONS, '--e_layers 2 --d_model 512 --d_ff 512')},
    'TimeMixer': {
        **{d: dict.fromkeys(PEMS_HORIZONS, _TM_PEMS) for d in PEMS_SETS},
        'Solar': dict.fromkeys(HORIZONS, '--e_layers 3 --d_model 512 --d_ff 2048 --batch_size 32 --learning_rate 0.001 '
                                         '--train_epochs 10 --patience 3 --use_norm 0 --down_sampling_layers 2 '
                                         '--down_sampling_window 2 --channel_independence 0')},
}
# relative cost of a backbone (only orders the job lists, longest first)
REL_COST = {'Informer': 1.0, 'TimesNet': 6.0, 'DLinear': 0.3, 'PatchTST': 3.0, 'iTransformer': 1.5, 'TimeMixer': 2.0,
            'TimeXer': 3.0, 'SMamba': 1.5}


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


def recipe_tag(extra):
    """short name of extra --dlt arguments: a changed recipe gets its own job names (dlt_<tag>), so that runs of the
    default recipe are neither skipped for nor mixed with it"""
    return re.sub(r'[^A-Za-z0-9.]+', '-', re.sub(r'--(dlt_)?', '', extra)).strip('-')


def plugin_all_args(model, ds, H):
    """Backbone settings of plugin_all: those of the plug-in study, on Solar and PEMS the official ones of the
    models' repositories (REPO_OFFICIAL) where they exist."""
    if H in REPO_OFFICIAL.get(model, {}).get(ds, {}):
        return (REPO_OFFICIAL[model][ds][H] + ' ' + MODEL_EXTRA.get(model, '')).strip(), f'{model} repository'
    return plugin_backbone_args(model, ds, H)


def backbone_args(model, ds, H):
    args, src = official_args(model, DATASETS[ds][1], H)
    if args is None and ds in FALLBACK_FROM.get(model, {}):
        other = FALLBACK_FROM[model][ds]
        args, src = official_args(model, DATASETS[other][1], H if H in HORIZONS else 96)
        src = f'{src} ({other} configuration)'
    if args is None:
        args, src = FALLBACK.get(model, {}).get(ds, ''), 'fallback'
    return (args + ' ' + MODEL_EXTRA.get(model, '')).strip(), src


class Gen:
    def __init__(self, a, suite):
        self.a, self.suite, self.jobs, self.files = a, suite, [], set()
        self.list = a.list or suite                             # name of the job list (results stay per suite)
        self.work = str(paths.WORK)
        os.makedirs(f'{self.work}/logs/{suite}', exist_ok=True)

    def add(self, name, model, ds, H, seed, args, cost, label=None):
        dflag, dfile, C, freq, _ = DATASETS[ds]
        dev = '--no_use_gpu --num_workers 0' if self.a.device == 'cpu' else '--num_workers 4'
        label = str(label) if label is not None else ('0' if ds.startswith('PEMS') else '48')
        cmd = (f'python -u run.py --task_name long_term_forecast --is_training 1 --model {model} '
               f'--model_id {self.suite}_{ds}_{H} --root_path {paths.root_of(dfile)} --data {dflag} '
               f'--data_path {dfile} --freq {freq} --features M --seq_len 96 --label_len {label} --pred_len {H} '
               f'--enc_in {C} --dec_in {C} --c_out {C} --des {name} --fix_seed {seed} {args} {dev} '
               f'--checkpoints {self.work}/ckpt --no_epoch_test --no_save_preds '
               f'--result_jsonl {self.work}/results_{self.suite}.jsonl '
               f'> {self.work}/logs/{self.suite}/{name}_s{seed}.log 2>&1')
        self.jobs.append((cost, f'{name}_s{seed}', cmd))
        self.files.add(dfile)

    def write(self):
        # seed-major order (every configuration gets its first seed early), longest first inside a seed
        by_seed = {}
        for j in self.jobs:
            by_seed.setdefault(j[1].rsplit('_s', 1)[1], []).append(j)
        ordered = [j for s in sorted(by_seed) for j in sorted(by_seed[s], key=lambda j: -j[0])]
        path = f'{self.work}/jobs_{self.list}.txt'
        open(path, 'w').write('\n'.join(j[2] for j in ordered) + '\n')
        hours = sum(j[0] for j in ordered) / 3600
        print(f'{self.list}: {len(ordered)} jobs -> {os.path.relpath(path, paths.REPO)}; rough CPU estimate {hours:.1f} core-hours')
        for f in sorted(self.files):
            if not paths.exists(f):
                print(f'  note: {paths.path_of(f)} is missing: ' + (
                    'Time-Series-Library will download it from Hugging Face' if f.endswith('.csv') else
                    'these jobs will fail; bash scripts/download_data.sh fetches it'))


def cost(ds, rel=1.0):
    """very rough single-thread CPU seconds of a 10-epoch DLT-dyn run (only used to order the jobs)"""
    _, _, C, _, n = DATASETS[ds]
    return rel * 10 * n / 32 * (0.0007 * min(C, 400) + 0.002) * 2.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('suite')
    p.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    p.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023])
    p.add_argument('--dlt', default='', help='extra --dlt_* arguments of the proposed model (e.g. a gate)')
    p.add_argument('--list', help='name of the job list (work/jobs_<list>.txt; default: the suite), e.g. to run a '
                                  'second list of the same suite next to a running queue; results still go to '
                                  'work/results_<suite>.jsonl')
    p.add_argument('--models', nargs='+', help='baselines / plugin_all / plugin_probe: only these models')
    p.add_argument('--datasets', nargs='+', help='baselines / ci_long / plugin_all / plugin_probe / allnb: only these datasets')
    a = p.parse_args()
    unknown = sorted(set(a.datasets or ()) - set(DATASETS))
    if unknown:
        raise SystemExit(f'unknown dataset(s) {unknown}; choose from {sorted(DATASETS)}')
    unknown = sorted(m for m in a.models or () if not (REPO / 'models' / f'{m}.py').exists())
    if unknown:
        raise SystemExit(f'unknown model(s) {unknown}: no file models/<name>.py')
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
    elif a.suite == 'plugin_all':
        # each backbone alone and + DLT-dyn on every dataset and horizon. Both arms go through DLTPlug ("alone":
        # --dlt_variant ci, which builds the backbone only and is identical to running it directly). label_len 48 on
        # PEMS too (iTransformer / S-Mamba protocol; only Informer reads it). --dlt_lr comes first so that --dlt can
        # override it.
        arm = 'dlt' + (f'_{recipe_tag(a.dlt)}' if a.dlt else '')
        for ds, base, seed in itertools.product(a.datasets or PLUGIN_ALL_SETS, a.models or BASELINES, S):
            for H in PEMS_HORIZONS if ds in PEMS_SETS else HORIZONS:
                bargs, _ = plugin_all_args(base, ds, H)
                rel = REL_COST.get(base, 2.0)
                g.add(f'{ds}_{H}_{base}_alone', 'DLTPlug', ds, H, seed,
                      f'--plug_base {base} --dlt_variant ci {bargs}', cost(ds, rel), label=48)
                g.add(f'{ds}_{H}_{base}_{arm}', 'DLTPlug', ds, H, seed,
                      f'--plug_base {base} --dlt_lr 0.003 {dlt} {bargs}', cost(ds, rel + 1), label=48)
    elif a.suite == 'plugin_probe':
        # Why does the DLT-dyn branch hurt PatchTST / iTransformer on Traffic (GPU tier 1c)? Three variants of the
        # "+ DLT-dyn" arm at horizon 96, to be compared with the finished alone / default arms of plugin_large:
        #   samelr       the branch trains with the learning rate of the backbone (no --dlt_lr)
        #   gate         the branch starts at 4.7 % strength (--dlt_gate static); branch lr 3e-3 as before
        #   gate_samelr  both
        for ds, seed in itertools.product(a.datasets or ('Traffic',), S):
            for base in a.models or ('PatchTST', 'iTransformer'):
                bargs, _ = plugin_backbone_args(base, ds, 96)
                rel = {'DLinear': 0.3, 'PatchTST': 3.0, 'iTransformer': 1.5}[base] + 1
                for name, extra in (('samelr', ''), ('gate', '--dlt_gate static --dlt_lr 0.003'),
                                    ('gate_samelr', '--dlt_gate static')):
                    g.add(f'{ds}_96_{base}_dlt_{name}', 'DLTPlug', ds, 96, seed,
                          f'--plug_base {base} {dlt} {extra} {bargs}', cost(ds, rel))
    elif a.suite == 'allnb':
        # every other variable as a neighbour (dense path); follow-up of the P2 neighbour study, where all
        # neighbours beat the CCF top-8 on Weather and PEMS08. DLT-dyn on Electricity / Traffic: GPU.
        allnb = '--dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl dense --dlt_neighbours all'
        for ds, seed in itertools.product(('PEMS08', 'PEMS04'), S):
            for v, rel in (('dltdyn', 4.0 if ds == 'PEMS08' else 12.0), ('dlt', 0.4 if ds == 'PEMS08' else 1.2)):
                g.add(f'{ds}_12_{v}_all', 'DLT', ds, 12, seed, f'--dlt_variant {v} {allnb} {DLT_TRAIN} {PEMS_PROTOCOL}',
                      cost(ds, rel))
        for ds, H, seed in itertools.product(a.datasets or ('Weather', 'ECL'), HORIZONS, S):
            for v in ('dltdyn', 'dlt'):
                if v == 'dltdyn' and ds in ('ECL', 'Traffic') and a.device == 'cpu':
                    continue                                   # ~4 h per run on one CPU thread
                g.add(f'{ds}_{H}_{v}_all', 'DLT', ds, H, seed, f'--dlt_variant {v} {allnb} {DLT_TRAIN}',
                      cost(ds, 0.6 if v == 'dlt' else 1.0))
        for seed in S:                                         # strongest backbone of the plug-in study
            bargs, _ = plugin_backbone_args('iTransformer', 'PEMS08', 12)
            g.add('PEMS08_12_iTransformer_dlt_all', 'DLTPlug', 'PEMS08', 12, seed,
                  f'--plug_base iTransformer --dlt_variant dltdyn {allnb} --dlt_lr 0.003 {bargs} {PEMS_PROTOCOL}',
                  cost('PEMS08', 6.0))
    elif a.suite == 'ci_long':                                 # the same linear model without the cross branch
        for ds, H, seed in itertools.product(a.datasets or LONG_SETS, HORIZONS, S):
            g.add(f'{ds}_{H}_ci', 'DLT', ds, H, seed, f'--dlt_variant ci {DLT_TRAIN}', cost(ds, 0.1))
    elif a.suite == 'baselines':
        for ds, H, seed in itertools.product(a.datasets or LONG_SETS, HORIZONS, S):
            for base in a.models or BASELINES:
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
