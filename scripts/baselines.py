"""The six baselines of the study and their official settings.

    python scripts/baselines.py [model ...]        prints the settings of every dataset and horizon, with their source

BASELINES are the models of the S-Mamba paper (Wang et al., Neurocomputing 2025, Tables 2-4) that the study keeps:
S-Mamba, iTransformer, PatchTST, RLinear, DLinear and TiDE. Their published numbers (scripts/p1/reported.json,
block smamba_paper) are the reference of the baseline check (scripts/p1/baseline_check.py).

args(model, ds, H) -> (run.py arguments, source): the hyper-parameters of the official script line for this dataset
and horizon, unchanged. The scripts are copied in scripts/official/ (see its README for the commits):
  SMamba        scripts/official/SMamba/          github.com/wzhwzhwzh0921/S-D-Mamba  scripts/multivariate_forecasting
  iTransformer  scripts/official/iTransformer/    github.com/thuml/iTransformer       scripts/multivariate_forecasting
  PatchTST      scripts/official/TSLib/           github.com/thuml/Time-Series-Library scripts/long_term_forecast
  DLinear, TiDE scripts/official/TSLib/           (the iTransformer paper reproduced its baselines in this library)
  RLinear       RLINEAR below, from scripts/official/RLinear/script.md (github.com/plumprc/RTSF): learning rate,
                batch size and dropout per dataset, 15 epochs and patience 10 (the defaults of that code)
Every argument of the official line is taken except those that name the data or the run (paths, ids, sizes, the
number of variables). c_out is kept for TiDE only, where it is the decoder output dimension (8), not the number of
variables. Unset arguments have the same defaults in run.py as in those repositories.
Datasets without an official line take a documented fallback (FALLBACK below; the source says which).

protocol(ds): the evaluation protocol of the official loaders, the same for every baseline:
  --freq h             no official script sets --freq: four time features (timeF) on every dataset
  --drop_last          the training and validation loaders drop their last incomplete batch
  --test_batch_size 1  the test runs with batch 1 (no window is dropped)
  PEMS and Solar: --no_time_marks (x_mark = None, as the iTransformer / S-Mamba code); PEMS: --pems_split
  itransformer (the split of their PEMS loader).
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / 'scripts' / 'official'
BASELINES = ('SMamba', 'iTransformer', 'PatchTST', 'RLinear', 'DLinear', 'TiDE')
NAME = {'SMamba': 'S-Mamba'}
PEMS = ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')
NO_MARKS = ('Solar',) + PEMS
FILES = {'ETTh1': 'ETTh1.csv', 'ETTh2': 'ETTh2.csv', 'ETTm1': 'ETTm1.csv', 'ETTm2': 'ETTm2.csv',
         'Exchange': 'exchange_rate.csv', 'Weather': 'weather.csv', 'ECL': 'electricity.csv', 'Traffic': 'traffic.csv',
         'Solar': 'solar_AL.txt', 'PEMS03': 'PEMS03.npz', 'PEMS04': 'PEMS04.npz', 'PEMS07': 'PEMS07.npz',
         'PEMS08': 'PEMS08.npz'}
SCRIPTS = {'SMamba': ('SMamba', 'S_Mamba'), 'iTransformer': ('iTransformer', 'iTransformer'),
           'PatchTST': ('TSLib', 'PatchTST'), 'DLinear': ('TSLib', 'DLinear'), 'TiDE': ('TSLib', 'TiDE')}
# arguments of an official line that name the data or the run, not the model or its training
NOT_SETTINGS = {'is_training', 'root_path', 'data_path', 'model_id', 'model', 'data', 'features', 'seq_len',
                'label_len', 'pred_len', 'enc_in', 'dec_in', 'c_out', 'des', 'itr', 'task_name', 'gpu', 'use_gpu',
                'use_multi_gpu', 'devices', 'checkpoints', 'num_workers', 'freq', 'target'}

# RLinear: scripts/official/RLinear/script.md (input 336 there; the iTransformer paper ran it with input 96)
RLINEAR_TRAIN = '--train_epochs 15 --patience 10'
RLINEAR = {'ETTh1': '--learning_rate 0.005 --batch_size 128', 'ETTm1': '--learning_rate 0.005 --batch_size 128',
           'ETTh2': '--learning_rate 0.005 --batch_size 256', 'ETTm2': '--learning_rate 0.005 --batch_size 128',
           'Weather': '--learning_rate 0.005 --batch_size 128',
           'ECL': '--learning_rate 0.001 --batch_size 16 --dropout 0'}
# datasets without an official line: the configuration of this dataset is used (same horizon; PEMS: horizon 96)
FALLBACK = {
    'RLinear': {'Exchange': 'ETTh1', **{d: 'ECL' for d in ('Traffic', 'Solar') + PEMS}},  # few / many variables
    'PatchTST': {d: 'ECL' for d in NO_MARKS},                    # many variables, as Electricity
    'TiDE': {d: 'ETTh1' for d in FILES if d != 'ETTh1'},         # the only TiDE script of the library
    'DLinear': {},                                               # no line: the library defaults (DLinear has no
}                                                                # settings of its own besides training)


def _lines(folder):
    """{(data file, pred_len, model): ({argument: value}, script)} of the run.py lines of the official scripts"""
    out = {}
    for path in sorted((OFFICIAL / folder).rglob('*.sh')):
        txt = re.sub(r'\\[ \t]*\r?\n', ' ', path.read_text(errors='replace'))
        var = dict(re.findall(r'^\s*(\w+)=(\S+)', txt, re.M))
        for line in txt.splitlines():
            if 'run.py' not in line or line.lstrip().startswith('#'):
                continue
            toks, d, i = line.split(), {}, 0
            while i < len(toks):
                if toks[i].startswith('--'):
                    val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith('--') else ''
                    if val.startswith('$'):
                        val = var.get(val[1:].strip('{}'), val)
                    d[toks[i][2:]] = val.strip('\'"')
                    i += 2 if val else 1
                else:
                    i += 1
            key = (Path(d.get('data_path', '')).name, str(d.get('pred_len')), d.get('model', '').strip('$'))
            out.setdefault(key, (d, str(path.relative_to(REPO))))
    return out


_CACHE = {}


def official_line(model, ds, H):
    """(settings dict, script) of the official line, or (None, None)"""
    folder, name = SCRIPTS[model]
    if folder not in _CACHE:
        _CACHE[folder] = _lines(folder)
    return _CACHE[folder].get((FILES[ds], str(H), name), (None, None))


def _settings(model, d):
    keep = {k: v for k, v in d.items() if k not in NOT_SETTINGS}
    if model == 'TiDE' and 'c_out' in d:
        keep['c_out'] = d['c_out']                               # TiDE: decoder output dimension
    return ' '.join(f'--{k} {v}' if v != '' else f'--{k}' for k, v in keep.items())


def args(model, ds, H):
    """(run.py arguments, source) of the backbone's settings for this dataset and horizon (without the protocol)"""
    if model not in BASELINES:
        raise ValueError(f'{model} is not one of the baselines {BASELINES}')
    if model == 'RLinear':
        src_ds = ds if ds in RLINEAR else FALLBACK['RLinear'][ds]
        src = 'scripts/official/RLinear/script.md' + ('' if src_ds == ds else f' ({src_ds} configuration)')
        return f'{RLINEAR[src_ds]} {RLINEAR_TRAIN}', src
    d, path = official_line(model, ds, H)
    if d is not None:
        return _settings(model, d), path
    other = FALLBACK.get(model, {}).get(ds)
    if other is not None:
        h = H if official_line(model, other, H)[0] is not None else 96
        d, path = official_line(model, other, h)
        if d is not None:
            return _settings(model, d), f'{path} ({other} configuration' + (f', horizon {h})' if h != H else ')')
    if model == 'DLinear':
        return '', 'Time-Series-Library defaults (no script for this dataset)'
    raise KeyError(f'no official or fallback setting for {model} on {ds}-{H}')


def protocol(ds):
    p = '--freq h --drop_last --test_batch_size 1'
    if ds in NO_MARKS:
        p += ' --no_time_marks'
    if ds in PEMS:
        p += ' --pems_split itransformer'
    return p


def horizons(ds):
    return (12, 24, 48, 96) if ds in PEMS else (96, 192, 336, 720)


def main():
    for model in sys.argv[1:] or BASELINES:
        print(f'## {NAME.get(model, model)}')
        for ds in FILES:
            for H in horizons(ds):
                a, src = args(model, ds, H)
                print(f'{ds:8s} {H:4d}  {a or "(defaults)":70s}  {src}')


if __name__ == '__main__':
    main()
