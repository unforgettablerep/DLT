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
  long_extra   P1  DLT-dyn on the other five datasets of the S-Mamba benchmark: Solar (horizons 96/192/336/720)
                   and PEMS03/04/07/08 (12/24/48/96), with the protocol of the baseline check (normalised MSE / MAE,
                   no time marks, the PEMS split of the iTransformer loader), so that DLT-dyn has a full row in the
                   S-Mamba-style results table (--datasets restricts the sets)
  eff_tune     P2  DLT-dyn in the efficiency setting of the S-Mamba paper (Fig. 5): PEMS07 (horizon 12), Electricity,
                   Traffic and ETTm1 (horizon 96); batch size 16, the epochs of the official S-Mamba scripts (Electricity
                   5, otherwise 10), patience 3, no AMP, the protocol of the baseline check. The recipe is --dlt "<args>"
                   (appended last, so it can override --learning_rate or the DLT defaults) and named by --tag; the
                   recipes are compared on the validation loss: python scripts/p2/eff_tune.py
  baselines    P1  the baseline check: the six baselines (scripts/baselines.py: S-Mamba, iTransformer, PatchTST,
                   RLinear, DLinear, TiDE) alone, with their official settings and the protocol of the official
                   loaders, on all 13 datasets (horizons 96/192/336/720; PEMS 12/24/48/96), to be compared with the
                   S-Mamba paper: python scripts/p1/baseline_check.py. With --backbone_lr x y ... the same settings
                   at these learning rates (learning-rate check; the check takes the one with the lowest validation
                   loss, and says so)
  sensitivity  P2  DLT-dyn, one factor at a time: L in {12,24,96}, R in {2,8}, k in {4,16} (default 48/4/8)
                   on ETTh1-96, Weather-96, PEMS08-96
  neighbours   P2  DLT-dyn / DLT with random or all neighbours instead of the CCF top-k (Weather-96, PEMS08-96)
  ci_long      P1  the channel-independent linear model (DLT without the cross branch) on the long-term benchmark
  allnb        P1  DLT-dyn / DLT with all other variables as neighbours (dense path): PEMS04/08-12, Weather and
                   Electricity (DLT-dyn there only with --device gpu), iTransformer + DLT-dyn on PEMS08-12
  plugin_fusion P1 how the branch enters the backbone (models/DLTPlug.py): each backbone alone and with the arms
                   of FUSION_ARMS (output = Eq. 2, refine = distributed-lag correction over past + forecast, input =
                   neighbour signal into the backbone input; joint or decoupled training; per-variable selection on
                   validation data), same settings and protocol as plugin_all. Default: RLinear, DLinear and
                   iTransformer on ETTh2, Exchange, Weather (horizon 96) and PEMS08 (horizon 12); --horizons all for
                   every horizon. The table: python scripts/p1/fusion_table.py
  plugin_all   P1  every baseline alone and + DLT-dyn (DLTPlug, branch lr 3e-3) on the 13 datasets, the backbone
                   with the settings and the protocol of the baseline check (scripts/baselines.py).
                   GPU. The table (section 6 checks the published numbers): python scripts/p1/plugin_table.py
--models / --datasets restrict the baselines, ci_long and plugin_all suites, e.g. the CPU-feasible subset
    python scripts/exps.py baselines --models RLinear DLinear --datasets ETTh1 ETTh2 ETTm1 ETTm2 Exchange Weather
Learning-rate check (plugin_all), for backbones that run far behind their published numbers (plugin_table.py,
section 6): --backbone_lr 0.0005 0.001 runs the backbone alone at these learning rates; --select_lr then runs both
arms at the learning rate with the lowest validation loss of the backbone alone, where it is not the setting's own.
--dlt "<args>" adds to or overrides the DLT-dyn arguments, e.g. --dlt "--dlt_gate static" or --dlt "--dlt_lr 0"
(plugin_all: the branch then trains with the backbone's learning rate). --list NAME writes the jobs to
work/jobs_NAME.txt instead of work/jobs_<suite>.txt, so that a second list can run next to a running queue.
"""
import argparse, os, re, json, itertools, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import baselines  # noqa: E402

REPO = paths.REPO
# the six baselines of every comparison (scripts/baselines.py: those of the S-Mamba paper)
BASELINES = baselines.BASELINES
HORIZONS = (96, 192, 336, 720)
LONG_SETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'ECL', 'Traffic')
PEMS_SETS = ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')
PEMS_HORIZONS = (12, 24, 48, 96)                               # iTransformer / S-Mamba protocol, input 96
PLUGIN_ALL_SETS = LONG_SETS + ('Solar',) + PEMS_SETS           # the datasets of plugin_all
# The iTransformer and S-Mamba code gives no time marks to the model on PEMS and Solar (x_mark = None); their
# settings were tuned that way, and every baseline run follows it (scripts/baselines.py, protocol)
NO_MARKS_SETS = baselines.NO_MARKS

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


# plugin_fusion: arm name -> DLTPlug arguments after the default recipe (--dlt_lr 0.003 and DLT_DEFAULT)
EFF_SETS = (('ETTm1', 96), ('ECL', 96), ('Traffic', 96), ('PEMS07', 12))   # S-Mamba, Fig. 5
EFF_EPOCHS = {'ETTm1': 10, 'ECL': 5, 'Traffic': 10, 'PEMS07': 10}           # official S-Mamba scripts


FUSION_ARMS = {
    'dlt': '',                                                               # output, joint: plugin_all's arm (Eq. 2)
    'dlt_out-dec': '--dlt_train decoupled --dlt_select variable',
    'dlt_refine': '--dlt_fusion refine',
    'dlt_refine-dec': '--dlt_fusion refine --dlt_train decoupled --dlt_select variable',
    'dlt_input': '--dlt_fusion input',
}
FUSION_SETS = (('ETTh2', 96), ('Exchange', 96), ('Weather', 96), ('PEMS08', 12))
FUSION_MODELS = ('RLinear', 'DLinear', 'iTransformer')
def runtime_args(model, ds):
    """options that change memory or speed but not the results: TiDE on data with many variables recomputes each
    variable's forward pass in the backward pass (--tide_checkpoint; same outputs and gradients, unit-tested).
    Without it TiDE needs 25-48 GB for its activations on PEMS04 at batch 512 (horizons 24-96)."""
    return '--tide_checkpoint' if model == 'TiDE' and DATASETS[ds][2] > 100 else ''


# relative cost of a backbone (only orders the job lists, longest first)
REL_COST = {'SMamba': 1.5, 'iTransformer': 1.5, 'PatchTST': 3.0, 'RLinear': 0.2, 'DLinear': 0.3, 'TiDE': 2.0}


# Learning-rate check of a backbone alone: where a backbone runs far behind its published numbers, its learning rate
# is chosen on the validation loss of the backbone alone from its own setting and these values; both arms then use it.
LR_GRID = (1e-4, 5e-4, 1e-3)


def lr_suffix(lr):
    return f'@lr{lr:g}'


def setting_lr(bargs):
    m = re.search(r'--learning_rate (\S+)', bargs)
    return float(m.group(1)) if m else 1e-4                    # run.py default


def plugin_records():
    """records of plugin_all from results/p1 and the work directory (a run in both: the work directory's)"""
    runs = {}
    for f in (REPO / 'results' / 'p1' / 'results_plugin_all.jsonl', paths.WORK / 'results_plugin_all.jsonl'):
        if f.exists():
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                runs[(r['des'], r.get('seed'))] = r
    return list(runs.values())


def selected_lrs(records):
    """(dataset, H, backbone) -> learning-rate suffix ('' = the setting's own) whose backbone-alone run has the lowest
    best validation loss (mean over seeds), for the settings with learning-rate check runs (arm alone@lr<x>)"""
    ds_of = {v[1]: k for k, v in DATASETS.items()}
    vals = {}
    for r in records:
        if r.get('model') != 'DLTPlug' or r.get('best_vali') is None or r['data_path'] not in ds_of:
            continue
        ds, H, base = ds_of[r['data_path']], r['pred_len'], r['plug_base']
        stem = f'{ds}_{H}_{base}_alone'
        if r['des'] == stem or r['des'].startswith(stem + '@lr'):
            vals.setdefault((ds, H, base), {}).setdefault(r['des'][len(stem):], []).append(r['best_vali'])
    return {key: min(v, key=lambda sfx: sum(v[sfx]) / len(v[sfx])) for key, v in vals.items() if len(v) > 1}


def recipe_tag(extra):
    """short name of extra --dlt arguments: a changed recipe gets its own job names (dlt_<tag>), so that runs of the
    default recipe are neither skipped for nor mixed with it"""
    return re.sub(r'[^A-Za-z0-9.]+', '-', re.sub(r'--(dlt_)?', '', extra)).strip('-')


def plugin_all_args(model, ds, H):
    """(settings, source) of a baseline for this dataset and horizon: scripts/baselines.py (without the protocol)"""
    return baselines.args(model, ds, H)


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
    p.add_argument('--backbone_lr', type=float, nargs='+',
                   help='plugin_all / baselines: learning-rate check, the backbone alone only, at these learning rates (<job>@lr<x>)')
    p.add_argument('--select_lr', action='store_true',
                   help='plugin_all: both arms at the learning rate whose backbone-alone run has the lowest validation '
                        'loss, for the settings with a learning-rate check (arms alone@lr<x>, dlt@lr<x>)')
    p.add_argument('--tag', help='eff_tune: name of the recipe given by --dlt (job names <ds>_<H>_dltdyn_<tag>)')
    p.add_argument('--list', help='name of the job list (work/jobs_<list>.txt; default: the suite), e.g. to run a '
                                  'second list of the same suite next to a running queue; results still go to '
                                  'work/results_<suite>.jsonl')
    p.add_argument('--horizons', default='first', choices=['first', 'all'], help='plugin_fusion: the first or all horizons')
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
    elif a.suite == 'eff_tune':
        tag = a.tag or 'default'
        for (ds, H), seed in itertools.product(EFF_SETS, S):
            if a.datasets and ds not in a.datasets:
                continue
            train = f'--batch_size 16 --train_epochs {EFF_EPOCHS[ds]} --patience 3 --learning_rate 0.003'
            g.add(f'{ds}_{H}_dltdyn_{tag}', 'DLT', ds, H, seed,
                  f'{DLT_DEFAULT} {train} {baselines.protocol(ds)} {a.dlt or ""}'.strip(), cost(ds, 2.0))
    elif a.suite == 'long_extra':
        for ds in a.datasets or ('PEMS08', 'PEMS04', 'Solar', 'PEMS03', 'PEMS07'):
            for H, seed in itertools.product(PEMS_HORIZONS if ds in PEMS_SETS else HORIZONS, S):
                g.add(f'{ds}_{H}_dltdyn', 'DLT', ds, H, seed, f'{dlt} {DLT_TRAIN} {baselines.protocol(ds)}', cost(ds))
    elif a.suite in ('long_small', 'long_large'):
        sets = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather') if a.suite == 'long_small' else ('ECL', 'Traffic')
        for ds, H, seed in itertools.product(sets, HORIZONS, S):
            g.add(f'{ds}_{H}_dltdyn', 'DLT', ds, H, seed, f'{dlt} {DLT_TRAIN}', cost(ds))
    elif a.suite == 'plugin_all':
        # each backbone alone and + DLT-dyn on every dataset and horizon. Both arms go through DLTPlug ("alone":
        # --dlt_variant ci, which builds the backbone only and is identical to running it directly). The backbone
        # has the settings and the protocol of the baseline check (scripts/baselines.py). --dlt_lr comes first so
        # that --dlt can override it.
        arm = 'dlt' + (f'_{recipe_tag(a.dlt)}' if a.dlt else '')
        chosen = selected_lrs(plugin_records()) if a.select_lr else {}
        for ds, base, seed in itertools.product(a.datasets or PLUGIN_ALL_SETS, a.models or BASELINES, S):
            for H in PEMS_HORIZONS if ds in PEMS_SETS else HORIZONS:
                bargs = f'{baselines.protocol(ds)} {plugin_all_args(base, ds, H)[0]} {runtime_args(base, ds)}'.strip()
                rel = REL_COST.get(base, 2.0)
                if a.backbone_lr:                             # learning-rate check: the backbone alone only
                    for lr in a.backbone_lr:
                        if abs(lr - setting_lr(bargs)) > 1e-12:
                            g.add(f'{ds}_{H}_{base}_alone{lr_suffix(lr)}', 'DLTPlug', ds, H, seed,
                                  f'--plug_base {base} --dlt_variant ci {bargs} --learning_rate {lr:g}', cost(ds, rel), label=48)
                    continue
                sfx = ''
                if a.select_lr:                               # both arms at the learning rate chosen on validation
                    sfx = chosen.get((ds, H, base), '')
                    if not sfx:
                        continue                              # the setting's own learning rate: nothing to add
                    bargs = f'{bargs} --learning_rate {sfx[3:]}'
                g.add(f'{ds}_{H}_{base}_alone{sfx}', 'DLTPlug', ds, H, seed,
                      f'--plug_base {base} --dlt_variant ci {bargs}', cost(ds, rel), label=48)
                g.add(f'{ds}_{H}_{base}_{arm}{sfx}', 'DLTPlug', ds, H, seed,
                      f'--plug_base {base} --dlt_lr 0.003 {dlt} {bargs}', cost(ds, rel + 1), label=48)
    elif a.suite == 'plugin_fusion':
        sets = [(ds, H) for ds, H0 in FUSION_SETS if not a.datasets or ds in a.datasets
                for H in ((PEMS_HORIZONS if ds in PEMS_SETS else HORIZONS) if a.horizons == 'all' else (H0,))]
        if a.datasets:
            sets += [(ds, H) for ds in a.datasets if ds not in dict(FUSION_SETS)
                     for H in ((PEMS_HORIZONS if ds in PEMS_SETS else HORIZONS) if a.horizons == 'all'
                               else ((12,) if ds in PEMS_SETS else (96,)))]
        for (ds, H), base, seed in itertools.product(sets, a.models or FUSION_MODELS, S):
            bargs = f'{baselines.protocol(ds)} {plugin_all_args(base, ds, H)[0]} {runtime_args(base, ds)}'.strip()
            rel = REL_COST.get(base, 2.0)
            g.add(f'{ds}_{H}_{base}_alone', 'DLTPlug', ds, H, seed,
                  f'--plug_base {base} --dlt_variant ci {bargs}', cost(ds, rel), label=48)
            for arm, extra in FUSION_ARMS.items():
                g.add(f'{ds}_{H}_{base}_{arm}', 'DLTPlug', ds, H, seed,
                      f'--plug_base {base} --dlt_lr 0.003 {dlt} {extra} {bargs}', cost(ds, rel + 1), label=48)
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
            bargs, _ = baselines.args('iTransformer', 'PEMS08', 12)
            g.add('PEMS08_12_iTransformer_dlt_all', 'DLTPlug', 'PEMS08', 12, seed,
                  f'--plug_base iTransformer --dlt_variant dltdyn {allnb} --dlt_lr 0.003 {bargs} {PEMS_PROTOCOL}',
                  cost('PEMS08', 6.0))
    elif a.suite == 'ci_long':                                 # the same linear model without the cross branch
        for ds, H, seed in itertools.product(a.datasets or LONG_SETS, HORIZONS, S):
            g.add(f'{ds}_{H}_ci', 'DLT', ds, H, seed, f'--dlt_variant ci {DLT_TRAIN}', cost(ds, 0.1))
    elif a.suite == 'baselines':
        # the baseline check: each baseline alone, official settings and protocol (scripts/baselines.py).
        # --backbone_lr: learning-rate check, the same settings at these learning rates (jobs <name>@lr<x>);
        # baseline_check.py then takes, per setting, the learning rate with the lowest validation loss
        for ds, base, seed in itertools.product(a.datasets or PLUGIN_ALL_SETS, a.models or BASELINES, S):
            for H in PEMS_HORIZONS if ds in PEMS_SETS else HORIZONS:
                bargs, _ = baselines.args(base, ds, H)
                full = f'{baselines.protocol(ds)} {bargs} {runtime_args(base, ds)}'.strip()
                if a.backbone_lr:
                    for lr in a.backbone_lr:
                        if abs(lr - setting_lr(full)) > 1e-12:
                            g.add(f'{ds}_{H}_{base}{lr_suffix(lr)}', base, ds, H, seed, f'{full} --learning_rate {lr:g}',
                                  cost(ds, REL_COST[base]), label=48)
                    continue
                g.add(f'{ds}_{H}_{base}', base, ds, H, seed, full, cost(ds, REL_COST[base]), label=48)
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
