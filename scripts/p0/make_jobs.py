"""Job list of the priority-0 grid: 13 configurations x 5 settings x 3 seeds (TSLib, CPU, 1 thread per job).

    python scripts/p0/make_jobs.py <data dir> <work dir>      # writes <work dir>/jobs.txt
Each job is run from tslib/ (see run_p0.sh); it appends one JSON line to <work dir>/results_p0.jsonl,
writes its log to <work dir>/logs/ and its checkpoint to <work dir>/ckpt/. Jobs are ordered phase by phase
(ETTh1 / Weather / PEMS08 first, then Electricity), seed by seed, longest first (estimated from bench_*.jsonl).
"""
import json, sys, os
from pathlib import Path
DATA = os.path.abspath(sys.argv[1])
SP = os.path.abspath(sys.argv[2])
os.makedirs(SP, exist_ok=True)
BENCH = Path(__file__).resolve().parents[2] / 'results' / 'p0'
settings = {   # name: (data args, C, H, bench key)
    'ETTh1_96':  ('--data ETTh1 --data_path ETTh1.csv', 7, 96, 'ETTh1'),
    'Weather_96': ('--data custom --data_path weather.csv --freq t', 21, 96, 'Weather'),
    'PEMS08_24': ('--data PEMS --data_path PEMS08.npz --freq t', 170, 24, 'PEMS08_96'),
    'PEMS08_96': ('--data PEMS --data_path PEMS08.npz --freq t', 170, 96, 'PEMS08_96'),
    'ECL_96':    ('--data custom --data_path electricity.csv', 321, 96, 'ECL'),
}
configs = [(v, 'none') for v in ('ci', 'scalar', 'shift', 'tap3', 'dlt', 'full', 'scalardyn', 'dltdyn',
                                  'topk', 'mill', 'milldyn')] + [('dltdyn', 'moe'), ('dltdyn', 'static')]
impl = lambda v: 'basis' if v == 'dltdyn' else 'kernel'      # identical maths; basis is faster on CPU for dltdyn
cost = {}
for key in ('ETTh1', 'Weather', 'PEMS08_96', 'ECL'):
    for l in open(BENCH / f'bench_{key}.jsonl'):
        r = json.loads(l); cost[(key, r['variant'], r['gate'], r['impl'])] = r['epoch_s'] * 10
seeds = (2021, 2022, 2023)
jobs = []
for phase in (('ETTh1_96', 'Weather_96', 'PEMS08_24', 'PEMS08_96'), ('ECL_96',)):
    for seed in seeds:
        batch = []
        for s in phase:
            dargs, C, H, bkey = settings[s]
            for v, g in configs:
                name = f'{s}_{v}_{g}_s{seed}'
                cmd = (f'python -u run.py --task_name long_term_forecast --is_training 1 --model DLT --model_id P0_{s} '
                       f'--root_path {DATA} {dargs} --features M --seq_len 96 --label_len 48 --pred_len {H} '
                       f'--enc_in {C} --dec_in {C} --c_out {C} --dlt_variant {v} --dlt_gate {g} --dlt_impl {impl(v)} '
                       f'--dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_topk 5 --learning_rate 0.003 --batch_size 32 '
                       f'--train_epochs 10 --patience 3 --num_workers 0 --no_use_gpu --des P0 --fix_seed {seed} '
                       f'--checkpoints {SP}/ckpt --no_epoch_test --no_save_preds --result_jsonl {SP}/results_p0.jsonl '
                       f'> {SP}/logs/{name}.log 2>&1')
                batch.append((cost.get((bkey, v, g, impl(v)), cost.get((bkey, v, g, 'kernel'))), name, cmd))
        jobs += sorted(batch, reverse=True)                     # longest first inside each seed batch
open(f'{SP}/jobs.txt', 'w').write('\n'.join(j[2] for j in jobs) + '\n')
tot = sum(j[0] for j in jobs)
print(len(jobs), 'jobs; model-compute estimate', round(tot / 3600, 1), 'core-hours, ~', round(tot / 3600 / 4, 1), 'h on 4 cores')
pa = sum(j[0] for j in jobs if not j[1].startswith('ECL'))
print('phase A (ETTh1/Weather/PEMS08):', round(pa / 3600 / 4, 1), 'h')
