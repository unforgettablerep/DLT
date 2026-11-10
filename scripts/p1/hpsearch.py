"""Random hyper-parameter search for DLT-dyn on the validation split (fairness w.r.t. MillGNN's 20-trial NNI search).

    python scripts/p1/hpsearch.py make   --data <dir> --work <dir> [--datasets ETTh1 ...] [--H 96] [--trials 20] [--device cpu]
    bash scripts/run_queue.sh <work>/jobs_hp.txt
    python scripts/p1/hpsearch.py select --work <dir>          # -> <work>/tuned.json (+ printed table)
    python scripts/p1/hpsearch.py tuned  --data <dir> --work <dir> [--horizons 96 192 336 720] [--seeds ...]
    bash scripts/run_queue.sh <work>/jobs_tuned.txt
--tag <suffix> keeps a second search in the same work dir, e.g. PEMS at horizon 12 (TimeMixer protocol):
    hpsearch.py make --datasets PEMS04 PEMS08 --H 12 --tag _pems ...; select --tag _pems; tuned --horizons 12 --tag _pems

Every trial is one training run with seed 2021; the configuration with the lowest best validation loss is
selected per dataset (searched at --H and reused for all horizons when --horizons is wider: a budget choice of
the CPU run, recorded in tuned.json). The test split is never used for selection.
"""
import argparse, json, os, random, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exps

SPACE = {'learning_rate': [0.001, 0.003, 0.01], 'batch_size': [16, 32, 64], 'dlt_k': [4, 8, 16],
         'dlt_L': [24, 48, 96], 'dlt_R': [2, 4, 8], 'dlt_dyn_dim': [8, 16, 32], 'dlt_gate': ['none', 'static', 'moe']}
DEFAULT = {'learning_rate': 0.003, 'batch_size': 32, 'dlt_k': 8, 'dlt_L': 48, 'dlt_R': 4, 'dlt_dyn_dim': 16,
           'dlt_gate': 'none'}


def trials(n, seed=0):
    rng = random.Random(seed)
    out, seen = [dict(DEFAULT)], {tuple(DEFAULT.values())}          # trial 0 = the default configuration
    while len(out) < n:
        c = {k: rng.choice(v) for k, v in SPACE.items()}
        if tuple(c.values()) not in seen:
            seen.add(tuple(c.values())); out.append(c)
    return out


def cfg_args(c):
    return (f"--dlt_variant dltdyn --dlt_impl basis --learning_rate {c['learning_rate']} --batch_size {c['batch_size']} "
            f"--dlt_k {c['dlt_k']} --dlt_L {c['dlt_L']} --dlt_R {c['dlt_R']} --dlt_dyn_dim {c['dlt_dyn_dim']} "
            f"--dlt_gate {c['dlt_gate']} --train_epochs 10 --patience 3")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['make', 'select', 'tuned'])
    p.add_argument('--data'); p.add_argument('--work', required=True)
    p.add_argument('--datasets', nargs='+', default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather'])
    p.add_argument('--H', type=int, default=96)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--device', default='cpu', choices=['cpu', 'gpu'])
    p.add_argument('--horizons', type=int, nargs='+', default=list(exps.HORIZONS))
    p.add_argument('--seeds', type=int, nargs='+', default=[2021, 2022, 2023])
    p.add_argument('--tag', default='', help='suffix of the suite / file names (hp<tag>, tuned<tag>)')
    a = p.parse_args()
    HP, TU = 'hp' + a.tag, 'tuned' + a.tag
    work = os.path.abspath(a.work)
    if a.cmd == 'make':
        a.suite = HP; g = exps.Gen(a, HP)
        for ds in a.datasets:
            for t, c in enumerate(trials(a.trials)):
                proto = exps.PEMS_PROTOCOL if ds.startswith('PEMS') else ''
                g.add(f'{ds}_{a.H}_t{t:02d}', 'DLT', ds, a.H, 2021, f'{cfg_args(c)} {proto} --dlt_hp_trial {t}',
                      exps.cost(ds, c['dlt_k'] / 8 * c['dlt_L'] / 48))
        g.write()
    elif a.cmd == 'select':
        rows = [json.loads(l) for l in open(f'{work}/results_{HP}.jsonl')]
        best = {}
        name = lambda r: r['des'].split('_')[0]                   # job name = <dataset>_<H>_t<trial>
        for r in rows:
            key = name(r)
            if key not in best or r['best_vali'] < best[key]['best_vali']:
                best[key] = r
        tuned = {}
        print('| dataset | trials | best val | default val | lr | batch | k | L | R | d | gate |')
        print('|---|---|---|---|---|---|---|---|---|---|---|')
        for ds, r in sorted(best.items()):
            n = sum(1 for x in rows if name(x) == ds)
            dflt = [x for x in rows if name(x) == ds and x.get('hp_trial') == 0]
            c = {'learning_rate': r['lr'], 'batch_size': r['batch_size'], 'dlt_k': r['k'], 'dlt_L': r['L'],
                 'dlt_R': r['R'], 'dlt_dyn_dim': r.get('dyn_dim', 16), 'dlt_gate': r['gate']}
            tuned[ds] = {'config': c, 'searched_at_H': r['pred_len'], 'trials': n, 'best_vali': r['best_vali']}
            print(f"| {ds} | {n} | {r['best_vali']:.4f} | {dflt[0]['best_vali'] if dflt else float('nan'):.4f} | "
                  + ' | '.join(str(v) for v in c.values()) + ' |')
        json.dump(tuned, open(f'{work}/{TU}.json', 'w'), indent=1)
    else:
        tuned = json.load(open(f'{work}/{TU}.json'))
        a.suite = TU; g = exps.Gen(a, TU)
        for ds, t in tuned.items():
            for H, seed in itertools.product(a.horizons, a.seeds):
                proto = exps.PEMS_PROTOCOL if ds.startswith('PEMS') else ''
                g.add(f'{ds}_{H}_tuned', 'DLT', ds, H, seed, f"{cfg_args(t['config'])} {proto}", exps.cost(ds))
        g.write()


if __name__ == '__main__':
    main()
