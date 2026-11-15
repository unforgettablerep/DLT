"""Plug-in study (suite plugin_all of scripts/exps.py): every backbone alone and + DLT-dyn, side by side.

    python scripts/p1/plugin_table.py [--metric mse|mae] [--arm dlt] [--seeds 2021] [--out FILE] [--files F ...]

Reads results_plugin_all.jsonl from results/p1 and from the work directory (fixed locations; a run present in both
is taken from work/), or the files given with --files (e.g. copied from another machine). Several seeds are averaged per setting. PEMS: input 96, MSE / MAE on the normalised scale
(protocol of iTransformer and S-Mamba). --arm picks the "+ DLT" arm: dlt is the default recipe; a list written with
--dlt "<args>" has its own arm name (dlt_<args>), which the script lists when it finds one.

Sections (markdown):
  1. change of the metric by the branch, averaged over the horizons: dataset x backbone (bold: the branch helps;
     (k/4): only k horizons have both arms),
  2. every dataset and horizon: backbone alone / + DLT (change),
  3. per backbone: settings with both arms, how often the branch helps, mean and median change of MSE and MAE,
     epochs run and best epoch of both arms (an arm that stops much earlier than the backbone alone points to the
     training recipe rather than to the branch),
  4. per dataset: median change of the test metric and of the best validation loss, how often the validation loss
     improves while the test metric gets worse (a validation / test mismatch), best epoch of both arms,
  5. the validation rule: the branch is kept only where it lowers the best validation loss of the backbone alone
     (a decision made on validation data only); mean change and settings made worse by more than 2 %, with and
     without the rule. Settings whose test metric is NaN are left out of every section and counted at the top,
  6. the backbone alone against its published numbers (scripts/p1/reported.json: long-term sets and Solar, and
     PEMS with the protocol of iTransformer), and + DLT against the published backbone alone. A backbone more
     than 10 % behind its published numbers is marked: its comparison is not valid until its learning rate has
     been checked (scripts/exps.py --backbone_lr / --select_lr).
Where a learning-rate check was run (arms alone@lr<x>), the backbone alone and + DLT are taken at the learning
rate whose backbone-alone run has the lowest validation loss; the header lists these settings.
"""
import argparse, collections, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
import exps  # noqa: E402

SUITE = 'plugin_all'
NAME = {'SMamba': 'S-Mamba'}
DS_OF_FILE = {v[1]: k for k, v in exps.DATASETS.items()}
DATASETS = list(exps.PLUGIN_ALL_SETS)


def load(seeds, files=None):
    runs = {}
    for f in files or [d / f'results_{SUITE}.jsonl' for d in (paths.REPO / 'results' / 'p1', paths.WORK)]:
        f = Path(f)                                                     # later files win
        if not f.exists():
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:                                # a record cut off by an interruption
                continue
            if r.get('model') == 'DLTPlug' and (not seeds or r.get('seed') in seeds):
                runs[(r['des'], r.get('seed'))] = r
    cells = collections.defaultdict(list)                               # (dataset, H, backbone, arm) -> records
    for r in runs.values():
        ds, H, base = DS_OF_FILE[r['data_path']], r['pred_len'], r['plug_base']
        stem = f'{ds}_{H}_{base}_'
        if r['des'].startswith(stem):
            cells[(ds, H, base, r['des'][len(stem):])].append(r)
    return cells


def mean(rs, k):
    v = [r[k] for r in rs if r.get(k) is not None]
    return sum(v) / len(v) if v else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metric', default='mse', choices=['mse', 'mae'])
    p.add_argument('--arm', default='dlt')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--out')
    p.add_argument('--files', nargs='+')
    a = p.parse_args()
    cells = load(a.seeds, a.files)
    k = a.metric
    arms = sorted({key[3].split('@')[0] for key in cells} - {'alone'})
    bases = [b for b in exps.BASELINES if any(key[2] == b for key in cells)] + \
        sorted({key[2] for key in cells} - set(exps.BASELINES))
    horizons = lambda ds: exps.PEMS_HORIZONS if ds in exps.PEMS_SETS else exps.HORIZONS
    variants = collections.defaultdict(dict)                             # (ds, H, base) -> {lr suffix: alone}
    for (ds, H, base, arm), rs in cells.items():
        if arm == 'alone' or arm.startswith('alone@lr'):
            variants[(ds, H, base)][arm[len('alone'):]] = rs
    vali = lambda rs: mean(rs, 'best_vali') if mean(rs, 'best_vali') is not None else float('inf')
    pair, nan, pending, chosen, alone_sel = {}, [], [], {}, {}           # pair: (ds, H, base) -> (alone, + DLT)
    for key, v in variants.items():
        sfx = min(v, key=lambda x: vali(v[x]))                           # learning rate chosen on validation
        alone_sel[key] = v[sfx]
        if sfx:
            chosen[key] = sfx
        partner = cells.get(key + (a.arm + sfx,))
        label = f'{key[0]}-{key[1]} {NAME.get(key[2], key[2])}'
        if partner is None:
            if sfx and (key + (a.arm,)) in cells:
                pending.append(f'{label} ({sfx[1:]})')
            continue
        both = (v[sfx], partner)
        if all(m == m for x in both for m in [mean(x, k)]):                # NaN != NaN
            pair[key] = both
        else:
            nan.append(label)
    seeds = sorted({r.get('seed') for rs in cells.values() for r in rs})
    out = [f'# Plug-in study: backbone alone vs + DLT-dyn (arm `{a.arm}`)', '',
           f'Suite `{SUITE}`, seed(s) {", ".join(map(str, seeds)) or "-"}; {len(pair)} settings with both arms. '
           f'Metric: {k.upper()} (PEMS: normalised scale). Change = (+ DLT) / alone - 1; **bold**: the branch helps.']
    if nan:
        out.append(f'Left out, test {k.upper()} is NaN in at least one arm: {", ".join(sorted(nan))} '
                   f'(python scripts/gpu/diag_nonfinite.py <job> finds where).')
    if chosen:
        out.append(f'Learning rate chosen on the validation loss of the backbone alone: '
                   f'{", ".join(f"{d}-{h} {NAME.get(b, b)} {s[1:]}" for (d, h, b), s in sorted(chosen.items()))}.')
    if pending:
        out.append(f'+ DLT at the chosen learning rate still missing (left out): {", ".join(sorted(pending))} '
                   f'(python scripts/exps.py plugin_all --select_lr ...).')
    if arms and arms != [a.arm]:
        out.append(f'"+ DLT" arms found: {", ".join(arms)} (choose one with --arm).')
    fmt = lambda d: f'{100 * d:+.1f} %'
    hdr = '| dataset | ' + ' | '.join(NAME.get(b, b) for b in bases) + ' |'
    sep = '|---|' + '---:|' * len(bases)

    # 1. averaged over the horizons
    out += ['', f'## 1. Change of {k.upper()}, averaged over the horizons', '', hdr, sep]
    col = collections.defaultdict(list)
    for ds in DATASETS:
        row = []
        for b in bases:
            hs = [H for H in horizons(ds) if (ds, H, b) in pair]
            if not hs:
                row.append('')
                continue
            m0 = sum(mean(pair[(ds, H, b)][0], k) for H in hs) / len(hs)
            m1 = sum(mean(pair[(ds, H, b)][1], k) for H in hs) / len(hs)
            d = m1 / m0 - 1
            col[b].append(d)
            cell = fmt(d) + ('' if len(hs) == len(horizons(ds)) else f' ({len(hs)}/{len(horizons(ds))})')
            row.append(f'**{cell}**' if d < 0 else cell)
        if any(row):
            out.append(f'| {ds} | ' + ' | '.join(row) + ' |')
    out.append('| mean | ' + ' | '.join(fmt(sum(col[b]) / len(col[b])) if col[b] else '' for b in bases) + ' |')

    # 2. every dataset and horizon
    out += ['', f'## 2. {k.upper()} of the backbone alone / + DLT-dyn (change)', '',
            '| dataset | H | ' + ' | '.join(NAME.get(b, b) for b in bases) + ' |', '|---|---:|' + '---:|' * len(bases)]
    for ds in DATASETS:
        for H in horizons(ds):
            row = []
            for b in bases:
                if (ds, H, b) in pair:
                    m0, m1 = mean(pair[(ds, H, b)][0], k), mean(pair[(ds, H, b)][1], k)
                    cell = f'{m0:.3f} / {m1:.3f} ({fmt(m1 / m0 - 1)})'
                    row.append(f'**{cell}**' if m1 < m0 else cell)
                else:
                    alone, dlt = (ds, H, b, 'alone') in cells, (ds, H, b, a.arm) in cells
                    row.append('' if not (alone or dlt) else ('alone only' if alone else '+ DLT only'))
            if any(row):
                out.append(f'| {ds} | {H} | ' + ' | '.join(row) + ' |')

    # 3. per backbone
    out += ['', '## 3. Per backbone', '',
            '| backbone | settings | branch helps (MSE) | mean / median change MSE | mean change MAE | '
            'epochs alone / + DLT | best epoch alone / + DLT |', '|---|---:|---:|---:|---:|---:|---:|']
    for b in bases:
        ps = [v for key, v in pair.items() if key[2] == b]
        if not ps:
            continue
        dm = [mean(x[1], 'mse') / mean(x[0], 'mse') - 1 for x in ps]
        da = [mean(x[1], 'mae') / mean(x[0], 'mae') - 1 for x in ps]
        ep = [sum(mean(x[i], key) or 0 for x in ps) / len(ps) for key in ('epochs', 'best_epoch') for i in (0, 1)]
        out.append(f'| {NAME.get(b, b)} | {len(ps)} | {sum(d < 0 for d in dm)} of {len(ps)} | '
                   f'{fmt(sum(dm) / len(dm))} / {fmt(statistics.median(dm))} | {fmt(sum(da) / len(da))} | '
                   f'{ep[0]:.1f} / {ep[1]:.1f} | {ep[2]:.1f} / {ep[3]:.1f} |')
    # 4. per dataset: training behaviour
    out += ['', '## 4. Per dataset: test vs validation, early stopping', '',
            '| dataset | settings | branch helps | median change test | median change validation | '
            'validation better, test worse | best epoch alone / + DLT |', '|---|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        ps = [v for key, v in pair.items() if key[0] == ds]
        if not ps:
            continue
        dt = [mean(x[1], k) / mean(x[0], k) - 1 for x in ps]
        dv = [mean(x[1], 'best_vali') / mean(x[0], 'best_vali') - 1 for x in ps
              if mean(x[0], 'best_vali') and mean(x[1], 'best_vali')]
        mix = sum(v < 0 < t for v, t in zip(dv, dt))
        ba, bd = (sum(mean(x[i], 'best_epoch') or 0 for x in ps) / len(ps) for i in (0, 1))
        out.append(f'| {ds} | {len(ps)} | {sum(d < 0 for d in dt)} of {len(ps)} | {fmt(statistics.median(dt))} | '
                   f'{fmt(statistics.median(dv)) if dv else "–"} | {mix} of {len(dv)} | {ba:.1f} / {bd:.1f} |')
    # 5. the validation rule
    out += ['', '## 5. Validation rule: the branch is kept only where it lowers the validation loss', '',
            '| dataset | settings | branch kept | always + DLT: mean change, worse by > 2 % | with the rule: mean change, '
            'worse by > 2 % |', '|---|---:|---:|---:|---:|']
    for ds in DATASETS + ['all']:
        ps = [v for key, v in pair.items() if key[0] == ds or ds == 'all']
        ps = [x for x in ps if mean(x[0], 'best_vali') and mean(x[1], 'best_vali')]
        if not ps:
            continue
        d = [mean(x[1], k) / mean(x[0], k) - 1 for x in ps]
        keep = [mean(x[1], 'best_vali') < mean(x[0], 'best_vali') for x in ps]
        r = [dd if kk else 0.0 for dd, kk in zip(d, keep)]
        out.append(f'| {ds} | {len(ps)} | {sum(keep)} | {fmt(sum(d) / len(d))}, {sum(x > 0.02 for x in d)} | '
                   f'{fmt(sum(r) / len(r))}, {sum(x > 0.02 for x in r)} |')
    # 6. against the published numbers
    R = json.load(open(Path(__file__).with_name('reported.json')))
    col = 0 if k == 'mse' else 1

    def published(ds, H, base):
        v = (R['pems_96'] if ds in exps.PEMS_SETS else R['long_term']).get(base, {}).get(ds, {}).get(str(H))
        return v[col] if v else None
    rows_a, rows_d, flagged = [], [], []
    for ds in DATASETS:
        ca, cd = [], []
        for b in bases:
            ga = [mean(alone_sel[(ds, H, b)], k) / published(ds, H, b) - 1 for H in horizons(ds)
                  if (ds, H, b) in alone_sel and published(ds, H, b)
                  and mean(alone_sel[(ds, H, b)], k) == mean(alone_sel[(ds, H, b)], k)]
            gd = [mean(pair[(ds, H, b)][1], k) / published(ds, H, b) - 1 for H in horizons(ds)
                  if (ds, H, b) in pair and published(ds, H, b)]
            bad = bool(ga) and max(ga) > 0.10
            if bad:
                flagged.append(f'{ds} {NAME.get(b, b)}')
            ca.append('' if not ga else (f'**{fmt(sum(ga) / len(ga))} (worst {fmt(max(ga))})**' if bad else fmt(sum(ga) / len(ga))))
            cd.append('' if not gd else fmt(sum(gd) / len(gd)) + (' (not valid)' if bad else ''))
        if any(ca):
            rows_a.append(f'| {ds} | ' + ' | '.join(ca) + ' |')
        if any(cd):
            rows_d.append(f'| {ds} | ' + ' | '.join(cd) + ' |')
    if rows_a:
        out += ['', f'## 6. Against the published numbers ({k.upper()}, mean over the horizons)', '',
                'Backbone alone against its published result (**bold**: more than 10 % behind at some horizon):', '',
                hdr, sep] + rows_a
        out += ['', '+ DLT against the published backbone alone:', '', hdr, sep] + rows_d
        if flagged:
            out += ['', f'Learning-rate check needed (more than 10 % behind the published result): {", ".join(flagged)}.']
    text = '\n'.join(out) + '\n'
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == '__main__':
    main()
