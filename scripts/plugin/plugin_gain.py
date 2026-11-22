"""Plug-in study: which backbones and which datasets gain from the DLT branch, and what the gain goes with.

    python scripts/plugin/plugin_gain.py [--arm dlt] [--seeds 2021] [--files F ...] [--diag FILE] [--out FILE]

The arms are paired as in plugin_table.py (same records, learning rate chosen on validation, settings whose MSE is
NaN left out). Every dataset x backbone comparison is classified by its backbone alone against the published numbers
(assets/reported.json; as section 6 of plugin_table.py):
  behind     more than 10 % behind the published result at some horizon (or behind the published average where
             only that is known): left out, the comparison is not valid;
  unchecked  no published result for this backbone and dataset: kept;
  valid      within 10 %: kept.
A kept comparison is marked "old" when one of its runs used settings other than those plugin_all generates now
(src/configs/baseline_profiles.py: e_layers, d_model, d_ff, n_heads, learning rate, batch size, epochs, patience;
on PEMS and Solar the protocol without time marks): it is to be run again.
Change = (+ DLT) / alone - 1 of the test MSE. Sections (markdown):
  1. dataset x backbone: change of the MSE averaged over the horizons (as plugin_table.py section 1), with the marks;
  2. per backbone and 3. per dataset, kept comparisons only: settings, median change, settings where the branch
     helps (with a two-sided sign test; the settings share data, so the p-value is optimistic), the range of the
     cells of section 1, the mean change when the branch is kept only where it lowers the validation loss;
  4. what the size of the gain goes with:
     a. headroom: Spearman correlation between the change and how far the backbone alone is behind the best
        backbone alone of the same dataset and horizon (kept comparisons),
     b. the best + DLT result against the best backbone alone, per dataset and horizon,
     c. the horizon,
     d. the data (with --diag, output/plugin/leadlag_diag.json of scripts/plugin/leadlag_diag.py): the linear diagnostic next to
        the change for DLinear and the median change over the backbones.
"""
import argparse, collections, json, math, re, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plugin_table as pt  # noqa: E402
from run import parse_args  # noqa: E402
from src.configs import baseline_profiles  # noqa: E402
from src.configs.dlt_profiles import NO_MARKS, PEMS  # noqa: E402

NAME = pt.NAME
FIELDS = {'e_layers': 'e_layers', 'd_model': 'd_model', 'd_ff': 'd_ff', 'n_heads': 'n_heads',
          'learning_rate': 'learning_rate', 'batch_size': 'batch_size', 'train_epochs': 'train_epochs',
          'patience': 'patience'}                                       # option -> result column


def run_defaults():
    """defaults of scripts/run.py for the options of FIELDS"""
    args = parse_args(['--model', 'DLTPlug'])
    return {opt: float(getattr(args, opt)) for opt in FIELDS}


DEFAULTS = run_defaults()


def expected(base, ds, H):
    """the settings plugin_all generates now for the backbone (run.py defaults where the arguments set none)"""
    toks = baseline_profiles.args(base, ds, H)[0].split()
    d = dict(DEFAULTS)
    for i, t in enumerate(toks[:-1]):
        if t.startswith('--') and t[2:] in FIELDS:
            d[t[2:]] = float(toks[i + 1])
    return d


def is_old(r, base, ds, H, lr_checked):
    want = expected(base, ds, H)
    for opt, field in FIELDS.items():
        if opt == 'learning_rate' and lr_checked:                      # an arm of the learning-rate check
            continue
        if r.get(field) is not None and abs(float(r[field]) - want[opt]) > 1e-12:
            return True
    return ds in NO_MARKS and r.get('time_marks') is not False


def spearman(x, y):
    if len(x) < 2:
        return float('nan')
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def sign_p(k, n):
    """two-sided binomial sign test, p = 1/2"""
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', default='dlt')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--files', nargs='+')
    p.add_argument('--diag', help='JSON written by scripts/plugin/leadlag_diag.py --out (e.g. output/plugin/leadlag_diag.json)')
    p.add_argument('--out')
    a = p.parse_args()
    cells = pt.load(a.seeds, a.files)
    pair, nan, pending, chosen, alone_sel = pt.pair_arms(cells, a.arm, 'mse')
    m = pt.mean
    bases = [b for b in baseline_profiles.BASELINES if any(k[2] == b for k in pair)]
    dsets = [d for d in pt.DATASETS if any(k[0] == d for k in pair)]
    cls, old = {}, {}
    for ds in dsets:
        for b in bases:
            hs = [H for H in pt.horizons(ds) if (ds, H, b) in pair]
            if not hs:
                continue
            ga, _ = pt.baseline_gaps(ds, b, alone_sel, pair, 'mse')
            cls[(ds, b)] = 'unchecked' if not ga else ('behind' if max(ga) > 0.10 else 'valid')
            old[(ds, b)] = any(is_old(r, b, ds, H, bool(chosen.get((ds, H, b))))
                               for H in hs for arm in (0, 1) for r in pair[(ds, H, b)][arm])
    S = []                                                               # one entry per setting with both arms
    for (ds, H, b), (ra, rd) in pair.items():
        if (ds, b) not in cls:
            continue
        S.append(dict(ds=ds, H=H, b=b, cls=cls[(ds, b)], old=old[(ds, b)], alone=m(ra, 'mse'), dlt=m(rd, 'mse'),
                      d=m(rd, 'mse') / m(ra, 'mse') - 1, va=m(ra, 'best_vali'), vd=m(rd, 'best_vali')))
    kept = [s for s in S if s['cls'] != 'behind']
    fmt = lambda d: f'{100 * d:+.1f} %'
    seeds = sorted({r.get('seed') for rs in cells.values() for r in rs})
    n_old = sum(1 for s in kept if s['old'])
    out = [f'# Plug-in study: which backbones and datasets gain from the DLT branch (arm `{a.arm}`)', '',
           f'Seed(s) {", ".join(map(str, seeds))}; {len(S)} settings with both arms; kept (backbone alone within 10 % of '
           f'its published result, or no published result): {len(kept)}; left out (more than 10 % behind): '
           f'{len(S) - len(kept)}. Of the kept settings, {n_old} come from runs with settings or a protocol that '
           f'plugin_all no longer generates (marked old; to be run again).']
    if nan:
        out.append(f'Not paired, test MSE is NaN: {", ".join(sorted(nan))}.')
    if pending:
        out.append(f'Not paired, + DLT at the chosen learning rate missing: {", ".join(sorted(pending))}.')

    # 1. matrix
    mark = lambda ds, b: {'behind': ' (behind)', 'unchecked': ' (?)', 'valid': ''}[cls[(ds, b)]] + (' (old)' if old[(ds, b)] else '')
    out += ['', '## 1. Change of the MSE averaged over the horizons', '',
            '(behind): the backbone alone is more than 10 % behind its published result, not valid; (?): no published '
            'result; (old): run with settings or a protocol that plugin_all no longer generates. **Bold**: kept and the '
            'branch helps.', '',
            '| dataset | ' + ' | '.join(NAME.get(b, b) for b in bases) + ' |', '|---|' + '---:|' * len(bases)]
    for ds in dsets:
        row = []
        for b in bases:
            v = [s for s in S if s['ds'] == ds and s['b'] == b]
            if not v:
                row.append('')
                continue
            d = statistics.mean(s['dlt'] for s in v) / statistics.mean(s['alone'] for s in v) - 1
            c = fmt(d) + mark(ds, b)
            row.append(f'**{c}**' if d < 0 and cls[(ds, b)] != 'behind' else c)
        out.append(f'| {ds} | ' + ' | '.join(row) + ' |')

    def block(title, key, groups):
        rows = ['', title, '',
                '| | settings | median change | branch helps | sign test p | cells (section 1) | with the validation rule |',
                '|---|---:|---:|---:|---:|---|---:|']
        for g in groups:
            v = [s for s in kept if s[key] == g]
            if not v:
                continue
            d = [s['d'] for s in v]
            k = sum(x < 0 for x in d)
            other = 'b' if key == 'ds' else 'ds'
            cell = {}
            for s in v:
                cell.setdefault(s[other], []).append(s)
            cv = sorted((statistics.mean(s['dlt'] for s in c) / statistics.mean(s['alone'] for s in c) - 1, o)
                        for o, c in cell.items())
            rng = f'{fmt(cv[0][0])} ({NAME.get(cv[0][1], cv[0][1])}) to {fmt(cv[-1][0])} ({NAME.get(cv[-1][1], cv[-1][1])})'
            rule = [s['d'] if s['vd'] is not None and s['va'] is not None and s['vd'] < s['va'] else 0.0 for s in v]
            n_old_g = sum(s['old'] for s in v)
            rows.append(f'| {NAME.get(g, g)} | {len(v)}{f" ({n_old_g} old)" if n_old_g else ""} | '
                        f'{fmt(statistics.median(d))} | {k} of {len(v)} | {sign_p(k, len(v)):.2g} | {rng} | '
                        f'{fmt(statistics.mean(rule))} |')
        return rows
    out += block('## 2. Per backbone (kept comparisons)', 'b', bases)
    out += block('## 3. Per dataset (kept comparisons)', 'ds', dsets)

    # 4. what the gain goes with
    out += ['', '## 4. What the size of the gain goes with']
    best = {}
    for s in kept:
        k = (s['ds'], s['H'])
        best[k] = min(best.get(k, float('inf')), s['alone'])
    gap = lambda s: s['alone'] / best[(s['ds'], s['H'])] - 1
    rows = [f'All kept settings: rho = {spearman([gap(s) for s in kept], [s["d"] for s in kept]):+.2f} '
            f'(n = {len(kept)}).']
    per = []
    for ds in dsets:
        v = [s for s in kept if s['ds'] == ds]
        if len({s['b'] for s in v}) >= 3:
            per.append(f'{ds} {spearman([gap(s) for s in v], [s["d"] for s in v]):+.2f} (n = {len(v)})')
    out += ['', '**a. Headroom.** Spearman correlation between the change and how far the backbone alone is behind the '
            'best backbone alone of the same dataset and horizon (a negative value: the further behind, the larger the '
            'gain). ' + ' '.join(rows) + ' Per dataset: ' + ', '.join(per) + '.']
    out += ['', '**b. Best + DLT result against the best backbone alone** (kept comparisons; the backbone that reaches '
            'each):', '', '| dataset | H | best alone | best + DLT | change |', '|---|---:|---|---|---:|']
    for ds in dsets:
        for H in pt.horizons(ds):
            v = [s for s in kept if s['ds'] == ds and s['H'] == H]
            if len(v) < 3:
                continue
            ba, bd = min(v, key=lambda s: s['alone']), min(v, key=lambda s: s['dlt'])
            out.append(f'| {ds} | {H} | {ba["alone"]:.4f} {NAME.get(ba["b"], ba["b"])} | {bd["dlt"]:.4f} '
                       f'{NAME.get(bd["b"], bd["b"])} | {fmt(bd["dlt"] / ba["alone"] - 1)} |')
    out += ['', '**c. Horizon.** Mean change per horizon (kept comparisons):', '']
    for name, group in (('ETT and Exchange', ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange')), ('Weather', ('Weather',)),
                        ('Electricity', ('ECL',)), ('Traffic', ('Traffic',)), ('Solar', ('Solar',)), ('PEMS', PEMS)):
        hs = sorted({s['H'] for s in kept if s['ds'] in group})
        if hs:
            out.append(f'- {name}: ' + ', '.join(
                f'{H}: {fmt(statistics.mean(s["d"] for s in kept if s["ds"] in group and s["H"] == H))}' for H in hs))
    if a.diag and Path(a.diag).exists():
        diag = {r['dataset']: r for r in json.load(open(a.diag))}
        out += ['', '**d. Data.** Linear diagnostic (scripts/plugin/leadlag_diag.py, mean over the horizons): change of the '
                'test MSE of a shared channel-independent ridge forecaster (SHARED, as DLinear) when per-variable '
                'weights are added (OWN), when the input windows of the 8 DLT neighbours are added (NB, a cross-variable '
                'branch on a shared backbone, as DLT on DLinear), and what the neighbours add beyond the own past (BOTH '
                'against OWN); training split in brackets. Lag 0: share of the neighbour pairs whose cross-correlation '
                'peaks at lag 0. Level r: mean |correlation| of a variable with its first neighbour.', '',
                '| dataset | lag 0 | level r | OWN | NB | BOTH vs OWN | DLinear + DLT | median over the backbones |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        xs, ys, zs = [], [], []
        for ds, r in diag.items():
            hv = r['horizons'].values()
            mm = lambda k: statistics.mean(v[k] for v in hv)
            cell = lambda k: f'{fmt(mm(k + "_test"))} ({fmt(mm(k + "_train"))})'
            dl = [s for s in kept if s['ds'] == ds and s['b'] == 'DLinear']
            dlc = statistics.mean(s['dlt'] for s in dl) / statistics.mean(s['alone'] for s in dl) - 1 if dl else None
            med = statistics.median(s['d'] for s in kept if s['ds'] == ds) if any(s['ds'] == ds for s in kept) else None
            out.append(f'| {ds} | {100 * r["lag0_share"]:.0f} % | {r["level_corr_first"]:.2f} | {cell("own")} | '
                       f'{cell("nb")} | {cell("lead")} | {fmt(dlc) if dlc is not None else "-"} | '
                       f'{fmt(med) if med is not None else "-"} |')
            if dlc is not None:
                xs.append(mm('nb_test')); ys.append(dlc)
            if med is not None:
                zs.append((mm('nb_test'), med))
        if len(xs) >= 4:
            out += ['', f'Spearman correlation over the datasets, NB against DLinear + DLT: {spearman(xs, ys):+.2f} '
                    f'(n = {len(xs)}); NB against the median over the backbones: '
                    f'{spearman([z[0] for z in zs], [z[1] for z in zs]):+.2f} (n = {len(zs)}).']
    text = '\n'.join(out) + '\n'
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == '__main__':
    main()
