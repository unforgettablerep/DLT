"""Tuning DLT-dyn in the efficiency setting of the S-Mamba paper: compare the recipes of the eff_tune suite.

    python scripts/efficiency/eff_tune.py [--files F ...] [--out FILE]

Reads output/efficiency/eff_tune.tsv (or --files). One recipe = one --tag of
scripts/efficiency/make_jobs.py eff_tune (job names <ds>_<H>_dltdyn_<tag>). Per dataset, the recipes are ranked by
their best validation loss (the selection criterion; the test split plays no part in it). The table also gives the
test MSE / MAE, the epochs run, the best epoch and the training time per epoch; the selected recipe is marked with *.
Published S-Mamba results at the same setting are printed for reference (assets/reported.json). Writes
output/efficiency/eff_tune.md (or --out). selected() / recipe() give the chosen recipe per dataset to
scripts/efficiency/run_efficiency.py --tuned.
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.configs.dlt_profiles import EFFICIENCY_SETTING  # noqa: E402
from src.utils.paths import OUTPUT, REPO  # noqa: E402
from src.utils.result_io import latest, read_results  # noqa: E402

SETS = list(EFFICIENCY_SETTING.items())
STORE = OUTPUT / 'efficiency' / 'eff_tune.tsv'
SUPERSEDED = OUTPUT / 'efficiency' / 'eff_tune_superseded.tsv'


def load(files=None):
    cells = collections.defaultdict(list)
    for r in latest(read_results(*(files or [STORE]))):
        ds, H, _, tag = r['des'].split('_', 3)
        cells[(ds, int(H), tag)].append(r)
    return cells


def selected(cells=None):
    """{dataset: (tag, a record of the recipe with the lowest mean validation loss)} over the eff_tune records."""
    cells = load() if cells is None else cells
    out = {}
    for ds, H in SETS:
        tags = [t for (d, h, t) in cells if d == ds and h == H]
        if tags:
            t = min(tags, key=lambda t: statistics.mean(r['best_vali'] for r in cells[(ds, H, t)]))
            out[ds] = (t, cells[(ds, H, t)][0])
    return out


def recipe(r, impl='basis'):
    """DLT arguments of a tuning record (the learning rate does not affect time or memory)."""
    return (f"--dlt_variant {r.get('dlt_variant') or 'dltdyn'} --dlt_k {r['dlt_k']} --dlt_L {r['dlt_L']} "
            f"--dlt_R {r['dlt_R']} --dlt_impl {impl}" + (' --dlt_neighbours all' if r.get('dlt_neighbours') == 'all' else ''))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+')
    p.add_argument('--out', default=str(OUTPUT / 'efficiency' / 'eff_tune.md'))
    a = p.parse_args()
    cells = load(a.files)
    rep = json.load(open(REPO / 'assets' / 'reported.json'))['smamba_paper']
    m = lambda rs, k: statistics.mean(r[k] for r in rs if r.get(k) is not None)
    out = ['# DLT-dyn in the efficiency setting of the S-Mamba paper: recipes ranked by validation loss', '',
           'Batch 16, epochs of the official S-Mamba scripts, patience 3, no AMP. * = selected (lowest validation loss).',
           's / epoch: wall-clock time of the tuning run on a shared CPU (one thread per run, several runs in parallel);',
           'it is not an efficiency measurement (see scripts/efficiency/run_efficiency.py --setting smamba).', '']
    for ds, H in SETS:
        tags = sorted({t for (d, h, t) in cells if d == ds and h == H},
                      key=lambda t: m(cells[(ds, H, t)], 'best_vali'))
        if not tags:
            continue
        s = rep['S_Mamba'][ds][str(H)]
        it = rep['iTransformer'][ds][str(H)]
        out += [f'## {ds}, horizon {H} (published: S-Mamba {s[0]:.3f} / {s[1]:.3f}, iTransformer {it[0]:.3f} / {it[1]:.3f})',
                '', '| recipe | seeds | validation loss | test MSE | test MAE | epochs | best epoch | s / epoch | change vs S-Mamba |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for i, t in enumerate(tags):
            rs = cells[(ds, H, t)]
            sec = statistics.mean(statistics.mean(r['epoch_secs']) for r in rs if r.get('epoch_secs'))
            mse = m(rs, 'mse')
            out.append(f'| {t}{" *" if i == 0 else ""} | {len(rs)} | {m(rs, "best_vali"):.4f} | {mse:.4f} | '
                       f'{m(rs, "mae"):.4f} | {m(rs, "epochs"):.1f} | {m(rs, "best_epoch"):.1f} | {sec:.0f} | '
                       f'{100 * (mse / s[0] - 1):+.1f} % |')
        out.append('')
        best = cells[(ds, H, tags[0])][0]
        out += [f'Selected recipe: `{tags[0]}`: `{recipe(best)}`, learning rate {best["learning_rate"]:g}.', '']
    if SUPERSEDED.exists():
        out += ['`eff_tune_superseded.tsv`: the first run of r1 on ETTm1, on the kernel path; it was repeated on',
                'the basis path used by all other runs and gave the same validation loss and test MSE to four decimals.', '']
    out += ['Time and memory of the selected recipes and of the baselines in this setting (one command; the recipes are',
            'read from the tuning records):', '',
            '    python scripts/efficiency/run_efficiency.py --setting smamba --device gpu --tuned', '']
    text = '\n'.join(out) + '\n'
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)


if __name__ == '__main__':
    main()
