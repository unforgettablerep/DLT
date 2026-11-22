"""Take over the run records of a pre-refactor work directory into the result stores of output/.

    python scripts/utils/migrate_work.py [work]            # dry run: what would be added to which store
    python scripts/utils/migrate_work.py [work] --apply    # append those rows

Before the refactor every suite wrote one JSON line per run into work/results_<suite>.jsonl. Now every suite has a
tracked TSV store, output/<category>/<suite>.tsv. Each record is converted (src/utils/result_io.py, migrate_record)
and appended to the store of its suite unless the store already holds the same run (same seed, dataset, horizon,
model and test MSE), so the command can be repeated. The queue runners then skip the runs that are done.
Checkpoints (work/ckpt) are not moved; scripts/mechanism/kernels.py reads the P0 checkpoints there directly.
"""
import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import OUTPUT, REPO, rel  # noqa: E402
from src.utils.result_io import _read_jsonl, append_result, migrate_record, read_results  # noqa: E402

CATEGORY = {
    "long_small": "long_term_forecast", "long_large": "long_term_forecast", "long_extra": "long_term_forecast",
    "pems12": "long_term_forecast", "baselines": "benchmark", "plugin": "plugin", "plugin_all": "plugin",
    "plugin_fusion": "plugin", "plugin_large": "plugin", "plugin_probe": "plugin", "p0": "ablation",
    "ci_long": "ablation", "allnb": "ablation", "sensitivity": "hyperparam", "neighbours": "hyperparam",
    "eff_tune": "efficiency",
}


def category(suite):
    if suite in CATEGORY:
        return CATEGORY[suite]
    if suite.startswith("hp") or suite.startswith("tuned"):
        return "hyperparam"                       # hp<tag>, tuned<tag> of scripts/hyperparam/hpsearch.py
    return None


def key(row):
    """Identity of a run record: its test MSE together with seed, data, horizon and model (job names were not
    always stored); runs without a finite MSE are told apart by their job name and kernel."""
    mse = row.get("mse")
    run = (int(row.get("seed") or 2021), row.get("dataset"), row.get("pred_len"), row.get("model"), row.get("plug_base"))
    if isinstance(mse, float) and math.isfinite(mse):
        return run + (round(mse, 12),)
    return run + (str(mse), row.get("des"), row.get("dlt_variant"), row.get("dlt_gate"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("work", nargs="?", default=str(REPO / "work"))
    p.add_argument("--apply", action="store_true", help="append the rows (default: only report)")
    a = p.parse_args()
    files = sorted(Path(a.work).glob("results_*.jsonl"))
    if not files:
        print("no results_*.jsonl in {}".format(a.work))
        return
    total = 0
    for f in files:
        suite = f.stem[len("results_"):]
        cat = category(suite)
        if cat is None:
            print("skipped {}: no store for suite {}".format(f.name, suite))
            continue
        store = OUTPUT / cat / "{}.tsv".format(suite)
        have = {key(r) for r in read_results(store)}
        new = []
        for rec in _read_jsonl(str(f)):
            row = migrate_record(rec)
            if key(row) not in have:
                have.add(key(row))
                new.append(row)
        total += len(new)
        print("{:40s} -> {:45s} {:5d} new of {}".format(f.name, rel(store), len(new), len(have)))
        if a.apply and new:
            os.makedirs(store.parent, exist_ok=True)
            for row in new:
                append_result(str(store), row)
    print("{} rows {}".format(total, "appended" if a.apply else "to append (dry run; --apply appends them)"))


if __name__ == "__main__":
    main()
