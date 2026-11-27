"""What the lags of DLT-dyn do: python scripts/ablation/lagrole_table.py [--out FILE].

Reads output/ablation/lagrole.tsv (suite lagrole of scripts/ablation/make_jobs.py: CI, the lag-0 kernel with
input-dependent weights (scalar-dyn) and DLT-dyn with the maximum lag L = 1, 2, 4, 8, 12, 48 on Weather-96 and
PEMS08-96, three seeds, the same CCF neighbours (search over lags 0..48) in every arm, the test MSE of every forecast
step in the record) and writes output/ablation/lagrole.md:
  1. the test MSE by maximum lag: mean ± std, the change against lag 0 with a Welch t-test, and the share of the gain of
     the full kernel (L 48) that L reaches;
  2. the change against lag 0 by forecast step (bins of the horizon), from the per-step MSE averaged over the seeds: a
     lead of the neighbours by tau steps can only inform the first tau steps, a better estimate of the current state
     helps every step;
  3. checks: the arms ci, scalardyn and dltdyn against the same runs of P0 (output/ablation/p0.tsv), and L 12 against
     the sensitivity run with the neighbours searched over lags 0..12 only (output/hyperparam/sensitivity.tsv).
"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import OUTPUT  # noqa: E402
from src.utils.result_io import latest, read_results  # noqa: E402

SETTINGS = ("Weather_96", "PEMS08_96")
LAGS = (1, 2, 4, 8, 12)
ARMS = ["ci", "scalardyn"] + ["dltdyn_L{}".format(L) for L in LAGS] + ["dltdyn"]
LABEL = {"ci": "CI (no cross branch)", "scalardyn": "lag 0 (scalar-dyn)", "dltdyn": "L 48 (DLT-dyn)",
         **{"dltdyn_L{}".format(L): "L {}".format(L) for L in LAGS}}
BINS = ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 96))


def mean_std(v):
    return "{:.4f}".format(statistics.mean(v)) + (" ± {:.4f}".format(statistics.stdev(v)) if len(v) > 1 else "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUTPUT / "ablation" / "lagrole.md"))
    a = p.parse_args()
    runs = collections.defaultdict(dict)                    # (setting, arm) -> {seed: record}
    for r in latest(read_results(OUTPUT / "ablation" / "lagrole.tsv")):
        if r.get("status") == "ok":
            s, arm = r["des"].split("_", 2)[:2], r["des"].split("_", 2)[2]
            runs[("_".join(s), arm)][r["seed"]] = r
    mse = lambda s, arm: [r["mse"] for r in runs[(s, arm)].values()]
    L = ["# What the lags of DLT-dyn do", "",
         "Suite `lagrole` (CPU, three seeds): every arm has the same model and neighbours (CCF search over lags 0..48);",
         "only the maximum lag L of the kernels changes. Lag 0 is the lag-0 kernel with input-dependent weights (P0's",
         "scalar-dyn); L 48 is DLT-dyn as in the paper.", "",
         "## 1. Test MSE by maximum lag", "",
         "| | " + " | ".join("{0} MSE | vs lag 0 | p | share of the L-48 gain".format(s) for s in SETTINGS) + " |",
         "|---|" + "---:|---:|---:|---:|" * len(SETTINGS)]
    for arm in ARMS:
        cells = []
        for s in SETTINGS:
            v, z, f = mse(s, arm), mse(s, "scalardyn"), mse(s, "dltdyn")
            if not v:
                cells += ["", "", "", ""]
                continue
            ch = "{:+.1f} %".format(100 * (statistics.mean(v) / statistics.mean(z) - 1)) if z and arm != "scalardyn" else ""
            pv = "{:.3f}".format(stats.ttest_ind(v, z, equal_var=False).pvalue) if len(v) > 1 and len(z) > 1 and arm not in ("scalardyn",) else ""
            share = ""
            if arm.startswith("dltdyn") and z and f and statistics.mean(z) != statistics.mean(f):
                share = "{:.0%}".format((statistics.mean(z) - statistics.mean(v)) / (statistics.mean(z) - statistics.mean(f)))
            cells += ["{} ({})".format(mean_std(v), len(v)), ch, pv, share]
        L.append("| {} | {} |".format(LABEL[arm], " | ".join(cells)))
    L += ["", "## 2. Change of the MSE against lag 0 by forecast step", "",
          "Per-step test MSE averaged over the seeds, then over the steps of each bin; the CI row is CI against lag 0 (what",
          "the lag-0 neighbours add, with the sign reversed). Last column: the share of the total MSE reduction against lag 0",
          "that falls on steps 1 to 8 (8 % of the horizon).", ""]
    for s in SETTINGS:
        steps = {arm: np.mean([r["mse_step"] for r in runs[(s, arm)].values() if r.get("mse_step")], axis=0)
                 for arm in ARMS if any(r.get("mse_step") for r in runs[(s, arm)].values())}
        if "scalardyn" not in steps:
            continue
        z = steps["scalardyn"]
        L += ["### {}".format(s), "", "| vs lag 0 | " + " | ".join("step {}".format(lo) if lo == hi else "steps {}-{}".format(lo, hi) for lo, hi in BINS)
              + " | all | share of steps 1-8 |", "|---|" + "---:|" * (len(BINS) + 2)]
        for arm in ARMS:
            if arm == "scalardyn" or arm not in steps:
                continue
            x = steps[arm]
            cells = ["{:+.1f} %".format(100 * (x[lo - 1:hi].mean() / z[lo - 1:hi].mean() - 1)) for lo, hi in BINS]
            cells.append("{:+.1f} %".format(100 * (x.mean() / z.mean() - 1)))
            gain = z - x
            cells.append("{:.0%}".format(gain[:8].sum() / gain.sum()) if gain.sum() > 0 else "")
            L.append("| {} | {} |".format(LABEL[arm], " | ".join(cells)))
        L.append("")
    # checks
    L += ["## 3. Checks", ""]
    p0 = collections.defaultdict(dict)
    for r in latest(read_results(OUTPUT / "ablation" / "p0.tsv")):
        if r.get("status") == "ok" and (r.get("dlt_gate") or "none") == "none":
            p0[("{}_{}".format(r["dataset"], r["pred_len"]), r["dlt_variant"])][r["seed"]] = r["mse"]
    diffs = []
    for s in SETTINGS:
        for arm in ("ci", "scalardyn", "dltdyn"):
            for seed, r in runs[(s, arm)].items():
                if seed in p0[(s, arm)]:
                    diffs.append(abs(r["mse"] - p0[(s, arm)][seed]))
    if diffs:
        L.append("- The arms ci, scalardyn and dltdyn against the same runs of P0: {} runs, largest difference of the test MSE {:.1e}.".format(len(diffs), max(diffs)))
    sens = collections.defaultdict(list)
    for r in latest(read_results(OUTPUT / "hyperparam" / "sensitivity.tsv")):
        if r.get("status") == "ok" and r["des"].endswith("_L12"):
            sens["{}_{}".format(r["dataset"], r["pred_len"])].append(r["mse"])
    for s in SETTINGS:
        v = mse(s, "dltdyn_L12")
        if v and sens.get(s):
            L.append("- {} with L 12: {} with the neighbours of the search over lags 0..48, {} with those of the search over "
                     "lags 0..12 (sensitivity study).".format(s, mean_std(v), mean_std(sens[s])))
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
