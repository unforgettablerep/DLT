"""Stand-alone DLT-dyn against the same model without the cross-variable branch (CI), per dataset and horizon.

    python scripts/multivariate_forecasting/long_table.py [--device all|cpu|gpu] [--out FILE]

Reads DLT-dyn from output/long_term_forecast/{long_small,long_large,long_extra}.tsv, CI from output/ablation/ci_long.tsv
and the searched configurations from output/hyperparam/tuned.tsv where they exist, and writes
output/long_term_forecast/summary.md: test MSE and MAE of every arm (mean ± std over the seeds), the change of DLT-dyn
(and of the searched configuration) against CI with a Welch t-test over the seeds, and per dataset the mean change over
its horizons. Runs whose status is not ok are left out and counted; the device column names where the runs ran.
"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.configs.dlt_profiles import DATASET_META  # noqa: E402
from src.utils.paths import OUTPUT  # noqa: E402
from src.utils.result_io import device_class, latest, read_results  # noqa: E402

STORES = {"dltdyn": [OUTPUT / "long_term_forecast" / f for f in ("long_small.tsv", "long_large.tsv", "long_extra.tsv")],
          "ci": [OUTPUT / "ablation" / "ci_long.tsv"],
          "tuned": [OUTPUT / "hyperparam" / "tuned.tsv"]}


def ms(v):
    if not v:
        return ""
    return "{:.4f}".format(statistics.mean(v)) + (" ± {:.4f}".format(statistics.stdev(v)) if len(v) > 1 else "")


def change(v, ref):
    return statistics.mean(v) / statistics.mean(ref) - 1 if v and ref else None


def pct(x):
    return "" if x is None else "{:+.1f} %".format(100 * x)


def pval(v, ref):
    return "{:.3f}".format(stats.ttest_ind(v, ref, equal_var=False).pvalue) if len(v) > 1 and len(ref) > 1 else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="all", choices=["all", "cpu", "gpu"])
    p.add_argument("--out", default=str(OUTPUT / "long_term_forecast" / "summary.md"))
    a = p.parse_args()
    cells, devices, skipped = collections.defaultdict(list), collections.defaultdict(set), 0
    for arm, files in STORES.items():
        for r in latest(read_results(*files)):
            if r.get("model") != "DLT" or not (r.get("des") or "").endswith("_" + arm):
                continue
            if a.device != "all" and device_class(r) != a.device:
                continue
            if r.get("status") != "ok":
                skipped += 1
                continue
            key = (r["dataset"], int(r["pred_len"]))
            cells[key + (arm,)].append(r)
            devices[r["dataset"]].add(device_class(r) or "?")
    order = list(DATASET_META)
    settings = sorted({k[:2] for k in cells}, key=lambda s: (order.index(s[0]) if s[0] in order else 99, s[1]))
    L = ["# Stand-alone DLT-dyn against CI", "",
         "Test MSE and MAE, mean ± std over the seeds (n = seeds of DLT-dyn / CI). Change = DLT-dyn / CI - 1; p = Welch",
         "t-test over the seeds; tuned = the configuration of the validation search (hyperparam/tuned.tsv).",
         "{} runs whose status is not ok are left out.".format(skipped), "",
         "| dataset | H | device | n | DLT-dyn MSE | CI MSE | change | p | DLT-dyn MAE | CI MAE | change | tuned MSE | vs CI |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    per_ds = collections.defaultdict(lambda: collections.defaultdict(list))
    for ds, H in settings:
        get = lambda arm, m: [r[m] for r in cells.get((ds, H, arm), [])]
        dyn, ci, tu = get("dltdyn", "mse"), get("ci", "mse"), get("tuned", "mse")
        c_mse, c_mae, c_tu = change(dyn, ci), change(get("dltdyn", "mae"), get("ci", "mae")), change(tu, ci)
        for name, c in (("mse", c_mse), ("mae", c_mae), ("tuned", c_tu)):
            if c is not None:
                per_ds[ds][name].append(c)
        L.append("| {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            ds, H, "/".join(sorted(devices[ds])), len(dyn), len(ci), ms(dyn), ms(ci), pct(c_mse), pval(dyn, ci),
            ms(get("dltdyn", "mae")), ms(get("ci", "mae")), pct(c_mae), ms(tu), pct(c_tu)))
    L += ["", "## Mean change over the horizons", "", "| dataset | horizons | DLT-dyn vs CI, MSE | MAE | tuned vs CI, MSE |",
          "|---|---:|---:|---:|---:|"]
    for ds in sorted(per_ds, key=lambda d: order.index(d) if d in order else 99):
        c = per_ds[ds]
        m = lambda k: pct(statistics.mean(c[k])) if c[k] else ""
        L.append("| {} | {} | {} | {} | {} |".format(ds, len(c["mse"]), m("mse"), m("mae"), m("tuned")))
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
