"""Tables of the sensitivity studies: python scripts/hyperparam/sens_table.py [--out FILE].

Reads output/hyperparam/sensitivity.tsv (DLT-dyn with L, R or k changed one at a time on ETTh1-96, Weather-96 and
PEMS08-96) and output/hyperparam/neighbours.tsv (DLT and DLT-dyn with 8 random or all other variables as neighbours
instead of the CCF top-8, Weather-96 and PEMS08-96). The defaults (L 48, R 4, k 8, CCF neighbours) are the runs of the
P0 grid (output/ablation/p0.tsv), the same configuration and seeds. Writes output/hyperparam/sensitivity.md: test MSE
(mean ± std over the seeds), change against the default and a Welch t-test over the seeds. ETTh1 has seven
variables, so k is capped at 6 there: k 8 and k 16 are the same model.
"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import OUTPUT  # noqa: E402
from src.utils.result_io import latest, read_results  # noqa: E402

SETTINGS = (("ETTh1", 96), ("Weather", 96), ("PEMS08", 96))
FACTORS = (("maximum lag L", "dlt_L", (12, 24, 48, 96), 48), ("basis size R", "dlt_R", (2, 4, 8), 4),
           ("neighbours k", "dlt_k", (4, 8, 16), 8))
NEIGHBOURS = (("ccf", "CCF top-8 (default)"), ("random", "8 random"), ("all", "all other variables"))


def key(r):
    return (r["dataset"], int(r["pred_len"]), r.get("dlt_variant"), int(r.get("dlt_L") or 48), int(r.get("dlt_R") or 4),
            int(r.get("dlt_k") or 8), r.get("dlt_neighbours") or "ccf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUTPUT / "hyperparam" / "sensitivity.md"))
    a = p.parse_args()
    S = collections.defaultdict(list)
    for r in latest(read_results(OUTPUT / "hyperparam" / "sensitivity.tsv", OUTPUT / "hyperparam" / "neighbours.tsv")):
        if r.get("status") == "ok" and r.get("model") == "DLT":
            S[key(r)].append(r["mse"])
    for r in latest(read_results(OUTPUT / "ablation" / "p0.tsv")):       # the defaults: P0, no gate
        if r.get("status") == "ok" and r.get("dlt_variant") in ("dlt", "dltdyn") and (r.get("dlt_gate") or "none") == "none":
            S[key(r)].append(r["mse"])

    def cell(k, ref):
        v, d = S.get(k), S.get(ref)
        if not v:
            return ""
        s = "{:.4f}".format(statistics.mean(v)) + (" ± {:.4f}".format(statistics.stdev(v)) if len(v) > 1 else "")
        if d and k != ref:
            s += " ({:+.1f} %".format(100 * (statistics.mean(v) / statistics.mean(d) - 1))
            s += ", p {:.3f})".format(stats.ttest_ind(v, d, equal_var=False).pvalue) if len(v) > 1 and len(d) > 1 else ")"
        return s

    L = ["# Sensitivity of DLT-dyn and neighbour selection", "",
         "Test MSE, mean ± std over the seeds; in brackets the change against the default (L 48, R 4, k 8, CCF top-8",
         "neighbours: the P0 runs) and a Welch t-test over the seeds.", "",
         "## One factor at a time (DLT-dyn)", "",
         "| factor | value | " + " | ".join("{}-{}".format(ds, H) for ds, H in SETTINGS) + " |", "|---|---|" + "---:|" * len(SETTINGS)]
    for name, col, values, default in FACTORS:
        for x in values:
            cells = []
            for ds, H in SETTINGS:
                d = (ds, H, "dltdyn", 48, 4, 8, "ccf")
                k = (ds, H, "dltdyn", x if col == "dlt_L" else 48, x if col == "dlt_R" else 4, x if col == "dlt_k" else 8, "ccf")
                cells.append("= k 8 (capped at 6)" if ds == "ETTh1" and col == "dlt_k" and x == 16 else cell(k, d))
            L.append("| {} | {}{} | {} |".format(name, x, " (default)" if x == default else "", " | ".join(cells)))
    L += ["", "## Neighbour selection", "", "| model | neighbours | " + " | ".join("{}-{}".format(ds, H) for ds, H in SETTINGS[1:]) + " |",
          "|---|---|" + "---:|" * (len(SETTINGS) - 1)]
    for v, vn in (("dlt", "DLT"), ("dltdyn", "DLT-dyn")):
        for nb, label in NEIGHBOURS:
            cells = [cell((ds, H, v, 48, 4, 8, nb), (ds, H, v, 48, 4, 8, "ccf")) for ds, H in SETTINGS[1:]]
            L.append("| {} | {} | {} |".format(vn, label, " | ".join(cells)))
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
