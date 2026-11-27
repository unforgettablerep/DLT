"""Table of the all-neighbour suite: python scripts/ablation/allnb_table.py [--out FILE].

Reads output/ablation/allnb.tsv (suite allnb of scripts/ablation/make_jobs.py: all other variables as neighbours,
dense path, arm _all, against the CCF top-8, arm _ccf) and, for DLT-dyn with the CCF top-8 at the long horizons, the
runs of output/long_term_forecast/{long_small,long_large}.tsv. Writes output/ablation/allnb.md: test MSE (mean ± std
over the seeds) of both arms, the change of _all against _ccf and a Welch t-test over the seeds, per dataset, horizon
and model; for iTransformer on PEMS08 also the backbone alone. Arms are compared only within one device.
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

LONG = [OUTPUT / "long_term_forecast" / f for f in ("long_small.tsv", "long_large.tsv")]


def ms(v):
    if not v:
        return ""
    return "{:.4f}".format(statistics.mean(v)) + (" ± {:.4f}".format(statistics.stdev(v)) if len(v) > 1 else "")


def vs(v, ref):
    if not v or not ref:
        return ""
    s = "{:+.1f} %".format(100 * (statistics.mean(v) / statistics.mean(ref) - 1))
    return s + (" (p {:.3f})".format(stats.ttest_ind(v, ref, equal_var=False).pvalue) if len(v) > 1 and len(ref) > 1 else "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUTPUT / "ablation" / "allnb.md"))
    a = p.parse_args()
    C = collections.defaultdict(list)                      # (device, dataset, H, model, arm) -> MSEs
    for r in latest(read_results(OUTPUT / "ablation" / "allnb.tsv")):
        if r.get("status") != "ok":
            continue
        ds, H, rest = r["des"].split("_", 2)               # e.g. Weather_96_dlt_all, PEMS08_12_iTransformer_dlt_ccf
        model, arm = (rest.split("_", 1) if rest.startswith("iTransformer") else rest.rsplit("_", 1))
        C[(device_class(r), ds, int(H), model, arm)].append(r["mse"])
    for r in latest(read_results(*LONG)):                  # DLT-dyn with the CCF top-8 at the long horizons
        if r.get("status") == "ok" and r.get("model") == "DLT" and r["des"].endswith("_dltdyn"):
            C[(device_class(r), r["dataset"], int(r["pred_len"]), "dltdyn", "ccf")].append(r["mse"])
    order = list(DATASET_META)
    settings = sorted({k[:3] for k in C if k[4] == "all"}, key=lambda s: (s[0] or "", order.index(s[1]) if s[1] in order else 99, s[2]))
    L = ["# All other variables as neighbours", "",
         "Test MSE, mean ± std over the seeds: the CCF top-8 (default) and all other variables as neighbours (dense",
         "path); change of all against CCF with a Welch t-test over the seeds. Both arms from the same device.", "",
         "| device | dataset | H | model | CCF top-8 | all neighbours | change |", "|---|---|---:|---|---:|---:|---:|"]
    for dev, ds, H in settings:
        for model, label in (("dlt", "DLT"), ("dltdyn", "DLT-dyn")):
            ccf, al = C.get((dev, ds, H, model, "ccf"), []), C.get((dev, ds, H, model, "all"), [])
            if ccf or al:
                L.append("| {} | {} | {} | {} | {} | {} | {} |".format(dev, ds, H, label, ms(ccf), ms(al), vs(al, ccf)))
    plug = sorted({k[:3] for k in C if k[3] == "iTransformer"})
    if plug:
        L += ["", "## iTransformer + DLT-dyn (joint training, branch at 3e-3)", "",
              "| device | dataset | H | alone | + DLT-dyn, CCF top-8 | vs alone | + DLT-dyn, all neighbours | vs alone | vs CCF |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for dev, ds, H in plug:
            g = lambda arm: C.get((dev, ds, H, "iTransformer", arm), [])
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(dev, ds, H, ms(g("alone")), ms(g("dlt_ccf")), vs(g("dlt_ccf"), g("alone")),
                                                                         ms(g("dlt_all")), vs(g("dlt_all"), g("alone")), vs(g("dlt_all"), g("dlt_ccf"))))
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
