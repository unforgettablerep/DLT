"""Table of the guard study: python scripts/ablation/guard_table.py [--files F ...] [--out FILE].

Reads output/ablation/guard.tsv (suite guard of scripts/ablation/make_jobs.py: stand-alone DLT-dyn with the guards of
the plug-in) and writes output/ablation/guard.md. For every setting: test MSE of each arm (mean ± std over the seeds),
its change against DLT-dyn as in the paper (arm dyn) and against the self head alone (arm ci), a Welch t-test against
dyn, the share of variables whose correction the per-variable selection kept, and the epoch of the best validation
loss (two-stage arms: stage 1 + branch). Then, per arm, the change against ci averaged over the weakly coupled
settings (ETTh1, ETTh2, Exchange: the harm to remove) and the coupled ones (Weather, PEMS08: the gain to keep).
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

ARMS = {
    "ci": "self head alone (CI)",
    "dyn": "DLT-dyn (paper)",
    "dyn_sel": "+ per-variable selection",
    "dyn_lr": "branch at 1/10 learning rate",
    "dyn_lr_sel": "1/10 learning rate + selection",
    "dyn_2s": "two-stage training",
    "dyn_2s_sel": "two-stage + selection",
}
WEAK = ("ETTh1", "ETTh2", "Exchange")


def mean(v):
    return statistics.mean(v) if v else float("nan")


def ms(v):
    return "{:.4f}".format(mean(v)) + (" ± {:.4f}".format(statistics.stdev(v)) if len(v) > 1 else "")


def pct(x):
    return "{:+.1f} %".format(100 * x) if x == x else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="+")
    p.add_argument("--out", default=str(OUTPUT / "ablation" / "guard.md"))
    a = p.parse_args()
    cells = collections.defaultdict(list)                       # (dataset, H, arm) -> records
    for r in latest(read_results(*(a.files or [OUTPUT / "ablation" / "guard.tsv"]))):
        if r.get("status") != "ok" or r.get("model") != "DLT":
            continue
        ds, H, arm = r["des"].split("_", 2)
        if arm in ARMS:
            cells[(ds, int(H), arm)].append(r)
    settings = sorted({(ds, H) for ds, H, _ in cells}, key=lambda s: (s[0] not in WEAK, s[0], s[1]))
    L = ["# Stand-alone DLT-dyn with the guards of the plug-in", "",
         "Test MSE (mean ± std over seeds) of the arms of suite `guard`; vs dyn / vs ci = change of the mean MSE against",
         "DLT-dyn as in the paper / the self head alone; p = Welch t-test against dyn; kept = share of variables whose",
         "correction the per-variable selection kept; best ep. = epoch of the best validation loss (two-stage: stage 1 +",
         "branch epochs).", ""]
    summary = collections.defaultdict(lambda: {"weak": [], "coupled": []})
    for ds, H in settings:
        dyn = [r["mse"] for r in cells.get((ds, H, "dyn"), [])]
        ci = [r["mse"] for r in cells.get((ds, H, "ci"), [])]
        L += ["## {} H={}".format(ds, H), "", "| arm | n | MSE | vs dyn | vs ci | p | kept | best ep. |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, label in ARMS.items():
            rs = cells.get((ds, H, arm))
            if not rs:
                continue
            v = [r["mse"] for r in rs]
            vs_dyn = mean(v) / mean(dyn) - 1 if dyn and arm != "dyn" else float("nan")
            vs_ci = mean(v) / mean(ci) - 1 if ci and arm != "ci" else float("nan")
            pv = "{:.3f}".format(stats.ttest_ind(v, dyn, equal_var=False).pvalue) if arm != "dyn" and len(v) > 1 and len(dyn) > 1 else ""
            kept = [r["dlt_kept"] for r in rs if r.get("dlt_kept") is not None]
            best = mean([r["best_epoch"] for r in rs if r.get("best_epoch") is not None])
            if any(r.get("branch_epochs") is not None for r in rs):
                best = "{:.1f}+{:.1f}".format(best, mean([r.get("branch_best_epoch") or 0 for r in rs]))
            else:
                best = "{:.1f}".format(best)
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(label, len(v), ms(v), pct(vs_dyn), pct(vs_ci), pv,
                                                                        "{:.0%}".format(mean(kept)) if kept else "", best))
            if arm != "ci" and vs_ci == vs_ci:
                summary[arm]["weak" if ds in WEAK else "coupled"].append(vs_ci)
        L.append("")
    L += ["## Over the settings: change against the self head alone (CI)", "",
          "| arm | weakly coupled: mean | worst | coupled: mean | least gain |", "|---|---:|---:|---:|---:|"]
    for arm, label in ARMS.items():
        s = summary.get(arm)
        if not s:
            continue
        w, c = s["weak"], s["coupled"]
        L.append("| {} | {} | {} | {} | {} |".format(label, pct(mean(w)), pct(max(w)) if w else "", pct(mean(c)), pct(max(c)) if c else ""))
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
