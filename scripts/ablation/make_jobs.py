"""Job lists of the ablations: p0 (controlled comparison of cross-variable kernels), ci_long (DLT without the branch), allnb (all variables as neighbours),
guard (stand-alone DLT-dyn with the guards of the plug-in against harm on weakly coupled data), lagrole (what the lags do).

lagrole: Weather-96 and PEMS08-96, three seeds, every run with the test MSE of each forecast step (--step_metrics):
    ci, scalardyn, dltdyn   the P0 configurations of CI, the lag-0 kernel with input-dependent weights and DLT-dyn (L 48)
    dltdyn_L<L>             DLT-dyn with the maximum lag L = 1, 2, 4, 8, 12 (the sensitivity study has 12, 24, 96)
  A lead of the neighbours by tau steps can only inform the first tau steps of the horizon; a better estimate of the
  current state helps every step. The per-step errors tell the two apart (scripts/ablation/lagrole_table.py).

guard: seven arms on the weakly coupled ETTh1, ETTh2 and Exchange and the coupled Weather and PEMS08 (GUARD_SETTINGS),
three seeds, the recipe of the stand-alone runs:
    ci          the self head alone (no cross-variable branch)
    dyn         DLT-dyn as in the paper (joint training, one learning rate)
    dyn_sel     + per-variable selection: the correction only where it lowers the validation MSE
    dyn_lr      the branch at a tenth of the learning rate (--dlt_lr 3e-4)
    dyn_lr_sel  both
    dyn_2s      two-stage training: the self head on its own loss first, then the branch on its error
    dyn_2s_sel  two-stage training and per-variable selection (the plug-in recipe of the paper)
  round 2 (one recipe for every dataset; ETTm1 and ETTm2 added as data the guards were not designed on):
    dyn_2s_sig        two-stage training, correction kept only where it helps consistently over blocks of validation
                      windows (--dlt_select significant: one-sided t-test, never with fewer than three blocks)
    ci_last           the self head alone, relative to the last observation (--dlt_anchor last: the zero map is the
                      naive forecast)
    last              DLT-dyn relative to the last observation
    last_2s_sig       last observation + two-stage + significant selection
    diff_2s_sig       the branch on the first differences of the neighbours (--dlt_cross_input diff) + two-stage +
                      significant selection
    last_diff_2s_sig  all three
    dyn_2s_sig25      dyn_2s_sig at level 0.25 (the rule of at least three blocks unchanged)
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import jobs  # noqa: E402
from src.utils import paths  # noqa: E402

P0_SETTINGS = {
    "ETTh1_96": ("ETTh1", 96, "ETTh1"),
    "Weather_96": ("Weather", 96, "Weather"),
    "PEMS08_24": ("PEMS08", 24, "PEMS08_96"),
    "PEMS08_96": ("PEMS08", 96, "PEMS08_96"),
    "ECL_96": ("ECL", 96, "ECL"),
}
from src.ablation.registry import P0_CONFIGS, p0_args, p0_impl  # noqa: E402

GUARD_SETTINGS = (("ETTh1", 96), ("ETTh2", 96), ("ETTh2", 720), ("ETTm1", 96), ("ETTm2", 96), ("Exchange", 96), ("Exchange", 720),
                  ("Weather", 96), ("PEMS08", 96))
GUARD_ARMS = {
    "ci": "--dlt_variant ci",
    "dyn": "",
    "dyn_sel": "--dlt_select variable",
    "dyn_lr": "--dlt_lr 0.0003",
    "dyn_lr_sel": "--dlt_lr 0.0003 --dlt_select variable",
    "dyn_2s": "--dlt_train decoupled",
    "dyn_2s_sel": "--dlt_train decoupled --dlt_select variable",
    "dyn_2s_sig": "--dlt_train decoupled --dlt_select significant",
    "dyn_2s_sig25": "--dlt_train decoupled --dlt_select significant --dlt_select_alpha 0.25",
    "ci_last": "--dlt_variant ci --dlt_anchor last",
    "last": "--dlt_anchor last",
    "last_2s_sig": "--dlt_anchor last --dlt_train decoupled --dlt_select significant",
    "diff_2s_sig": "--dlt_cross_input diff --dlt_train decoupled --dlt_select significant",
    "last_diff_2s_sig": "--dlt_anchor last --dlt_cross_input diff --dlt_train decoupled --dlt_select significant",
}


def bench_costs():
    cost = {}
    for key in ("ETTh1", "Weather", "PEMS08_96", "ECL"):
        f = paths.OUTPUT / "ablation" / "bench_{}.jsonl".format(key)
        if f.exists():
            for line in open(f):
                if line.startswith("{"):
                    r = json.loads(line)
                    cost[(key, r["variant"], r["gate"], r["impl"])] = r["epoch_s"] * 10
    return cost


LAGROLE_SETTINGS = (("Weather", 96), ("PEMS08", 96))
LAGROLE_LAGS = (1, 2, 4, 8, 12)


def main():
    p = jobs.common_parser(__doc__)
    a = p.parse_args()
    jobs.check_names(a.datasets)
    g = jobs.JobList(a.suite, "ablation", a.device, a.list)
    S = a.seeds
    if a.suite == "p0":
        cost = bench_costs()
        for s, (ds, H, bkey) in P0_SETTINGS.items():
            if a.datasets and ds not in a.datasets:
                continue
            for (v, gate), seed in itertools.product(P0_CONFIGS, S):
                args = "{} {}".format(" ".join("--{} {}".format(k, x) for k, x in p0_args(v, gate).items()), jobs.DLT_TRAIN)
                c = cost.get((bkey, v, gate, p0_impl(v)), cost.get((bkey, v, gate, "kernel"), jobs.cost(ds)))
                g.add("{}_{}_{}".format(s, v, gate), "DLT", ds, H, seed, args, c, label=48)
    elif a.suite == "ci_long":
        for ds, H, seed in itertools.product(a.datasets or jobs.LONG_SETS, jobs.LONG_HORIZONS, S):
            g.add("{}_{}_ci".format(ds, H), "DLT", ds, H, seed, "--dlt_variant ci {}".format(jobs.DLT_TRAIN), jobs.cost(ds, 0.1))
    elif a.suite == "allnb":
        # all other variables as neighbours (dense path, arm _all) against the CCF top-8 (arm _ccf, the computation path
        # of P0) with the same device, protocol and seeds. PEMS04 / PEMS08 at horizon 12: DLT and DLT-dyn, and on PEMS08
        # iTransformer alone and + DLT-dyn (joint training, branch at 3e-3). Other datasets at the four long horizons:
        # DLT and DLT-dyn; DLT-dyn with the CCF top-8 is the run of long_small / long_large there (allnb_table.py).
        allnb = "--dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl dense --dlt_neighbours all"
        ccf = lambda v: "--dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl {} --dlt_neighbours ccf".format(p0_impl(v))
        for ds in a.datasets or ("PEMS08", "PEMS04", "Weather", "ECL"):
            pems = ds in jobs.PEMS_SETS
            for H, seed in itertools.product((12,) if pems else jobs.LONG_HORIZONS, S):
                for v in ("dltdyn", "dlt"):
                    if v == "dltdyn" and ds in ("ECL", "Traffic") and a.device == "cpu":
                        continue
                    rel = {"PEMS08": 4.0, "PEMS04": 12.0}.get(ds, 1.0) if pems else 1.0
                    rel *= (0.1 if pems else 0.6) if v == "dlt" else 1.0
                    g.add("{}_{}_{}_all".format(ds, H, v), "DLT", ds, H, seed, "--dlt_variant {} {} {}".format(v, allnb, jobs.DLT_TRAIN), jobs.cost(ds, rel))
                    if pems or v == "dlt":
                        g.add("{}_{}_{}_ccf".format(ds, H, v), "DLT", ds, H, seed, "--dlt_variant {} {} {}".format(v, ccf(v), jobs.DLT_TRAIN), jobs.cost(ds, 0.6))
            if ds == "PEMS08":
                bargs = jobs.baseline_profiles.args("iTransformer", "PEMS08", 12)[0]
                for seed in S:
                    for arm, extra, rel in (("alone", "--dlt_variant ci", 2.0), ("dlt_ccf", "--dlt_variant dltdyn {} --dlt_lr 0.003".format(ccf("dltdyn")), 3.0),
                                            ("dlt_all", "--dlt_variant dltdyn {} --dlt_lr 0.003".format(allnb), 6.0)):
                        g.add("PEMS08_12_iTransformer_" + arm, "DLTPlug", "PEMS08", 12, seed, "--plug_base iTransformer {} {}".format(extra, bargs), jobs.cost("PEMS08", rel))
    elif a.suite == "guard":
        for (ds, H), (arm, extra), seed in itertools.product(GUARD_SETTINGS, GUARD_ARMS.items(), S):
            if a.datasets and ds not in a.datasets:
                continue
            args = " ".join(x for x in (jobs.DLT_DEFAULT, jobs.DLT_TRAIN, extra) if x)
            rel = 0.1 if arm.startswith("ci") else (2.0 if "2s" in arm else 1.0)
            g.add("{}_{}_{}".format(ds, H, arm), "DLT", ds, H, seed, args, jobs.cost(ds, rel))
    elif a.suite == "lagrole":
        p0 = lambda v: " ".join("--{} {}".format(k, x) for k, x in p0_args(v, "none").items())
        arms = {"ci": ("--dlt_variant ci", 0.1), "scalardyn": (p0("scalardyn"), 0.3), "dltdyn": (p0("dltdyn"), 1.0)}
        # the neighbours of every arm come from the CCF search over lags 0..48 (--dlt_ccf_L), as in P0
        arms.update({"dltdyn_L{}".format(L): ("{} --dlt_L {} --dlt_ccf_L 48".format(jobs.DLT_DEFAULT, L), 0.6) for L in LAGROLE_LAGS})
        for (ds, H), (arm, (args, rel)), seed in itertools.product(LAGROLE_SETTINGS, arms.items(), S):
            if a.datasets and ds not in a.datasets:
                continue
            g.add("{}_{}_{}".format(ds, H, arm), "DLT", ds, H, seed, "{} {} --step_metrics".format(args, jobs.DLT_TRAIN), jobs.cost(ds, rel), label=48)
    else:
        raise SystemExit("unknown suite {}; choose from p0, ci_long, allnb, guard, lagrole".format(a.suite))
    g.write()


if __name__ == "__main__":
    main()
