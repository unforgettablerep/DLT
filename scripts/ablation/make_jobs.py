"""Job lists of the ablations: p0 (controlled comparison of cross-variable kernels), ci_long (DLT without the branch), allnb (all variables as neighbours),
guard (stand-alone DLT-dyn with the guards of the plug-in against harm on weakly coupled data).

guard: seven arms on the weakly coupled ETTh1, ETTh2 and Exchange and the coupled Weather and PEMS08 (GUARD_SETTINGS),
three seeds, the recipe of the stand-alone runs (long_small; PEMS08 with the protocol of long_extra):
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
        allnb = "--dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl dense --dlt_neighbours all"
        for ds, seed in itertools.product(("PEMS08", "PEMS04"), S):
            for v, rel in (("dltdyn", 4.0 if ds == "PEMS08" else 12.0), ("dlt", 0.4 if ds == "PEMS08" else 1.2)):
                g.add("{}_12_{}_all".format(ds, v), "DLT", ds, 12, seed, "--dlt_variant {} {} {} {}".format(v, allnb, jobs.DLT_TRAIN, jobs.PEMS_PROTOCOL), jobs.cost(ds, rel))
        for ds, H, seed in itertools.product(a.datasets or ("Weather", "ECL"), jobs.LONG_HORIZONS, S):
            for v in ("dltdyn", "dlt"):
                if v == "dltdyn" and ds in ("ECL", "Traffic") and a.device == "cpu":
                    continue
                g.add("{}_{}_{}_all".format(ds, H, v), "DLT", ds, H, seed, "--dlt_variant {} {} {}".format(v, allnb, jobs.DLT_TRAIN), jobs.cost(ds, 0.6 if v == "dlt" else 1.0))
        for seed in S:
            bargs = jobs.baseline_profiles.args("iTransformer", "PEMS08", 12)[0]
            g.add(
                "PEMS08_12_iTransformer_dlt_all",
                "DLTPlug",
                "PEMS08",
                12,
                seed,
                "--plug_base iTransformer --dlt_variant dltdyn {} --dlt_lr 0.003 {} {}".format(allnb, bargs, jobs.PEMS_PROTOCOL),
                jobs.cost("PEMS08", 6.0),
            )
    elif a.suite == "guard":
        for (ds, H), (arm, extra), seed in itertools.product(GUARD_SETTINGS, GUARD_ARMS.items(), S):
            if a.datasets and ds not in a.datasets:
                continue
            proto = jobs.protocol_args(ds) if ds in jobs.PEMS_SETS else ""
            args = " ".join(x for x in (jobs.DLT_DEFAULT, jobs.DLT_TRAIN, proto, extra) if x)
            rel = 0.1 if arm.startswith("ci") else (2.0 if "2s" in arm else 1.0)
            g.add("{}_{}_{}".format(ds, H, arm), "DLT", ds, H, seed, args, jobs.cost(ds, rel))
    else:
        raise SystemExit("unknown suite {}; choose from p0, ci_long, allnb, guard".format(a.suite))
    g.write()


if __name__ == "__main__":
    main()
