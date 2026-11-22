"""Job lists of the ablations: p0 (controlled comparison of cross-variable kernels), ci_long (DLT without the branch), allnb (all variables as neighbours)."""
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
P0_CONFIGS = [(v, "none") for v in ("ci", "scalar", "shift", "tap3", "dlt", "full", "scalardyn", "dltdyn", "topk", "mill", "milldyn")] + [("dltdyn", "moe"), ("dltdyn", "static")]


def p0_impl(variant):
    return "basis" if variant == "dltdyn" else "kernel"


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
                args = "--dlt_variant {} --dlt_gate {} --dlt_impl {} --dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_topk 5 {}".format(v, gate, p0_impl(v), jobs.DLT_TRAIN)
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
    else:
        raise SystemExit("unknown suite {}; choose from p0, ci_long, allnb".format(a.suite))
    g.write()


if __name__ == "__main__":
    main()
