"""Job lists of the sensitivity studies: sensitivity (L, R, k one at a time) and neighbours (random or all neighbours instead of the CCF top-k)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import jobs  # noqa: E402


def main():
    p = jobs.common_parser(__doc__)
    a = p.parse_args()
    dlt = "{} {}".format(jobs.DLT_DEFAULT, a.dlt).strip()
    g = jobs.JobList(a.suite, "hyperparam", a.device, a.list)
    if a.suite == "sensitivity":
        for ds, H in (("ETTh1", 96), ("Weather", 96), ("PEMS08", 96)):
            for name, args, rel in (
                ("L12", "--dlt_L 12", 0.6),
                ("L24", "--dlt_L 24", 0.7),
                ("L96", "--dlt_L 96", 1.8),
                ("R2", "--dlt_R 2", 0.9),
                ("R8", "--dlt_R 8", 1.3),
                ("k4", "--dlt_k 4", 0.6),
                ("k16", "--dlt_k 16", 1.9),
            ):
                for seed in a.seeds:
                    g.add("{}_{}_{}".format(ds, H, name), "DLT", ds, H, seed, "{} {} {}".format(dlt, jobs.DLT_TRAIN, args), jobs.cost(ds, rel))
    elif a.suite == "neighbours":
        for ds, H in (("Weather", 96), ("PEMS08", 96)):
            for v, nb, impl, rel in (("dltdyn", "random", "basis", 1.0), ("dltdyn", "all", "dense", 4.0), ("dlt", "random", "kernel", 0.9), ("dlt", "all", "dense", 2.0)):
                for seed in a.seeds:
                    g.add(
                        "{}_{}_{}_{}".format(ds, H, v, nb),
                        "DLT",
                        ds,
                        H,
                        seed,
                        "--dlt_variant {} --dlt_k 8 --dlt_L 48 --dlt_R 4 --dlt_impl {} --dlt_neighbours {} {}".format(v, impl, nb, jobs.DLT_TRAIN),
                        jobs.cost(ds, rel),
                    )
    else:
        raise SystemExit("unknown suite {}; choose from sensitivity, neighbours (the validation search: scripts/hyperparam/hpsearch.py)".format(a.suite))
    g.write()


if __name__ == "__main__":
    main()
