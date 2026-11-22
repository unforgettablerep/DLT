"""Job list of the tuning rounds in the efficiency setting of the S-Mamba paper: python scripts/efficiency/make_jobs.py eff_tune --tag r3-R8 --dlt "--dlt_R 8"."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import jobs  # noqa: E402
from src.configs.dlt_profiles import EFFICIENCY_EPOCHS, EFFICIENCY_SETTING  # noqa: E402


def main():
    p = jobs.common_parser(__doc__)
    p.add_argument("--tag", help="name of the recipe given by --dlt (job names <ds>_<H>_dltdyn_<tag>)")
    a = p.parse_args()
    if a.suite != "eff_tune":
        raise SystemExit("unknown suite {}; the efficiency experiment has the suite eff_tune".format(a.suite))
    jobs.check_names(a.datasets)
    tag = a.tag or "default"
    g = jobs.JobList(a.suite, "efficiency", a.device, a.list)
    for ds, H in EFFICIENCY_SETTING.items():
        if a.datasets and ds not in a.datasets:
            continue
        for seed in a.seeds:
            train = "--batch_size 16 --train_epochs {} --patience 3 --learning_rate 0.003".format(EFFICIENCY_EPOCHS[ds])
            g.add("{}_{}_dltdyn_{}".format(ds, H, tag), "DLT", ds, H, seed, "{} {} {} {}".format(jobs.DLT_DEFAULT, train, jobs.protocol_args(ds), a.dlt).strip(), jobs.cost(ds, 2.0))
    g.write()


if __name__ == "__main__":
    main()
