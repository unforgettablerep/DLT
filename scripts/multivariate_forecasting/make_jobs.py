"""Job lists of the stand-alone DLT-dyn runs (Tables 3-5 of the paper): python scripts/multivariate_forecasting/make_jobs.py <suite>."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import jobs  # noqa: E402

SUITES = {
    "long_small": "DLT-dyn on ETTh1/2, ETTm1/2, Exchange, Weather, horizons 96-720",
    "long_large": "DLT-dyn on Electricity and Traffic, horizons 96-720 (GPU recommended)",
    "long_extra": "DLT-dyn on Solar and PEMS03/04/07/08",
}


def main():
    p = jobs.common_parser(__doc__ + "\nSuites: " + "; ".join("{}: {}".format(k, v) for k, v in SUITES.items()))
    a = p.parse_args()
    if a.suite not in SUITES:
        raise SystemExit("unknown suite {}; choose from {}".format(a.suite, list(SUITES)))
    jobs.check_names(a.datasets)
    g = jobs.JobList(a.suite, "long_term_forecast", a.device, a.list)
    dlt = "{} {}".format(jobs.DLT_DEFAULT, a.dlt).strip()
    S = a.seeds
    if a.suite == "long_extra":
        for ds in a.datasets or ("PEMS08", "PEMS04", "Solar", "PEMS03", "PEMS07"):
            for H, seed in itertools.product(jobs.horizons(ds), S):
                g.add("{}_{}_dltdyn".format(ds, H), "DLT", ds, H, seed, "{} {}".format(dlt, jobs.DLT_TRAIN), jobs.cost(ds))
    else:
        sets = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather") if a.suite == "long_small" else ("ECL", "Traffic")
        for ds, H, seed in itertools.product(a.datasets or sets, jobs.LONG_HORIZONS, S):
            g.add("{}_{}_dltdyn".format(ds, H), "DLT", ds, H, seed, "{} {}".format(dlt, jobs.DLT_TRAIN), jobs.cost(ds))
    g.write()


if __name__ == "__main__":
    main()
