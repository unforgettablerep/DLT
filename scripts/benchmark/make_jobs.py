"""Job list of the baseline check (the six baselines alone, official settings and protocol): python scripts/benchmark/make_jobs.py baselines."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import jobs  # noqa: E402


def main():
    p = jobs.common_parser(__doc__)
    p.add_argument("--backbone_lr", type=float, nargs="+", help="learning-rate check: the same settings at these learning rates (<job>@lr<x>)")
    a = p.parse_args()
    if a.suite != "baselines":
        raise SystemExit("unknown suite {}; the benchmark has the suite baselines".format(a.suite))
    jobs.check_names(a.datasets, a.models)
    from src.configs.model_parser_dict import canonical_model

    models = [canonical_model(m) for m in a.models] if a.models else jobs.BASELINES
    g = jobs.JobList(a.suite, "benchmark", a.device, a.list)
    for ds, base, seed in itertools.product(a.datasets or jobs.ALL_SETS, models, a.seeds):
        for H in jobs.horizons(ds):
            full = jobs.baseline_args(base, ds, H)
            if a.backbone_lr:
                for lr in a.backbone_lr:
                    if abs(lr - jobs.setting_lr(full)) > 1e-12:
                        g.add("{}_{}_{}{}".format(ds, H, base, jobs.lr_suffix(lr)), base, ds, H, seed, "{} --learning_rate {:g}".format(full, lr), jobs.cost(ds, jobs.REL_COST[base]), label=48)
                continue
            g.add("{}_{}_{}".format(ds, H, base), base, ds, H, seed, full, jobs.cost(ds, jobs.REL_COST[base]), label=48)
    g.write()


if __name__ == "__main__":
    main()
