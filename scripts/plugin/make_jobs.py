"""Job lists of the plug-in studies: plugin_all (every baseline alone and + DLT-dyn) and plugin_fusion (how the branch enters the backbone)."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import jobs  # noqa: E402

FUSION_ARMS = {
    "dlt": "",
    "dlt_out-dec": "--dlt_train decoupled --dlt_select variable",
    "dlt_refine": "--dlt_fusion refine",
    "dlt_refine-dec": "--dlt_fusion refine --dlt_train decoupled --dlt_select variable",
    "dlt_input": "--dlt_fusion input",
}
FUSION_SETS = (("ETTh2", 96), ("Exchange", 96), ("Weather", 96), ("PEMS08", 12))
FUSION_MODELS = ("RLinear", "DLinear", "iTransformer")


def selected_lrs(rows):
    """(dataset, H, backbone) -> learning-rate suffix ("" = the setting's own) with the lowest mean best validation loss of the backbone alone."""
    vals = {}
    for r in rows:
        if r.get("model") != "DLTPlug" or r.get("best_vali") is None:
            continue
        ds, H, base = r["dataset"], r["pred_len"], r["plug_base"]
        stem = "{}_{}_{}_alone".format(ds, H, base)
        if r["des"] == stem or r["des"].startswith(stem + "@lr"):
            vals.setdefault((ds, H, base), {}).setdefault(r["des"][len(stem):], []).append(r["best_vali"])
    return {key: min(v, key=lambda sfx: sum(v[sfx]) / len(v[sfx])) for key, v in vals.items() if len(v) > 1}


def main():
    p = jobs.common_parser(__doc__)
    p.add_argument("--backbone_lr", type=float, nargs="+", help="plugin_all: learning-rate check, the backbone alone at these learning rates")
    p.add_argument("--select_lr", action="store_true", help="plugin_all: both arms at the learning rate with the lowest validation loss of the backbone alone")
    p.add_argument("--horizons", default="first", choices=["first", "all"], help="plugin_fusion: the first or all horizons")
    a = p.parse_args()
    jobs.check_names(a.datasets, a.models)
    from src.configs.model_parser_dict import canonical_model

    models = [canonical_model(m) for m in a.models] if a.models else None
    dlt = "{} {}".format(jobs.DLT_DEFAULT, a.dlt).strip()
    S = a.seeds
    g = jobs.JobList(a.suite, "plugin", a.device, a.list)
    if a.suite == "plugin_all":
        arm = "dlt" + ("_{}".format(jobs.recipe_tag(a.dlt)) if a.dlt else "")
        chosen = selected_lrs(jobs.records("plugin", "plugin_all")) if a.select_lr else {}
        for ds, base, seed in itertools.product(a.datasets or jobs.ALL_SETS, models or jobs.BASELINES, S):
            for H in jobs.horizons(ds):
                bargs = jobs.baseline_args(base, ds, H)
                rel = jobs.REL_COST.get(base, 2.0)
                if a.backbone_lr:
                    for lr in a.backbone_lr:
                        if abs(lr - jobs.setting_lr(bargs)) > 1e-12:
                            g.add("{}_{}_{}_alone{}".format(ds, H, base, jobs.lr_suffix(lr)), "DLTPlug", ds, H, seed, "--plug_base {} --dlt_variant ci {} --learning_rate {:g}".format(base, bargs, lr), jobs.cost(ds, rel), label=48)
                    continue
                sfx = ""
                if a.select_lr:
                    sfx = chosen.get((ds, H, base), "")
                    if not sfx:
                        continue
                    bargs = "{} --learning_rate {}".format(bargs, sfx[3:])
                g.add("{}_{}_{}_alone{}".format(ds, H, base, sfx), "DLTPlug", ds, H, seed, "--plug_base {} --dlt_variant ci {}".format(base, bargs), jobs.cost(ds, rel), label=48)
                g.add("{}_{}_{}_{}{}".format(ds, H, base, arm, sfx), "DLTPlug", ds, H, seed, "--plug_base {} --dlt_lr 0.003 {} {}".format(base, dlt, bargs), jobs.cost(ds, rel + 1), label=48)
    elif a.suite == "plugin_fusion":
        sets = [(ds, H) for ds, H0 in FUSION_SETS if not a.datasets or ds in a.datasets for H in (jobs.horizons(ds) if a.horizons == "all" else (H0,))]
        if a.datasets:
            sets += [(ds, H) for ds in a.datasets if ds not in dict(FUSION_SETS) for H in (jobs.horizons(ds) if a.horizons == "all" else ((12,) if ds in jobs.PEMS_SETS else (96,)))]
        for (ds, H), base, seed in itertools.product(sets, models or FUSION_MODELS, S):
            bargs = jobs.baseline_args(base, ds, H)
            rel = jobs.REL_COST.get(base, 2.0)
            g.add("{}_{}_{}_alone".format(ds, H, base), "DLTPlug", ds, H, seed, "--plug_base {} --dlt_variant ci {}".format(base, bargs), jobs.cost(ds, rel), label=48)
            for arm, extra in FUSION_ARMS.items():
                g.add("{}_{}_{}_{}".format(ds, H, base, arm), "DLTPlug", ds, H, seed, "--plug_base {} --dlt_lr 0.003 {} {} {}".format(base, dlt, extra, bargs), jobs.cost(ds, rel + 1), label=48)
    else:
        raise SystemExit("unknown suite {}; choose from plugin_all, plugin_fusion".format(a.suite))
    g.write()


if __name__ == "__main__":
    main()
