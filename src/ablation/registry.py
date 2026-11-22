"""Ablations of DLT: the cross-variable kernels and gates of the controlled comparison (P0) and the other ablation groups.

Every group is a set of DLT arguments that replaces part of the default recipe (DLT-dyn, k 8, L 48, R 4, basis path;
src/configs/dlt_profiles.py). scripts/ablation/make_jobs.py writes the job lists from it and
scripts/ablation/aggregate.py labels the tables with it.
"""
from src.layers.DLT_layers import GATES, IMPLS, NEIGHBOUR_MODES, VARIANTS

NO_ABLATION = "none"

# P0: (kernel variant, gate) in table order; every configuration shares neighbours, heads and training recipe
P0_CONFIGS = [(v, "none") for v in ("ci", "scalar", "shift", "tap3", "dlt", "full", "scalardyn", "dltdyn", "topk", "mill", "milldyn")] + [
    ("dltdyn", "moe"),
    ("dltdyn", "static"),
]
P0_TABLE_ORDER = [
    ("ci", "none"), ("scalar", "none"), ("shift", "none"), ("tap3", "none"), ("full", "none"),
    ("topk", "none"), ("mill", "none"), ("milldyn", "none"), ("scalardyn", "none"),
    ("dlt", "none"), ("dltdyn", "none"), ("dltdyn", "moe"), ("dltdyn", "static"),
]
P0_LABELS = {
    ("ci", "none"): "CI (no cross branch)",
    ("scalar", "none"): "scalar (lag 0)",
    ("shift", "none"): "shift (CCF peak lag, LIFT-style)",
    ("tap3", "none"): "tap3 (CrossLinear-style)",
    ("full", "none"): "full (free, lags 0..48)",
    ("topk", "none"): "top-5 lags, free weights",
    ("mill", "none"): "MillGNN-static (top-5 + decay)",
    ("milldyn", "none"): "MillGNN-dyn (top-5 + decay + attention)",
    ("scalardyn", "none"): "scalar-dyn",
    ("dlt", "none"): "**DLT**",
    ("dltdyn", "none"): "**DLT-dyn**",
    ("dltdyn", "moe"): "DLT-dyn + MoVE softmax gate",
    ("dltdyn", "static"): "DLT-dyn + static gate (init 0.047)",
}
P0_GROUP_BREAK = {("full", "none"), ("milldyn", "none"), ("dltdyn", "none")}   # table separators after these


def p0_impl(variant):
    """Computation path of a P0 variant: DLT-dyn on the basis path (as trained in the paper), the others as a kernel."""
    return "basis" if variant == "dltdyn" else "kernel"


def p0_args(variant, gate):
    return {"dlt_variant": variant, "dlt_gate": gate, "dlt_impl": p0_impl(variant), "dlt_k": 8, "dlt_L": 48, "dlt_R": 4, "dlt_topk": 5}


# further ablation groups (name -> DLT arguments over the default recipe)
ABLATION_GROUPS = {
    **{"{}_{}".format(v, g) if g != "none" else v: p0_args(v, g) for v, g in P0_CONFIGS},
    "random_neighbours": {"dlt_neighbours": "random"},
    "all_neighbours": {"dlt_neighbours": "all", "dlt_impl": "dense"},
    "no_cross_branch": {"dlt_variant": "ci"},
}
ABLATION_CHOICES = [NO_ABLATION] + list(ABLATION_GROUPS)

assert set(P0_TABLE_ORDER) == set(P0_CONFIGS) == set(P0_LABELS)
assert all(v in VARIANTS and g in GATES for v, g in P0_CONFIGS)
assert all(a.get("dlt_neighbours", "ccf") in NEIGHBOUR_MODES and a.get("dlt_impl", "basis") in IMPLS for a in ABLATION_GROUPS.values())
