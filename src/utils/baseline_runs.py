"""The baselines alone from the baseline check, as the "alone" arm of the plug-in studies.

The plug-in lists of the GPU study run only the + DLT-dyn arms (scripts/plugin/make_jobs.py --skip_alone): the
backbone alone is the run of the baseline check (suite baselines of scripts/benchmark/make_jobs.py, stores
output/benchmark/baselines.tsv and baselines_gpu.tsv), the same model with the same settings, protocol and seed.
alone_cells() returns these runs in the layout of the plug-in tables (scripts/plugin/plugin_table.py and
fusion_table.py with --alone baselines; plug-in make_jobs.py --select_lr): (dataset, H, backbone, arm) -> records,
arm "alone" for the official learning rate and "alone@lr<x>" for the runs of the learning-rate check. Only records of
the check's protocol (four time features, dropped last batches: freq h and drop_last) and of the given device count;
for a run held twice (same job name and seed), the later record wins.
"""
import collections

from src.utils.paths import OUTPUT
from src.utils.result_io import device_class, read_results

STORES = (OUTPUT / "benchmark" / "baselines.tsv", OUTPUT / "benchmark" / "baselines_gpu.tsv")


def check_protocol(row):
    return row.get("freq") == "h" and bool(row.get("drop_last"))


def alone_cells(device="gpu", seeds=None, stores=STORES):
    """(dataset, H, backbone, "alone" | "alone@lr<x>") -> records; device "gpu", "cpu" or "all"."""
    runs = {}
    for f in stores:
        for r in read_results(f):
            if not check_protocol(r) or (device != "all" and device_class(r) != device):
                continue
            if seeds and r.get("seed") not in seeds:
                continue
            runs[(r.get("des"), r.get("seed"))] = r
    cells = collections.defaultdict(list)
    for r in runs.values():
        stem = "{}_{}_{}".format(r.get("dataset"), r.get("pred_len"), r.get("model"))
        des = r.get("des") or ""
        sfx = des[len(stem):] if des.startswith(stem) else None
        if sfx is None or (sfx and not sfx.startswith("@lr")):
            continue
        cells[(r["dataset"], r["pred_len"], r["model"], "alone" + sfx)].append(r)
    return cells
