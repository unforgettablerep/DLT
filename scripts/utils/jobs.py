import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.configs import baseline_profiles  # noqa: E402
from src.configs.dlt_profiles import CONSTANTS, DATASET_META, LONG_HORIZONS, PEMS, PEMS_HORIZONS, emit_cli_args, protocol, resolve_meta  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.result_io import latest, read_results  # noqa: E402

BASELINES = baseline_profiles.BASELINES
LONG_SETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "ECL", "Traffic")
PEMS_SETS = PEMS
ALL_SETS = LONG_SETS + ("Solar",) + PEMS_SETS
SEEDS = (2021, 2022, 2023)
DLT_DEFAULT = " ".join(emit_cli_args({k: CONSTANTS[k] for k in ("dlt_variant", "dlt_k", "dlt_L", "dlt_R", "dlt_impl")}))
DLT_TRAIN = " ".join(emit_cli_args({k: CONSTANTS[k] for k in ("learning_rate", "batch_size", "train_epochs", "patience")}))
REL_COST = {"S_Mamba": 1.5, "iTransformer": 1.5, "PatchTST": 3.0, "RLinear": 0.2, "DLinear": 0.3, "TiDE": 2.0}
LR_GRID = (1e-4, 5e-4, 1e-3)


def horizons(ds):
    return PEMS_HORIZONS if ds in PEMS_SETS else LONG_HORIZONS


def protocol_args(ds):
    return " ".join(emit_cli_args(protocol(ds)))


def baseline_args(model, ds, H):
    """Protocol, official settings and runtime options of a baseline (the arguments of the benchmark and plug-in runs)."""
    return " ".join(p for p in (protocol_args(ds), baseline_profiles.args(model, ds, H)[0], baseline_profiles.runtime_args(model, ds)) if p)


def cost(ds, rel=1.0):
    """Very rough single-thread CPU seconds of a 10-epoch DLT-dyn run (only orders the job lists)."""
    m = DATASET_META[ds]
    return rel * 10 * m["train_windows"] / 32 * (0.0007 * min(m["enc_in"], 400) + 0.002) * 2.2


def recipe_tag(extra):
    """Short name of extra --dlt arguments, so that a changed recipe gets its own job names."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", re.sub(r"--(dlt_)?", "", extra)).strip("-")


def lr_suffix(lr):
    return "@lr{:g}".format(lr)


def setting_lr(args):
    m = re.search(r"--learning_rate (\S+)", args)
    return float(m.group(1)) if m else 1e-4


def result_file(category, suite):
    return paths.OUTPUT / category / "{}.tsv".format(suite)


class JobList:
    """Shell commands of scripts/run.py, one per line, longest first within each seed: temp/jobs/jobs_<list>.txt."""

    def __init__(self, suite, category, device="cpu", list_name=None):
        self.suite, self.category, self.device = suite, category, device
        self.list = list_name or suite
        self.jobs, self.files = [], set()
        self.result = result_file(category, suite)
        self.logs = paths.LOG / category / suite
        os.makedirs(self.logs, exist_ok=True)
        os.makedirs(paths.JOBS, exist_ok=True)
        os.makedirs(self.result.parent, exist_ok=True)

    def add(self, name, model, ds, H, seed, args, cost_s, label=None):
        if label is None:
            label = 0 if ds in PEMS_SETS else 48
        meta = resolve_meta(ds, H, label_len=label, model_id="{}_{}_{}".format(self.suite, ds, H))
        if "--drop_last" not in args:                 # every run: the protocol of the official loaders
            args = "{} {}".format(protocol_args(ds), args).strip()
        meta.pop("freq")                                # in the protocol
        dev = "--no_use_gpu --num_workers 0" if self.device == "cpu" else "--num_workers 4"
        cmd = "python -u scripts/run.py --is_training 1 --model {} {} --des {} --seed {} {} {} --no_epoch_test --save_tmp 0 --result_file {} > {} 2>&1".format(
            model,
            " ".join(emit_cli_args({}, meta)),
            name,
            seed,
            args,
            dev,
            paths.rel(self.result),
            paths.rel(self.logs / "{}_s{}.log".format(name, seed)),
        )
        self.jobs.append((cost_s, "{}_s{}".format(name, seed), " ".join(cmd.split())))
        self.files.add(DATASET_META[ds]["data_path"])

    def write(self):
        by_seed = {}
        for j in self.jobs:
            by_seed.setdefault(j[1].rsplit("_s", 1)[1], []).append(j)
        ordered = [j for s in sorted(by_seed) for j in sorted(by_seed[s], key=lambda j: -j[0])]
        path = paths.JOBS / "jobs_{}.txt".format(self.list)
        path.write_text("\n".join(j[2] for j in ordered) + "\n")
        hours = sum(j[0] for j in ordered) / 3600
        print("{}: {} jobs -> {}; results -> {}; rough CPU estimate {:.1f} core-hours".format(self.list, len(ordered), paths.rel(path), paths.rel(self.result), hours))
        for f in sorted(self.files):
            if not paths.exists(f):
                print("  note: {} is missing: bash scripts/download_data.sh fetches it".format(paths.path_of(f)))
        return path


def records(category, suite):
    return latest(read_results(result_file(category, suite)))


def check_names(datasets=None, models=None):
    unknown = sorted(set(datasets or ()) - set(DATASET_META))
    if unknown:
        raise SystemExit("unknown dataset(s) {}; choose from {}".format(unknown, list(DATASET_META)))
    from src.configs.model_parser_dict import canonical_model, model_parser_dict

    unknown = sorted(m for m in models or () if canonical_model(m) not in model_parser_dict)
    if unknown:
        raise SystemExit("unknown model(s) {}".format(unknown))


def common_parser(description):
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("suite")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--dlt", default="", help="extra --dlt_* arguments of the proposed model")
    p.add_argument("--list", help="name of the job list (temp/jobs/jobs_<list>.txt; default: the suite)")
    p.add_argument("--models", nargs="+", help="only these models")
    p.add_argument("--datasets", nargs="+", help="only these datasets")
    return p
