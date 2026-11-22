from typing import Any

from src.utils import paths

LONG_HORIZONS = [96, 192, 336, 720]
PEMS_HORIZONS = [12, 24, 48, 96]
PEMS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")
NO_MARKS = ("Solar",) + PEMS

DATASET_META: dict[str, dict[str, Any]] = {
    "ETTh1": {"data": "ETTh1", "data_path": "ETTh1.csv", "enc_in": 7, "freq": "h", "train_windows": 8545},
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "freq": "h", "train_windows": 8545},
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "enc_in": 7, "freq": "t", "train_windows": 34465},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "freq": "t", "train_windows": 34465},
    "Weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "freq": "t", "train_windows": 36792},
    "Exchange": {"data": "custom", "data_path": "exchange_rate.csv", "enc_in": 8, "freq": "d", "train_windows": 5120},
    "ECL": {"data": "custom", "data_path": "electricity.csv", "enc_in": 321, "freq": "h", "train_windows": 18317},
    "Traffic": {"data": "custom", "data_path": "traffic.csv", "enc_in": 862, "freq": "h", "train_windows": 12185},
    "Solar": {"data": "Solar", "data_path": "solar_AL.txt", "enc_in": 137, "freq": "t", "train_windows": 36601},
    "PEMS03": {"data": "PEMS", "data_path": "PEMS03.npz", "enc_in": 358, "freq": "t", "train_windows": 15617},
    "PEMS04": {"data": "PEMS", "data_path": "PEMS04.npz", "enc_in": 307, "freq": "t", "train_windows": 10088},
    "PEMS07": {"data": "PEMS", "data_path": "PEMS07.npz", "enc_in": 883, "freq": "t", "train_windows": 16827},
    "PEMS08": {"data": "PEMS", "data_path": "PEMS08.npz", "enc_in": 170, "freq": "t", "train_windows": 10606},
}

DATASET_ORDER = list(DATASET_META)

_ALIASES = {
    "electricity": "ECL",
    "weather": "Weather",
    "traffic": "Traffic",
    "exchange": "Exchange",
    "exchange_rate": "Exchange",
    "solar": "Solar",
    "solar-energy": "Solar",
    "solar_energy": "Solar",
    "solar_al": "Solar",
}

CONSTANTS: dict[str, Any] = {
    "dlt_variant": "dltdyn",
    "dlt_k": 8,
    "dlt_L": 48,
    "dlt_R": 4,
    "dlt_impl": "basis",
    "learning_rate": 0.003,
    "batch_size": 32,
    "train_epochs": 10,
    "patience": 3,
}

EFFICIENCY_SETTING = {"ETTm1": 96, "ECL": 96, "Traffic": 96, "PEMS07": 12}
EFFICIENCY_EPOCHS = {"ETTm1": 10, "ECL": 5, "Traffic": 10, "PEMS07": 10}
EFFICIENCY_RECIPES: dict[str, dict[str, Any]] = {
    "ETTm1": {},
    "ECL": {"dlt_R": 8, "dlt_k": 16},
    "Traffic": {"dlt_k": 16},
    "PEMS07": {"dlt_k": 16},
}

CLI_ORDER = [
    "dlt_variant",
    "dlt_k",
    "dlt_L",
    "dlt_R",
    "dlt_impl",
    "dlt_neighbours",
    "dlt_gate",
    "learning_rate",
    "batch_size",
    "train_epochs",
    "patience",
    "freq",
    "drop_last",
    "test_batch_size",
    "no_time_marks",
    "pems_split",
]

META_ORDER = ["root_path", "data_path", "data", "model_id", "features", "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out", "freq"]


def canonical_dataset(name):
    name = str(name)
    if name in DATASET_META:
        return name
    low = name.lower()
    for ds in DATASET_META:
        if ds.lower() == low:
            return ds
    for ds, meta in DATASET_META.items():
        if meta["data_path"].lower() == low or meta["data_path"].rsplit(".", 1)[0].lower() == low:
            return ds
    return _ALIASES.get(low, name)


def list_datasets():
    return list(DATASET_META)


def get_pred_lens(dataset):
    return list(PEMS_HORIZONS if canonical_dataset(dataset) in PEMS else LONG_HORIZONS)


def protocol(dataset):
    """Evaluation protocol of the official iTransformer / S-Mamba loaders, shared by every baseline run."""
    dataset = canonical_dataset(dataset)
    out = {"freq": "h", "drop_last": True, "test_batch_size": 1}
    if dataset in NO_MARKS:
        out["no_time_marks"] = True
    if dataset in PEMS:
        out["pems_split"] = "itransformer"
    return out


def resolve_meta(dataset, pred_len, label_len=None, model_id=None):
    dataset = canonical_dataset(dataset)
    if dataset not in DATASET_META:
        raise KeyError("Unknown dataset: {}. Supported: {}".format(dataset, list_datasets()))
    m = DATASET_META[dataset]
    if label_len is None:
        label_len = 0 if dataset in PEMS else 48
    return {
        "root_path": paths.rel(paths.root_of(m["data_path"])) + "/",
        "data_path": m["data_path"],
        "data": m["data"],
        "model_id": model_id or "{}_96_{}".format(dataset, pred_len),
        "features": "M",
        "seq_len": 96,
        "label_len": label_len,
        "pred_len": pred_len,
        "enc_in": m["enc_in"],
        "dec_in": m["enc_in"],
        "c_out": m["enc_in"],
        "freq": m["freq"],
    }


def resolve(dataset, pred_len, profile="default"):
    """DLT-dyn settings: "default" = the paper's stand-alone recipe (Solar and PEMS with the official-loader protocol); "efficiency" = S-Mamba Fig. 5."""
    dataset = canonical_dataset(dataset)
    if dataset not in DATASET_META:
        raise KeyError("Unknown dataset: {}. Supported: {}".format(dataset, list_datasets()))
    cfg = dict(CONSTANTS)
    if profile == "default":
        if dataset in NO_MARKS:
            cfg.update(protocol(dataset))
    elif profile == "efficiency":
        if EFFICIENCY_SETTING.get(dataset) != pred_len:
            raise KeyError("efficiency profile: {} only at horizon {}".format(dataset, EFFICIENCY_SETTING.get(dataset)))
        cfg.update({"batch_size": 16, "train_epochs": EFFICIENCY_EPOCHS[dataset], "patience": 3})
        cfg.update(protocol(dataset))
        cfg.update(EFFICIENCY_RECIPES[dataset])
    else:
        raise KeyError("Unknown profile: {} (default, efficiency)".format(profile))
    return cfg


def _fmt(value):
    if isinstance(value, float):
        return "{:g}".format(value)
    return str(value)


def emit_cli_args(cfg, meta=None):
    args = []
    if meta:
        cfg = {**meta, **cfg}
        order = META_ORDER + [k for k in CLI_ORDER if k not in META_ORDER]
    else:
        order = CLI_ORDER
    order = order + sorted(k for k in cfg if k not in order)
    for key in order:
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, bool):
            if value:
                args.append("--{}".format(key))
            continue
        args += ["--{}".format(key), _fmt(value)]
    return args


def to_cli_args(dataset, pred_len, *, with_meta=False, profile="default"):
    cfg = resolve(dataset, pred_len, profile)
    meta = resolve_meta(dataset, pred_len) if with_meta else None
    return emit_cli_args(cfg, meta)


def to_cli_string(dataset, pred_len, *, with_meta=False, profile="default"):
    return " ".join(to_cli_args(dataset, pred_len, with_meta=with_meta, profile=profile))
