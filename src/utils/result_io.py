import argparse
import fcntl
import json
import math
import os
import re
import time
from contextlib import contextmanager

from src.configs.dlt_profiles import DATASET_ORDER, canonical_dataset

METADATA_COLUMNS = ["dataset", "model", "seq_len", "label_len", "pred_len", "mse", "mae", "status", "checkpoint", "timestamp", "des", "seed"]

CONFIG_COLUMNS = [
    "d_model",
    "d_ff",
    "e_layers",
    "d_layers",
    "n_heads",
    "factor",
    "dropout",
    "use_norm",
    "learning_rate",
    "batch_size",
    "train_epochs",
    "patience",
    "lradj",
    "loss",
    "inverse",
    "vali_inverse",
    "time_marks",
    "drop_last",
    "test_batch_size",
    "pems_split",
    "freq",
    "dlt_variant",
    "dlt_k",
    "dlt_L",
    "dlt_R",
    "dlt_impl",
    "dlt_neighbours",
    "dlt_gate",
    "dlt_gate_init",
    "dlt_topk",
    "dlt_dyn_dim",
    "dlt_hp_trial",
    "plug_base",
    "dlt_lr",
    "dlt_fusion",
    "dlt_train",
    "dlt_select",
    "dlt_select_alpha",
    "dlt_anchor",
    "dlt_cross_input",
    "d_state",
    "mamba_impl",
]

RUN_COLUMNS = [
    "rmse",
    "mape",
    "mape_clip5",
    "mape_masked",
    "mse_backbone",
    "mae_backbone",
    "mse_full",
    "mae_full",
    "dlt_kept",
    "best_vali",
    "epochs",
    "best_epoch",
    "train_secs",
    "epoch_secs",
    "stage1_epochs",
    "branch_epochs",
    "branch_best_epoch",
    "branch_start_vali",
    "branch_best_vali",
    "n_params",
    "infer_ms",
    "test_nonfinite",
    "device",
    "torch",
]

RESULT_COLUMNS = METADATA_COLUMNS + CONFIG_COLUMNS + RUN_COLUMNS

STRING_COLUMNS = {
    "dataset",
    "model",
    "status",
    "timestamp",
    "des",
    "lradj",
    "loss",
    "pems_split",
    "freq",
    "dlt_variant",
    "dlt_impl",
    "dlt_neighbours",
    "dlt_gate",
    "plug_base",
    "dlt_fusion",
    "dlt_train",
    "dlt_select",
    "mamba_impl",
    "device",
    "torch",
}
BOOL_COLUMNS = {"inverse", "vali_inverse", "time_marks", "drop_last"}
JSON_COLUMNS = {"epoch_secs"}

DLT_MODELS = ("DLT", "DLTPlug")
DLT_ONLY = {c for c in CONFIG_COLUMNS if c.startswith("dlt_")}
PLUG_ONLY = {"plug_base", "dlt_fusion"}
BRANCH_RECIPE = {"dlt_lr": 0.0, "dlt_train": "joint", "dlt_select": "none"}   # DLTPlug always; DLT when not the default
STANDALONE_RECIPE = {"dlt_anchor": "mean", "dlt_cross_input": "level"}       # DLT only, when not the default
BACKBONE_ONLY = {"d_model", "d_ff", "e_layers", "d_layers", "n_heads", "factor", "use_norm", "dropout"}

LEGACY_KEYS = {
    "lr": "learning_rate",
    "variant": "dlt_variant",
    "gate": "dlt_gate",
    "gate_init": "dlt_gate_init",
    "k": "dlt_k",
    "L": "dlt_L",
    "R": "dlt_R",
    "topk": "dlt_topk",
    "impl": "dlt_impl",
    "neighbours": "dlt_neighbours",
    "dyn_dim": "dlt_dyn_dim",
    "hp_trial": "dlt_hp_trial",
    "fusion": "dlt_fusion",
}
LEGACY_DROP = {"setting", "model_id", "data", "data_path"}
LEGACY_SETTING_FIELDS = {"ll": "label_len", "dm": "d_model", "nh": "n_heads", "el": "e_layers", "dl": "d_layers", "df": "d_ff", "fc": "factor"}
MODEL_ALIASES = {"SMamba": "S_Mamba"}


def is_valid_result(row):
    mse = row.get("mse")
    return str(row.get("status", "")).startswith("ok") and isinstance(mse, (int, float)) and math.isfinite(mse)


def device_class(row):
    """"gpu", "cpu", or None for a record written before the device was recorded."""
    device = str(row.get("device") or "").strip().lower()
    if not device:
        return None
    return "cpu" if device == "cpu" else "gpu"


@contextmanager
def _locked(filepath):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
    with open(filepath + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value).replace("\t", " ").replace("\n", " ")


def _parse(col, text):
    if text == "":
        return None
    if col in STRING_COLUMNS:
        return text
    if col in BOOL_COLUMNS or text in ("True", "False"):
        return text == "True"
    if col in JSON_COLUMNS or text[:1] in "[{":
        try:
            return json.loads(text)
        except ValueError:
            return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _read_header(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None
    with open(filepath) as f:
        return f.readline().rstrip("\n").split("\t")


def _ordered_columns(keys):
    keys = list(keys)
    return [c for c in RESULT_COLUMNS if c in keys] + [k for k in keys if k not in RESULT_COLUMNS]


def write_rows(filepath, rows, columns=None):
    columns = columns or RESULT_COLUMNS
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(_cell(row.get(c)) for c in columns) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, filepath)


def append_result(filepath, row):
    """Append one row under a file lock; a row with columns the file lacks rewrites the file with the extended header."""
    with _locked(filepath):
        header = _read_header(filepath)
        extra = [k for k in row if k not in (header or RESULT_COLUMNS)]
        if header is None:
            write_rows(filepath, [row], _ordered_columns(list(RESULT_COLUMNS) + extra))
            return
        if extra:
            rows = _read_file(filepath)
            write_rows(filepath, rows + [row], header + extra)
            return
        with open(filepath, "a") as f:
            f.write("\t".join(_cell(row.get(c)) for c in header) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _read_file(filepath):
    rows = []
    with open(filepath) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cells = line.split("\t")
            rows.append({col: _parse(col, cells[i] if i < len(cells) else "") for i, col in enumerate(header)})
    return rows


def _read_jsonl(filepath):
    rows = []
    for line in open(filepath):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_results(*paths, missing_ok=True):
    """Rows of one or more TSV stores (legacy .jsonl records are converted on the fly)."""
    rows = []
    for path in paths:
        path = str(path)
        if not os.path.exists(path):
            if missing_ok:
                continue
            raise FileNotFoundError(path)
        if path.endswith(".jsonl"):
            rows += [migrate_record(r) for r in _read_jsonl(path)]
        else:
            rows += _read_file(path)
    return rows


def latest(rows, key=("des", "seed")):
    """The last row per key (re-runs replace earlier rows), in first-appearance order."""
    out = {}
    for r in rows:
        k = tuple(r.get(c) for c in key)
        out.pop(k, None)
        out[k] = r
    return list(out.values())


def _sort_key(row):
    ds = row.get("dataset")
    pos = DATASET_ORDER.index(ds) if ds in DATASET_ORDER else len(DATASET_ORDER)
    return (pos, str(ds), str(row.get("model")), row.get("pred_len") or 0, str(row.get("des")), row.get("seed") or 0, str(row.get("timestamp") or ""))


def compact_results(filepath, dedup=False):
    with _locked(filepath):
        header = _read_header(filepath)
        if header is None:
            return 0
        rows = _read_file(filepath)
        if dedup:
            rows = latest(rows, key=("des", "seed", "dataset", "model", "pred_len"))
        rows.sort(key=_sort_key)
        write_rows(filepath, rows, header)
        return len(rows)


def dataset_of(args):
    return canonical_dataset(os.path.basename(str(getattr(args, "data_path", "")))) or str(getattr(args, "data", "custom"))


def resolve_result_file(args):
    explicit = getattr(args, "result_file", "") or os.environ.get("RESULT_FILE", "")
    if explicit:
        return explicit
    dataset = dataset_of(args)
    ablation_dir = os.environ.get("ABLATION_OUTPUT_DIR")
    if ablation_dir:
        root = ablation_dir.rstrip("/")
        return os.path.join(os.path.dirname(root) or ".", "{}.tsv".format(os.path.basename(root)))
    benchmark_dir = os.environ.get("BENCHMARK_OUTPUT_DIR")
    if benchmark_dir:
        return os.path.join(benchmark_dir, "{}.tsv".format(dataset))
    return os.path.join("./output/long_term_forecast", "{}.tsv".format(dataset))


def _config_value(args, col):
    if col == "time_marks":
        return not getattr(args, "no_time_marks", False)
    if col == "test_batch_size":
        return getattr(args, "test_batch_size", 0) or getattr(args, "batch_size", None)
    return getattr(args, col, None)


def build_result_row(args, metrics, info=None, checkpoint=0):
    model = getattr(args, "model", "")
    mse, mae = float(metrics["mse"]), float(metrics["mae"])
    row = {
        "dataset": dataset_of(args),
        "model": model,
        "seq_len": getattr(args, "seq_len", None),
        "label_len": getattr(args, "label_len", None),
        "pred_len": getattr(args, "pred_len", None),
        "mse": mse,
        "mae": mae,
        "status": "ok" if math.isfinite(mse) and math.isfinite(mae) else "diverged",
        "checkpoint": checkpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "des": getattr(args, "des", ""),
        "seed": getattr(args, "seed", None),
    }
    for col in CONFIG_COLUMNS:
        if col in DLT_ONLY and model not in DLT_MODELS:
            continue
        if col in PLUG_ONLY and model != "DLTPlug":
            continue
        if col in BRANCH_RECIPE and model != "DLTPlug" and (model != "DLT" or getattr(args, col, BRANCH_RECIPE[col]) == BRANCH_RECIPE[col]):
            continue
        if col in STANDALONE_RECIPE and (model != "DLT" or getattr(args, col, STANDALONE_RECIPE[col]) == STANDALONE_RECIPE[col]):
            continue
        if col == "dlt_select_alpha" and getattr(args, "dlt_select", "none") != "significant":
            continue
        if col in BACKBONE_ONLY and model == "DLT":
            continue
        if col == "dlt_gate_init" and getattr(args, "dlt_gate", "none") != "static":
            continue
        if col == "dlt_hp_trial" and getattr(args, "dlt_hp_trial", -1) < 0:
            continue
        if col == "pems_split" and getattr(args, "data", "") != "PEMS":
            continue
        if col in ("d_state", "mamba_impl") and "S_Mamba" not in (model, getattr(args, "plug_base", None)):
            continue
        row[col] = _config_value(args, col)
    for col in RUN_COLUMNS:
        if col in metrics:
            row[col] = metrics[col]
        elif info and col in info:
            row[col] = info[col]
    for k, v in (info or {}).items():
        row[k] = v
    return row


def write_test_result(args, metrics, info=None, checkpoint=0):
    if os.environ.get("DLT_SKIP_RESULT_WRITE") == "1":
        print("Result write skipped (DLT_SKIP_RESULT_WRITE=1)")
        return None
    path = resolve_result_file(args)
    row = build_result_row(args, metrics, info, checkpoint)
    try:
        append_result(path, row)
        print("Results successfully written to: {}".format(path))
        return path
    except OSError as exc:
        print("ERROR: Failed to write results to {}: {}".format(path, exc))
        return None


def migrate_record(rec):
    """A record of the former JSONL stores (--result_jsonl) as a row of the TSV schema."""
    row = {}
    for k, v in rec.items():
        if k in LEGACY_DROP:
            continue
        row[LEGACY_KEYS.get(k, k)] = v
    row["model"] = MODEL_ALIASES.get(row.get("model"), row.get("model"))
    if row.get("plug_base") is not None:
        row["plug_base"] = MODEL_ALIASES.get(row["plug_base"], row["plug_base"])
    if rec.get("data_path"):
        row["dataset"] = canonical_dataset(os.path.basename(str(rec["data_path"])))
    else:
        row["dataset"] = canonical_dataset(rec["dataset"]) if rec.get("dataset") else rec.get("data")
    for key, col in LEGACY_SETTING_FIELDS.items():
        m = re.search(r"_{}(-?\d+)_".format(key), "_" + str(rec.get("setting", "")) + "_")
        if m and col not in row:
            if col in BACKBONE_ONLY and row.get("model") == "DLT":
                continue
            row[col] = int(m.group(1))
    mse, mae = rec.get("mse"), rec.get("mae")
    ok = isinstance(mse, (int, float)) and isinstance(mae, (int, float)) and math.isfinite(mse) and math.isfinite(mae)
    row["status"] = "ok" if ok else "diverged"
    row.setdefault("checkpoint", None)
    row.setdefault("timestamp", None)
    return row


def migrate_jsonl(src, dst):
    rows = [migrate_record(r) for r in _read_jsonl(src)]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    write_rows(dst, rows, _ordered_columns(keys))
    return len(rows)


def _main():
    parser = argparse.ArgumentParser(description="TSV result-store utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sort", help="canonically reorder TSV result files (--dedup keeps the last row per run)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--dedup", action="store_true")
    p = sub.add_parser("migrate", help="convert a legacy JSONL record file into the TSV schema")
    p.add_argument("src")
    p.add_argument("dst")
    p = sub.add_parser("show", help="print selected columns of a TSV store")
    p.add_argument("path")
    p.add_argument("--columns", default="dataset,model,pred_len,des,seed,mse,mae")
    a = parser.parse_args()
    if a.command == "sort":
        for path in a.paths:
            print("sorted {} ({} rows)".format(path, compact_results(path, dedup=a.dedup)))
    elif a.command == "migrate":
        print("{} -> {} ({} rows)".format(a.src, a.dst, migrate_jsonl(a.src, a.dst)))
    else:
        cols = a.columns.split(",")
        print("\t".join(cols))
        for r in read_results(a.path):
            print("\t".join(_cell(r.get(c)) for c in cols))


if __name__ == "__main__":
    _main()
