import re
import sys
from pathlib import Path

from src.configs.dlt_profiles import DATASET_META, NO_MARKS, PEMS, canonical_dataset, emit_cli_args, get_pred_lens, protocol, resolve_meta
from src.utils.paths import REPO

OFFICIAL = REPO / "scripts" / "benchmark" / "official"
BASELINES = ("S_Mamba", "iTransformer", "PatchTST", "RLinear", "DLinear", "TiDE")
DISPLAY = {"S_Mamba": "S-Mamba"}
SCRIPTS = {
    "S_Mamba": ("S_Mamba", "S_Mamba"),
    "iTransformer": ("iTransformer", "iTransformer"),
    "PatchTST": ("TSLib", "PatchTST"),
    "DLinear": ("TSLib", "DLinear"),
    "TiDE": ("TSLib", "TiDE"),
}
NOT_SETTINGS = {
    "is_training",
    "root_path",
    "data_path",
    "model_id",
    "model",
    "data",
    "features",
    "seq_len",
    "label_len",
    "pred_len",
    "enc_in",
    "dec_in",
    "c_out",
    "des",
    "itr",
    "task_name",
    "gpu",
    "use_gpu",
    "use_multi_gpu",
    "devices",
    "checkpoints",
    "num_workers",
    "freq",
    "target",
}
RLINEAR_TRAIN = "--train_epochs 15 --patience 10"
RLINEAR = {
    "ETTh1": "--learning_rate 0.005 --batch_size 128",
    "ETTm1": "--learning_rate 0.005 --batch_size 128",
    "ETTh2": "--learning_rate 0.005 --batch_size 256",
    "ETTm2": "--learning_rate 0.005 --batch_size 128",
    "Weather": "--learning_rate 0.005 --batch_size 128",
    "ECL": "--learning_rate 0.001 --batch_size 16 --dropout 0",
}
FALLBACK = {
    "RLinear": {"Exchange": "ETTh1", **{d: "ECL" for d in ("Traffic", "Solar") + PEMS}},
    "PatchTST": {d: "ECL" for d in NO_MARKS},
    "TiDE": {d: "ETTh1" for d in DATASET_META if d != "ETTh1"},
    "DLinear": {},
}
_CACHE = {}


def _lines(folder):
    """{(data file, pred_len, model): ({argument: value}, script)} of the run.py lines of the official scripts in a folder."""
    out = {}
    for path in sorted((OFFICIAL / folder).rglob("*.sh")):
        txt = re.sub(r"\\[ \t]*\r?\n", " ", path.read_text(errors="replace"))
        var = dict(re.findall(r"^\s*(\w+)=(\S+)", txt, re.M))
        for line in txt.splitlines():
            if "run.py" not in line or line.lstrip().startswith("#"):
                continue
            toks, d, i = line.split(), {}, 0
            while i < len(toks):
                if toks[i].startswith("--"):
                    val = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
                    if val.startswith("$"):
                        val = var.get(val[1:].strip("{}"), val)
                    d[toks[i][2:]] = val.strip("'\"")
                    i += 2 if val else 1
                else:
                    i += 1
            key = (Path(d.get("data_path", "")).name, str(d.get("pred_len")), d.get("model", "").strip("$"))
            out.setdefault(key, (d, str(path.relative_to(REPO))))
    return out


def official_line(model, ds, H):
    """(settings dict, script) of the official line, or (None, None)."""
    folder, name = SCRIPTS[model]
    if folder not in _CACHE:
        _CACHE[folder] = _lines(folder)
    return _CACHE[folder].get((DATASET_META[ds]["data_path"], str(H), name), (None, None))


def _settings(model, d):
    keep = {k: v for k, v in d.items() if k not in NOT_SETTINGS}
    if model == "TiDE" and "c_out" in d:
        keep["c_out"] = d["c_out"]
    return " ".join("--{} {}".format(k, v) if v != "" else "--{}".format(k) for k, v in keep.items())


def args(model, ds, H):
    """(arguments, source) of the backbone's official settings for this dataset and horizon, without the protocol."""
    from src.configs.model_parser_dict import canonical_model

    model, ds = canonical_model(model), canonical_dataset(ds)
    if model not in BASELINES:
        raise ValueError("{} is not one of the baselines {}".format(model, BASELINES))
    if model == "RLinear":
        src_ds = ds if ds in RLINEAR else FALLBACK["RLinear"][ds]
        src = "scripts/benchmark/official/RLinear/script.md" + ("" if src_ds == ds else " ({} configuration)".format(src_ds))
        return "{} {}".format(RLINEAR[src_ds], RLINEAR_TRAIN), src
    d, path = official_line(model, ds, H)
    if d is not None:
        return _settings(model, d), path
    other = FALLBACK.get(model, {}).get(ds)
    if other is not None:
        h = H if official_line(model, other, H)[0] is not None else 96
        d, path = official_line(model, other, h)
        if d is not None:
            return _settings(model, d), "{} ({} configuration{})".format(path, other, ", horizon {}".format(h) if h != H else "")
    if model == "DLinear":
        return "", "Time-Series-Library defaults (no script for this dataset)"
    raise KeyError("no official or fallback setting for {} on {}-{}".format(model, ds, H))


def protocol_args(ds):
    return " ".join(emit_cli_args(protocol(ds)))


def runtime_args(model, ds):
    """Options that change memory or speed but not the results (TiDE on data with more than 100 variables)."""
    return "--tide_checkpoint" if model == "TiDE" and DATASET_META[canonical_dataset(ds)]["enc_in"] > 100 else ""


def to_cli_string(model, ds, H, *, with_meta=False, runtime=True):
    """Meta flags (label_len 48) + the official-loader protocol + the official settings (+ runtime options)."""
    meta = resolve_meta(ds, H, label_len=48) if with_meta else None
    parts = [" ".join(emit_cli_args(protocol(ds), meta)), args(model, ds, H)[0]]
    if runtime:
        parts.append(runtime_args(model, ds))
    return " ".join(p for p in parts if p)


def _main():
    from src.configs.model_parser_dict import canonical_model

    for model in [canonical_model(m) for m in sys.argv[1:]] or BASELINES:
        print("## {}".format(DISPLAY.get(model, model)))
        for ds in DATASET_META:
            for H in get_pred_lens(ds):
                a, src = args(model, ds, H)
                print("{:8s} {:4d}  {:70s}  {}".format(ds, H, a or "(defaults)", src))


if __name__ == "__main__":
    _main()
