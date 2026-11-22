"""Fixed locations: data/ (datasets), temp/ (checkpoints, job lists, raw predictions), log/, output/ (tracked results)."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if (REPO / "data").exists():
    DATA = REPO / "data"
elif (REPO / "dataset").exists():
    DATA = REPO / "dataset"
elif (REPO.parent / "data").is_dir():
    DATA = REPO.parent / "data"
else:
    DATA = REPO / "data"
TEMP = REPO / "temp"
LOG = REPO / "log"
OUTPUT = REPO / "output"
CHECKPOINTS = TEMP / "checkpoints"
JOBS = TEMP / "jobs"
MAMBA_SRC = REPO.parent / "mamba"
CAUSAL_CONV1D_SRC = REPO.parent / "causal-conv1d"

SUBDIR = {
    "ETTh1.csv": "ETT-small",
    "ETTh2.csv": "ETT-small",
    "ETTm1.csv": "ETT-small",
    "ETTm2.csv": "ETT-small",
    "weather.csv": "weather",
    "electricity.csv": "electricity",
    "traffic.csv": "traffic",
    "exchange_rate.csv": "exchange_rate",
    "national_illness.csv": "illness",
    "solar_AL.txt": "Solar",
    "PEMS03_ids.txt": "PEMS",
    **{"PEMS0{}{}".format(i, ext): "PEMS" for i in (3, 4, 7, 8) for ext in (".npz", "_edges.csv")},
}


def root_of(file):
    sub = DATA / SUBDIR.get(file, "")
    if (sub / file).exists() or not (DATA / file).exists():
        return sub
    return DATA


def path_of(file):
    return root_of(file) / file


def exists(file):
    return path_of(file).exists()


def rel(path):
    path = Path(os.path.abspath(path))
    try:
        return "./" + str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def jobs_file(name):
    for cand in (Path(name), JOBS / name, JOBS / "jobs_{}.txt".format(name)):
        if cand.is_file():
            return cand.resolve()
    raise SystemExit("job list '{}' not found (looked for {}, {}, {})".format(name, name, JOBS / name, JOBS / "jobs_{}.txt".format(name)))


def _main():
    cmd, arg = (sys.argv[1:] + ["", ""])[:2]
    out = {
        "data": lambda: DATA,
        "temp": lambda: TEMP,
        "log": lambda: LOG,
        "output": lambda: OUTPUT,
        "root": lambda: root_of(arg),
        "file": lambda: path_of(arg),
        "jobs": lambda: jobs_file(arg),
    }.get(cmd)
    if out is None:
        raise SystemExit("usage: python -m src.utils.paths data|temp|log|output|root <file>|file <file>|jobs <name>")
    if cmd in ("temp", "log", "output"):
        os.makedirs(out(), exist_ok=True)
    print(out())


if __name__ == "__main__":
    _main()
