"""Print the job lines of job lists that have no result yet (a row with the job's --des and --seed in its --result_file)."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.paths import jobs_file  # noqa: E402
from src.utils.result_io import read_results  # noqa: E402

_done = {}


def finished(path):
    path = str(ROOT / path) if path and not os.path.isabs(path) else path
    if path not in _done:
        _done[path] = {(r.get("des"), int(r.get("seed") or 2021)) for r in read_results(path) if r.get("des") is not None}
    return _done[path]


def missing(job_list):
    out = []
    with open(jobs_file(job_list)) as f:
        lines = f.read().splitlines()
    for line in lines:
        if not line.strip():
            continue
        arg = lambda k, d=None: (re.search(r"--{} (\S+)".format(k), line) or [None, d])[1]
        key = (arg("des"), int(arg("seed", 2021)))
        if key not in finished(arg("result_file")):
            out.append(line)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        for line in missing(name):
            print(line)
