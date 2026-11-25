"""Print the job lines of job lists that have no result yet: no row with the job's --des and --seed in its --result_file
from the job's device (--no_use_gpu: the CPU, otherwise the GPU; a row without a recorded device counts for both), so
that a GPU list runs again what only the CPU has run."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.paths import jobs_file  # noqa: E402
from src.utils.result_io import device_class, read_results  # noqa: E402

_done = {}


def finished(path):
    path = str(ROOT / path) if path and not os.path.isabs(path) else path
    if path not in _done:
        done = {}                                      # (des, seed) -> devices of its rows ("gpu", "cpu", None)
        for r in read_results(path):
            if r.get("des") is not None:
                done.setdefault((r.get("des"), int(r.get("seed") or 2021)), set()).add(device_class(r))
        _done[path] = done
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
        device = "cpu" if "--no_use_gpu" in line.split() else "gpu"
        if not finished(arg("result_file")).get(key, set()) & {device, None}:
            out.append(line)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        for line in missing(name):
            print(line)
