"""Fixed locations of the datasets and of the experiment work directory.

Every script takes them from here: no environment variable and no path argument.

    DATA = <repo>/dataset    the datasets, either in the Time-Series-Library layout
                                 dataset/ETT-small/ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv
                                 dataset/weather/weather.csv          dataset/electricity/electricity.csv
                                 dataset/traffic/traffic.csv          dataset/exchange_rate/exchange_rate.csv
                                 dataset/PEMS/PEMS04.npz PEMS08.npz PEMS08_edges.csv
                             or with all files directly in dataset/
    WORK = <repo>/work       job lists, logs, checkpoints and result records of the experiment queues

Both directories are ignored by git. If the data or an existing work directory is somewhere else, make
`dataset` or `work` a symbolic link to it, or change the two lines below.

For shell scripts:  python scripts/paths.py data | work | root <file> | file <file> | jobs <list name or file>
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'dataset'
WORK = REPO / 'work'

SUBDIR = {'ETTh1.csv': 'ETT-small', 'ETTh2.csv': 'ETT-small', 'ETTm1.csv': 'ETT-small', 'ETTm2.csv': 'ETT-small',
          'weather.csv': 'weather', 'electricity.csv': 'electricity', 'traffic.csv': 'traffic',
          'exchange_rate.csv': 'exchange_rate', 'PEMS04.npz': 'PEMS', 'PEMS08.npz': 'PEMS',
          'PEMS04_edges.csv': 'PEMS', 'PEMS08_edges.csv': 'PEMS'}


def root_of(file):
    """Directory that holds `file`: its Time-Series-Library sub-directory if the file is there, else dataset/
    if the file is there; else the sub-directory again (a missing CSV is then downloaded by Time-Series-Library
    from Hugging Face; PEMS files are not)."""
    sub = DATA / SUBDIR.get(file, '')
    if (sub / file).exists() or not (DATA / file).exists():
        return sub
    return DATA


def path_of(file):
    return root_of(file) / file


def exists(file):
    return path_of(file).exists()


def jobs_file(name):
    """A job list by name: an existing file, or work/<name>, or work/jobs_<name>.txt."""
    for cand in (Path(name), WORK / name, WORK / f'jobs_{name}.txt'):
        if cand.is_file():
            return cand.resolve()
    raise SystemExit(f"job list '{name}' not found (looked for {name}, {WORK / name}, {WORK / ('jobs_' + name + '.txt')})")


if __name__ == '__main__':
    cmd, arg = (sys.argv[1:] + ['', ''])[:2]
    out = {'data': lambda: DATA, 'work': lambda: WORK, 'root': lambda: root_of(arg), 'file': lambda: path_of(arg),
           'jobs': lambda: jobs_file(arg)}.get(cmd)
    if out is None:
        raise SystemExit(__doc__)
    if cmd == 'work':
        os.makedirs(WORK, exist_ok=True)
    print(out())
