"""Fixed locations of the datasets, the experiment work directory and the Mamba sources.

Every script takes them from here: no environment variable and no path argument. The repository may sit next to
the data and the Mamba sources, as on the GPU machine:

    ~/DLT              this repository
    ~/data             the datasets (used when DLT/dataset does not exist)
    ~/mamba            state-spaces/mamba source checkout (used when mamba_ssm is not installed in the environment)
    ~/causal-conv1d    Dao-AILab/causal-conv1d source checkout (same)

    DATA = <repo>/dataset, or <repo>/../data if <repo>/dataset does not exist. The datasets, either in the
           Time-Series-Library layout
               ETT-small/ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv   weather/weather.csv   electricity/electricity.csv
               traffic/traffic.csv   exchange_rate/exchange_rate.csv   illness/national_illness.csv
               PEMS/PEMS03.npz PEMS04.npz PEMS07.npz PEMS08.npz (+ PEMS0x_edges.csv)   Solar/solar_AL.txt
           or with all files directly in the directory. scripts/download_data.sh fetches them all,
           scripts/check_data.py checks them.
    WORK = <repo>/work: job lists, logs, checkpoints and result records of the experiment queues.
    MAMBA_SRC, CAUSAL_CONV1D_SRC = <repo>/../mamba, <repo>/../causal-conv1d (models/SMamba.py looks there too).

dataset/ and work/ are ignored by git. If the data or an existing work directory is somewhere else, make
`dataset` or `work` a symbolic link to it, or change the lines below.

For shell scripts:  python scripts/paths.py data | work | root <file> | file <file> | jobs <list name or file>
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'dataset' if (REPO / 'dataset').exists() or not (REPO.parent / 'data').is_dir() else REPO.parent / 'data'
WORK = REPO / 'work'
MAMBA_SRC = REPO.parent / 'mamba'
CAUSAL_CONV1D_SRC = REPO.parent / 'causal-conv1d'

SUBDIR = {'ETTh1.csv': 'ETT-small', 'ETTh2.csv': 'ETT-small', 'ETTm1.csv': 'ETT-small', 'ETTm2.csv': 'ETT-small',
          'weather.csv': 'weather', 'electricity.csv': 'electricity', 'traffic.csv': 'traffic',
          'exchange_rate.csv': 'exchange_rate', 'national_illness.csv': 'illness', 'solar_AL.txt': 'Solar',
          **{f'PEMS0{i}{ext}': 'PEMS' for i in (3, 4, 7, 8) for ext in ('.npz', '_edges.csv')}, 'PEMS03_ids.txt': 'PEMS'}


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
