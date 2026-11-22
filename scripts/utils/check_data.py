"""Check that every dataset is in data/ (or ../data) with the published size: python scripts/utils/check_data.py [group ...]."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import paths  # noqa: E402

GROUPS = {
    'ETT': {'ETTh1.csv': (17420, 8), 'ETTh2.csv': (17420, 8), 'ETTm1.csv': (69680, 8), 'ETTm2.csv': (69680, 8)},
    'Exchange': {'exchange_rate.csv': (7588, 9)},
    'Weather': {'weather.csv': (52696, 22)},
    'ECL': {'electricity.csv': (26304, 322)},
    'Traffic': {'traffic.csv': (17544, 863)},
    'ILI': {'national_illness.csv': (966, 8)},
    'PEMS': {'PEMS03.npz': (26208, 358, 1), 'PEMS04.npz': (16992, 307, 3), 'PEMS07.npz': (28224, 883, 1),
             'PEMS08.npz': (17856, 170, 3)},
    'Solar': {'solar_AL.txt': (52560, 137)},
}


def shape(path):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            return z['data'].shape
    with open(path, 'rb') as f:
        first = f.readline()
        rows = 1 + sum(1 for line in f if line.strip())
    header = path.suffix == '.csv'
    return rows - header, first.count(b',') + 1


def main():
    groups = sys.argv[1:] or list(GROUPS)
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise SystemExit(f'unknown group(s) {unknown}; choose from {list(GROUPS)}')
    print(f'data directory: {paths.DATA}')
    bad = []
    for g in groups:
        for name, want in GROUPS[g].items():
            p = paths.path_of(name)
            where = p.relative_to(paths.DATA) if p.is_relative_to(paths.DATA) else p
            if not p.exists():
                bad.append(name)
                print(f'MISSING  {g:8s} {name:22s} {where}')
                continue
            try:
                got = tuple(shape(p))
            except Exception as e:                                  # a truncated or foreign file
                got = f'unreadable ({type(e).__name__})'
            ok = got == want
            bad += [] if ok else [name]
            size = ' x '.join(map(str, got)) if isinstance(got, tuple) else got
            print(f'{"OK" if ok else "SIZE":8s} {g:8s} {name:22s} {size:22s}' +
                  ('' if ok else f'expected {" x ".join(map(str, want))}  ') + f'{where}')
    print('\nall files present and complete' if not bad else f'\nmissing or incomplete: {", ".join(bad)}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
