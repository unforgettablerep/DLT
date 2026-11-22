import argparse
import sys

from src.configs.dlt_profiles import get_pred_lens, list_datasets, to_cli_string


def main():
    parser = argparse.ArgumentParser(description="Emit DLT CLI flags for a (dataset, pred_len) pair.", usage="python -m src.configs.dlt_cli [dataset] [pred_len] [options]")
    parser.add_argument("dataset", nargs="?", help="Dataset name (e.g. ETTh1)")
    parser.add_argument("pred_len", nargs="?", type=int, help="Prediction horizon")
    parser.add_argument("--with-meta", action="store_true", help="Include dataset metadata flags (root_path, data_path, enc_in, ...)")
    parser.add_argument("--profile", default="default", choices=["default", "efficiency"], help="default: the paper's stand-alone recipe; efficiency: S-Mamba Fig. 5")
    parser.add_argument("--list-datasets", action="store_true", help="List all supported dataset names and exit")
    parser.add_argument("--pred-lens", action="store_true", help="List pred_lens for the given dataset and exit")
    args = parser.parse_args()
    if args.list_datasets:
        for dataset in list_datasets():
            print(dataset)
        return
    if args.dataset is None:
        parser.error("dataset is required (unless --list-datasets)")
    if args.pred_lens:
        for pred_len in get_pred_lens(args.dataset):
            print(pred_len)
        return
    if args.pred_len is None:
        parser.error("pred_len is required (unless --pred-lens / --list-datasets)")
    try:
        print(to_cli_string(args.dataset, args.pred_len, with_meta=args.with_meta, profile=args.profile))
    except KeyError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
