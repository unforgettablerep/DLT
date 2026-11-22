import argparse
import sys

from src.configs.baseline_profiles import BASELINES, args, to_cli_string
from src.configs.dlt_profiles import get_pred_lens
from src.configs.model_parser_dict import canonical_model


def main():
    parser = argparse.ArgumentParser(description="Emit the official settings of a baseline for a (dataset, pred_len) pair.", usage="python -m src.configs.baseline_cli model [dataset] [pred_len] [options]")
    parser.add_argument("model", help="one of {}".format(", ".join(BASELINES)))
    parser.add_argument("dataset", nargs="?", help="Dataset name (e.g. ETTh1)")
    parser.add_argument("pred_len", nargs="?", type=int, help="Prediction horizon")
    parser.add_argument("--with-meta", action="store_true", help="Include dataset metadata flags (label_len 48)")
    parser.add_argument("--no-runtime", action="store_true", help="Leave out memory / speed options such as --tide_checkpoint")
    parser.add_argument("--pred-lens", action="store_true", help="List pred_lens for the given dataset and exit")
    parser.add_argument("--source", action="store_true", help="Print the official script the settings come from and exit")
    a = parser.parse_args()
    model = canonical_model(a.model)
    if a.dataset is None:
        parser.error("dataset is required")
    if a.pred_lens:
        for pred_len in get_pred_lens(a.dataset):
            print(pred_len)
        return
    if a.pred_len is None:
        parser.error("pred_len is required (unless --pred-lens)")
    try:
        if a.source:
            print(args(model, a.dataset, a.pred_len)[1])
            return
        print(to_cli_string(model, a.dataset, a.pred_len, with_meta=a.with_meta, runtime=not a.no_runtime))
    except (KeyError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
