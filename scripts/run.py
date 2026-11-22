import argparse
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.configs.model_parser_dict import canonical_model, model_parser_dict
from src.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.utils.checkpoint import checkpoint_exists, ckpt_prefix, find_latest_checkpoint_seq


def build_parser(argv=None):
    """The parser of a run: common arguments, then those of --model (and, for DLTPlug, of --plug_base)."""
    parser = argparse.ArgumentParser(description="Time Series Forecasting", conflict_handler="resolve")
    for add in model_parser_dict["__all__"]:
        add(parser)
    temp_args, _ = parser.parse_known_args(argv)
    models = []
    if temp_args.model:
        models.append(canonical_model(temp_args.model))
    if models and models[0] == "DLTPlug":
        from src.configs.custom_DLTPlug import add_dltplug_parser

        probe = argparse.ArgumentParser(add_help=False)
        add_dltplug_parser(probe)
        models.append(canonical_model(probe.parse_known_args(argv)[0].plug_base))
    for name in models:
        for add in model_parser_dict.get(name, []):
            add(parser)
    return parser


def parse_args(argv=None):
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    if args.model is None:
        parser.error("the following arguments are required: --model")
    args.model = canonical_model(args.model)
    if args.model not in model_parser_dict:
        parser.error("unknown model {}; choose from {}".format(args.model, [m for m in model_parser_dict if m != "__all__"]))
    if hasattr(args, "plug_base"):
        args.plug_base = canonical_model(args.plug_base)
    return args


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_setting(args, ii):
    return "{}_{}|{}|sl:{}|pl:{}|seed:{}|itr:{}".format(args.model_id, args.model, args.des, args.seq_len, args.pred_len, args.seed, ii)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(id_) for id_ in args.devices.split(",")]
        args.gpu = args.device_ids[0]
    print("Args in experiment:")
    print(args)
    Exp = Exp_Long_Term_Forecast
    if args.is_training == 1:
        for ii in range(args.itr):
            setting = build_setting(args, ii)
            exp = Exp(args)
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)
            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting)
            if args.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif args.is_training == 2:
        if args.checkpoint <= 0:
            args.checkpoint = find_latest_checkpoint_seq(args)
        if args.checkpoint <= 0 or not checkpoint_exists(args, args.checkpoint):
            print("ERROR: no checkpoint {}_{}.pth".format(ckpt_prefix(args), args.checkpoint))
            sys.exit(1)
        setting = build_setting(args, 0)
        exp = Exp(args)
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
    else:
        print("ERROR: --is_training must be 1 (train + test) or 2 (test a checkpoint)")
        sys.exit(1)


if __name__ == "__main__":
    main()
