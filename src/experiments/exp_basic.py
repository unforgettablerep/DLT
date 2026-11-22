import os

import torch

from src.model import DLT, DLTPlug, DLinear, PatchTST, RLinear, S_Mamba, TiDE, iTransformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "DLT": DLT,
            "DLTPlug": DLTPlug,
            "S_Mamba": S_Mamba,
            "iTransformer": iTransformer,
            "PatchTST": PatchTST,
            "RLinear": RLinear,
            "DLinear": DLinear,
            "TiDE": TiDE,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        args = self.args
        if args.use_gpu and args.gpu_type == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available: falling back to CPU")
            args.use_gpu = False
        if args.use_gpu and args.gpu_type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("MPS is not available: falling back to CPU")
            args.use_gpu = False
        if args.use_gpu and args.gpu_type == "cuda":
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if not args.use_multi_gpu else args.devices
            device = torch.device("cuda:0")
            print("Use GPU: cuda:0 (CUDA_VISIBLE_DEVICES={})".format(os.environ.get("CUDA_VISIBLE_DEVICES", "all")))
        elif args.use_gpu and args.gpu_type == "mps":
            device = torch.device("mps")
            print("Use GPU: mps")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_output_subdir(self, setting=None):
        args = self.args
        return "{}_{}_{}_sl{}_pl{}_s{}".format(args.model_id, args.model, args.des, args.seq_len, args.pred_len, args.seed)

    def _get_test_results_path(self, setting=None):
        return os.path.join("./temp/test_results", self._get_output_subdir(setting))

    def _get_results_path(self, setting=None):
        return os.path.join("./temp/results", self._get_output_subdir(setting))
