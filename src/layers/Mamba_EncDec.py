import importlib
import importlib.util
import re
import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

from src.utils.paths import CAUSAL_CONV1D_SRC, MAMBA_SRC

SOURCES = [CAUSAL_CONV1D_SRC, MAMBA_SRC]


def _forget(*packages):
    for name in [m for m in sys.modules if m.split(".")[0] in packages]:
        del sys.modules[name]


def _block_class():
    """mamba_ssm.modules.mamba_simple.Mamba without running mamba_ssm/__init__.py (no transformers dependency)."""
    if "mamba_ssm" not in sys.modules:
        spec = importlib.util.find_spec("mamba_ssm")
        if spec is None or spec.submodule_search_locations is None:
            raise ModuleNotFoundError("No module named 'mamba_ssm'", name="mamba_ssm")
        sys.modules["mamba_ssm"] = importlib.util.module_from_spec(spec)
    try:
        cls = importlib.import_module("mamba_ssm.modules.mamba_simple").Mamba
        if getattr(importlib.import_module("mamba_ssm.ops.selective_scan_interface"), "selective_scan_cuda", None) is None:
            raise ImportError(
                "mamba_ssm is installed without its CUDA kernel selective_scan_cuda "
                "(since mamba_ssm 2.3 a source install builds it only with MAMBA_KEEP_CUDA_BUILD=TRUE; the release wheels have it)"
            )
        if not hasattr(sys.modules["mamba_ssm"], "Mamba"):
            sys.modules["mamba_ssm"].Mamba = cls
        return cls
    except (ImportError, OSError) as e:
        _forget("mamba_ssm")
        raise ImportError(str(e)) from None


def import_mamba_ssm():
    """The mamba_ssm block from the environment, else from the source checkouts ../mamba and ../causal-conv1d."""
    try:
        return _block_class()
    except ImportError as env_error:
        added = [str(d) for d in SOURCES if d.is_dir() and str(d) not in sys.path]
        if not added:
            raise
        sys.path[:0] = added
        _forget("mamba_ssm", "causal_conv1d")
        try:
            return _block_class()
        except ImportError as src_error:
            for d in added:
                sys.path.remove(d)
            raise ImportError("environment: {}; source checkouts {}: {}".format(env_error, ", ".join(added), src_error)) from None


def mamba_ssm_version():
    init = getattr(sys.modules.get("mamba_ssm"), "__file__", None)
    m = re.search(r"__version__\s*=\s*[\"']([^\"']+)", Path(init).read_text()) if init else None
    return m.group(1) if m else "unknown"


def mamba_block(impl="auto", on_gpu=True):
    """(class, name, note): mamba_ssm when importable (CUDA only), else the PyTorch block of src/layers/MambaPy.py."""
    if impl == "auto" and not on_gpu:
        from src.layers.MambaPy import Mamba
        return Mamba, "torch", "this run is on the CPU; mamba_ssm needs CUDA"
    if impl in ("auto", "ssm"):
        try:
            Mamba = import_mamba_ssm()
            origin = sys.modules["mamba_ssm"].__file__
            try:
                import causal_conv1d  # noqa: F401
                conv = "with causal_conv1d (fused path)"
            except ImportError:
                conv = "without causal_conv1d (unfused path)"
            return Mamba, "mamba_ssm", "mamba_ssm {} from {}, {}".format(mamba_ssm_version(), Path(origin).parent, conv)
        except ImportError as e:
            if impl == "ssm":
                raise
            note = "mamba_ssm not usable ({}); see python scripts/utils/check_mamba.py".format(e)
    elif impl == "torch":
        note = "forced by --mamba_impl torch"
    else:
        raise ValueError("--mamba_impl must be auto, ssm or torch, got {}".format(impl))
    from src.layers.MambaPy import Mamba
    return Mamba, "torch", note


class EncoderLayer(nn.Module):
    """S-Mamba layer: bidirectional Mamba over the variate tokens, then a position-wise feed-forward network (post-norm)."""

    def __init__(self, Mamba, d_model, d_ff, d_state, dropout, activation):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)
        self.mamba_r = Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x: [B, N, D]
        x = self.norm1(x + self.mamba(x) + self.mamba_r(x.flip(dims=[1])).flip(dims=[1]))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)
