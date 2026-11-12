"""S-Mamba (Wang et al., "Is Mamba Effective for Time Series Forecasting?", Neurocomputing 2025, arXiv:2403.11144),
re-implemented for this Time-Series-Library snapshot after the official code (github.com/wzhwzhwzh0921/S-D-Mamba).

    tokens   one per variate: Linear(T -> D) of its look-back window; one more per time-mark feature (as iTransformer)
    layer    h <- LayerNorm(h + Mamba_fwd(h) + flip(Mamba_bwd(flip(h))))     bidirectional Mamba over the variate tokens
             h <- LayerNorm(h + FFN(h))                                      FFN: 1x1 conv, activation, 1x1 conv
    output   Linear(D -> H) of every variate token; input instance normalisation as in iTransformer (--use_norm)

The Mamba blocks use d_state = --d_state, d_conv = 2 and expand = 1, as fixed in the official model. The block is
``mamba_ssm.Mamba`` (CUDA only) when mamba_ssm can be imported, from the environment or from the source checkouts
next to the repository (../mamba and ../causal-conv1d, the fixed locations of scripts/paths.py), and
``layers/MambaPy.py`` otherwise: a pure-PyTorch block with the same parameters and outputs but a sequential scan
(slower). ``--mamba_impl ssm|torch`` forces one of them; the choice and its reason are printed and the choice is
stored in the result record. ``python scripts/gpu/check_mamba.py`` checks the installation.

Differences from the official file: the encoder layer there also builds two Mamba blocks and an attention layer
that never enter the forward pass; they are left out here, so the parameter count is that of the working model.
Official hyper-parameters per dataset: scripts/long_term_forecast/*_script/SMamba*.sh.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted

# source checkouts next to the repository (same locations as MAMBA_SRC / CAUSAL_CONV1D_SRC in scripts/paths.py)
SOURCES = [Path(__file__).resolve().parents[1].parent / d for d in ("causal-conv1d", "mamba")]


def import_mamba_ssm():
    """mamba_ssm.Mamba from the environment, else from the source checkouts next to the repository.
    Raises ImportError with both reasons if neither works."""
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError as env_error:
        added = [str(d) for d in SOURCES if d.is_dir() and str(d) not in sys.path]
        if not added:
            raise
        sys.path[:0] = added
        for name in [m for m in sys.modules if m.split(".")[0] in ("mamba_ssm", "causal_conv1d")]:
            del sys.modules[name]                                   # partial imports of the first attempt
        try:
            from mamba_ssm import Mamba
            return Mamba
        except ImportError as src_error:
            for d in added:
                sys.path.remove(d)
            raise ImportError(f"environment: {env_error}; source checkouts {', '.join(added)}: {src_error}") from None


def mamba_block(impl="auto", on_gpu=True):
    """(class, name, note) of the Mamba block: mamba_ssm if it can be imported (or forced), else the PyTorch block.
    mamba_ssm runs on CUDA only, so "auto" takes the PyTorch block for a run on the CPU."""
    if impl == "auto" and not on_gpu:
        from layers.MambaPy import Mamba
        return Mamba, "torch", "this run is on the CPU; mamba_ssm needs CUDA"
    if impl in ("auto", "ssm"):
        try:
            Mamba = import_mamba_ssm()
            origin = sys.modules[Mamba.__module__.split(".")[0]].__file__
            try:
                import causal_conv1d  # noqa: F401
                conv = "with causal_conv1d (fused path)"
            except ImportError:
                conv = "without causal_conv1d (unfused path)"
            return Mamba, "mamba_ssm", f"{Path(origin).parent}, {conv}"
        except ImportError as e:
            if impl == "ssm":
                raise
            note = f"mamba_ssm not usable ({e}); see scripts/gpu/check_mamba.py"
    elif impl == "torch":
        note = "forced by --mamba_impl torch"
    else:
        raise ValueError(f"--mamba_impl must be auto, ssm or torch, got {impl}")
    from layers.MambaPy import Mamba
    return Mamba, "torch", note


class BiMambaLayer(nn.Module):
    """Bidirectional Mamba over the tokens, then a position-wise feed-forward network (post-norm)."""

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

    def forward(self, x):                                            # (B, N, D)
        x = self.norm1(x + self.mamba(x) + self.mamba_r(x.flip(dims=[1])).flip(dims=[1]))
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name not in ("long_term_forecast", "short_term_forecast"):
            raise NotImplementedError("S-Mamba supports forecasting only")
        self.seq_len, self.pred_len = configs.seq_len, configs.pred_len
        self.use_norm = getattr(configs, "use_norm", 1)
        on_gpu = bool(getattr(configs, "use_gpu", True)) and torch.cuda.is_available()
        Mamba, self.mamba_impl, note = mamba_block(getattr(configs, "mamba_impl", "auto"), on_gpu)
        print(f"[S-Mamba] Mamba block: {self.mamba_impl} ({note})")
        D = configs.d_model
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, D, configs.embed, configs.freq, configs.dropout)
        self.layers = nn.ModuleList([
            BiMambaLayer(Mamba, D, configs.d_ff, getattr(configs, "d_state", 32), configs.dropout, configs.activation)
            for _ in range(configs.e_layers)])
        self.norm = nn.LayerNorm(D)
        self.projector = nn.Linear(D, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:                                            # as in iTransformer / Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        N = x_enc.shape[-1]
        h = self.enc_embedding(x_enc, x_mark_enc)                    # (B, N + marks, D)
        for layer in self.layers:
            h = layer(h)
        y = self.projector(self.norm(h)).permute(0, 2, 1)[:, :, :N]  # (B, H, N); the mark tokens are dropped
        if self.use_norm:
            y = y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc)[:, -self.pred_len:, :]
