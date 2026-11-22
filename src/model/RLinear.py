import torch
import torch.nn as nn

from src.layers.RevIN import RevIN


class Model(nn.Module):
    """RLinear (arXiv:2305.10721) as in github.com/plumprc/RTSF: RevIN, dropout, Linear(seq_len -> pred_len) shared by all variates."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, "individual", False)
        channels = configs.enc_in
        if self.individual:
            self.Linear = nn.ModuleList([nn.Linear(configs.seq_len, configs.pred_len) for _ in range(channels)])
        else:
            self.Linear = nn.Linear(configs.seq_len, configs.pred_len)
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(channels) if getattr(configs, "use_norm", 1) else None

    def forecast(self, x):
        # x: [B, L, C]
        x = self.rev(x, "norm") if self.rev is not None else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.stack([proj(x[:, :, i]) for i, proj in enumerate(self.Linear)], dim=-1)
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        return self.rev(pred, "denorm") if self.rev is not None else pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.forecast(x_enc)[:, -self.pred_len:, :]
