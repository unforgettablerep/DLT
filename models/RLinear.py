"""RLinear (Li et al., "Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping", 2023,
arXiv:2305.10721), as in the official code (github.com/plumprc/RTSF, models/RLinear.py and layers/Invertible.py):

    x  <- RevIN(x)                       instance normalisation with learnable affine parameters per variate
    x  <- Dropout(x)                     --dropout (the official --drop, default 0.1)
    y  <- Linear(seq_len -> pred_len)    one map shared by all variates (--individual: one per variate)
    y  <- RevIN^-1(y)

The official model returns (prediction, loss); here it returns the prediction, the loss is the MSE of the
experiment. Without --use_norm (0) RevIN is left out, as the official --rev switch does.
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """layers/Invertible.py of the official code"""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            dim2reduce = tuple(range(1, x.ndim - 1))
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.stdev + self.mean
        raise NotImplementedError(mode)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.individual = getattr(configs, 'individual', False)
        channels = configs.enc_in
        self.Linear = nn.ModuleList([nn.Linear(configs.seq_len, configs.pred_len) for _ in range(channels)]) \
            if self.individual else nn.Linear(configs.seq_len, configs.pred_len)
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(channels) if getattr(configs, 'use_norm', 1) else None

    def forecast(self, x):                                           # x: (B, L, C)
        x = self.rev(x, 'norm') if self.rev is not None else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.stack([proj(x[:, :, i]) for i, proj in enumerate(self.Linear)], dim=-1)
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        return self.rev(pred, 'denorm') if self.rev is not None else pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name not in ('long_term_forecast', 'short_term_forecast'):
            raise ValueError(f'RLinear: forecasting only, not {self.task_name}')
        return self.forecast(x_enc)[:, -self.pred_len:, :]
