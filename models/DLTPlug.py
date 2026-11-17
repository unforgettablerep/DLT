"""DLTPlug: the DLT cross-variable branch as a plug-in for any Time-Series-Library forecaster.

The backbone (``--plug_base``, e.g. DLinear, PatchTST, iTransformer) is built from ``models/`` with the usual TSLib
arguments and is left unchanged. The branch works on its own instance-normalised copy of the input; neighbours,
kernels and gates are the ``--dlt_*`` arguments of models/DLT.py. ``--dlt_variant ci`` gives the backbone alone
through the same code path. Here x~ is the input normalised per window and variable (mean mu_i, std sigma_i),
z_i = sum_{j in N(i)} K_ij * s_j the distributed-lag sum over the neighbours' series s_j.

How the branch enters (``--dlt_fusion``):
  output  y_hat = Backbone(X) + sigma_i * W_c z_i(x~)                   (Eq. 2 of the paper; the default)
          z over the look-back window, mapped to the horizon by a free linear head W_c (T -> H).
  refine  y_hat = y_b + sigma_i * [z_i(s)]_{T+1..T+H},  s_j = [x~_j ; (y_b,j - mu_j) / sigma_j],  y_b = Backbone(X)
          the kernels run over the neighbours' observed past followed by the backbone's forecast of them: a
          horizon step t+h takes the neighbours' observed values at lags >= h and their forecasts at lags < h.
          No free head: the correction keeps the distributed-lag form in time. Coefficients start at zero, so
          training starts from the backbone alone.
  input   y_hat = Backbone(X + g_i * sigma_i * z_i(x~))
          the neighbour signal enters the backbone's input; per-variable gains g start at zero.

How it is trained (``--dlt_train``):
  joint      one loss on y_hat; backbone and branch adapt to each other (the default).
  decoupled  the backbone is trained by its own loss L(y_b, y) as if it were alone, the branch by L(y_b' + c, y)
             with y_b' = y_b without gradient (boosting on the backbone's error); output and refine only.

``--dlt_select variable`` (output, refine): after training, the correction is kept only for the variables whose
validation MSE it lowers (a model-selection step on validation data; exp_long_term_forecasting.test).
"""
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DLT

FUSIONS = ('output', 'refine', 'input')
TRAINING = ('joint', 'decoupled')


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        base = getattr(configs, "plug_base", "DLinear")
        if base in ("DLT", "DLTPlug"):
            raise ValueError("--plug_base must be a forecaster other than DLT / DLTPlug")
        self.base = importlib.import_module(f"models.{base}").Model(configs)
        self.use_dlt = getattr(configs, "dlt_variant", "dltdyn") != "ci"
        self.fusion = getattr(configs, "dlt_fusion", "output")
        self.training_mode = getattr(configs, "dlt_train", "joint")
        if self.fusion not in FUSIONS:
            raise ValueError(f"--dlt_fusion must be one of {FUSIONS}, got {self.fusion}")
        if self.training_mode not in TRAINING:
            raise ValueError(f"--dlt_train must be one of {TRAINING}, got {self.training_mode}")
        if self.fusion == "input" and self.training_mode == "decoupled":
            raise ValueError("--dlt_train decoupled needs a separate correction (--dlt_fusion output or refine)")
        self.mae = getattr(configs, "loss", "MSE").upper() == "MAE"
        self.select_mask = None                      # (C,) per-variable keep mask, set by --dlt_select variable
        self._y_backbone = None
        if self.use_dlt:
            self.dlt = DLT.Model(configs, cross_only=True)
            if self.fusion in ("refine", "input"):
                del self.dlt.cross_head              # no free T -> H head
                with torch.no_grad():
                    for name in ("alpha", "a", "w"):  # start from the backbone alone
                        if hasattr(self.dlt, name):
                            getattr(self.dlt, name).zero_()
            if self.fusion == "input":
                self.dlt.in_gain = nn.Parameter(torch.zeros(configs.enc_in))
                with torch.no_grad():                # the gain starts at zero; the coefficients may not
                    for name in ("alpha", "a", "w"):
                        if hasattr(self.dlt, name):
                            getattr(self.dlt, name).normal_(0, 0.01)

    def prepare(self, train_data):
        if self.use_dlt:
            self.dlt.prepare(train_data)

    @staticmethod
    def _norm(x_enc):
        x = x_enc.permute(0, 2, 1)                   # (B, C, T)
        sd = x.std(-1, keepdim=True) + 1e-5
        mu = x.mean(-1, keepdim=True)
        return x, mu, sd

    def _backbone(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        y = self.base(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(y, tuple):                     # backbones that also return attention maps
            y = y[0]
        return y[:, -self.pred_len:, :]

    def refine(self, x_enc, y_b):
        """sigma_i * [z_i(s)]_{T+1..T+H} over s = [x~ ; normalised forecast]: (B, H, C)"""
        x, mu, sd = self._norm(x_enc)
        xn = (x - mu) / sd
        s = torch.cat([xn, (y_b.permute(0, 2, 1) - mu) / sd], dim=-1)          # (B, C, T + H)
        self.dlt._check_ready()
        z = self.dlt.cross(s, mod_input=xn)[..., -self.pred_len:]
        if self.dlt.gate == "static":
            z = torch.sigmoid(self.dlt.gate_logit)[:, None] * z
        return (z * sd).permute(0, 2, 1)

    def parts(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """(backbone forecast, correction), both (B, H, C); output and refine fusions"""
        y_b = self._backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.fusion == "output":
            c = self.dlt.correction(x_enc)
        else:
            c = self.refine(x_enc, y_b.detach() if self.training_mode == "decoupled" else y_b)
        return y_b, c

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if not self.use_dlt:
            return self._backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.fusion == "input":
            x, mu, sd = self._norm(x_enc)
            self.dlt._check_ready()
            z = self.dlt.cross((x - mu) / sd)                                   # (B, C, T)
            x_in = x_enc + (self.dlt.in_gain[:, None] * z * sd).permute(0, 2, 1)
            return self._backbone(x_in, x_mark_enc, x_dec, x_mark_dec)
        y_b, c = self.parts(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.select_mask is not None:
            c = c * self.select_mask.to(c.dtype)
        if self.training_mode == "decoupled" and self.training:
            self._y_backbone = y_b
            return y_b.detach() + c
        return y_b + c

    def extra_loss(self, batch_x, outputs, batch_y):
        """decoupled training: the backbone's own loss (exp_long_term_forecasting adds it to the loss of y_hat)"""
        if not (self.use_dlt and self.training_mode == "decoupled" and self.training) or self._y_backbone is None:
            return outputs.new_zeros(())
        y_b = self._y_backbone[:, -batch_y.shape[1]:, -batch_y.shape[2]:]
        self._y_backbone = None
        return F.l1_loss(y_b, batch_y) if self.mae else F.mse_loss(y_b, batch_y)
