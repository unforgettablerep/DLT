import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs.model_parser_dict import canonical_model
from src.model.DLT import Model as DLTModel

FUSIONS = ("output", "refine", "input")
TRAINING = ("joint", "decoupled")


class Model(nn.Module):
    """DLT branch as a plug-in for a backbone (--plug_base): fusion --dlt_fusion output|refine|input, training --dlt_train joint|decoupled."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        base = canonical_model(getattr(configs, "plug_base", "DLinear"))
        if base in ("DLT", "DLTPlug"):
            raise ValueError("--plug_base must be a forecaster other than DLT / DLTPlug")
        self.base = importlib.import_module("src.model.{}".format(base)).Model(configs)
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
            self.dlt = DLTModel(configs, cross_only=True)
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

    @property
    def split_output(self):
        """the forecast is the backbone's plus a separate correction (output and refine fusions)"""
        return self.use_dlt and self.fusion != "input"

    def backbone_parameters(self):
        return self.base.parameters()

    def branch_parameters(self):
        return list(self.dlt.parameters()) if self.use_dlt else []

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
        """[z_i(s)]_{T+1..T+H} over s = [x - mu ; y_b - mu] (each neighbour minus the mean of its window): (B, H, C).
        The series are centred but not divided by the window's spread: the data are standardised on the training
        split already, and dividing the backbone's forecast by the spread of a flat window (a rain gauge at zero, a
        stuck sensor) blows it up. The input-dependent coefficients come from the normalised window, as in the
        output fusion. The forecast enters without gradient: the backbone is trained through y_b + c only (joint)
        or through its own loss (decoupled), never to feed the correction."""
        x, mu, sd = self._norm(x_enc)
        s = torch.cat([x - mu, y_b.detach().permute(0, 2, 1) - mu], dim=-1)    # (B, C, T + H)
        self.dlt._check_ready()
        z = self.dlt.cross(s, mod_input=(x - mu) / sd)[..., -self.pred_len:]
        if self.dlt.gate == "static":
            z = torch.sigmoid(self.dlt.gate_logit)[:, None] * z
        return z.permute(0, 2, 1)

    def parts(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """(backbone forecast, correction), both (B, H, C); output and refine fusions"""
        y_b = self._backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        c = self.dlt.correction(x_enc) if self.fusion == "output" else self.refine(x_enc, y_b)
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
