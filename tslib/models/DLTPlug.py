"""DLTPlug: the DLT cross-variable branch as a plug-in for any Time-Series-Library forecaster.

    y_hat = Backbone(X) + sigma_i * W_c z_i,     z_i = sum_{j in N(i)} K_ij * x~_j   (Eq. 2 of the paper)

The backbone (``--plug_base``, e.g. DLinear, PatchTST, iTransformer) is built from ``models/`` with the
usual TSLib arguments and is left unchanged. The DLT branch works on its own RevIN-normalised copy of the
input and adds a de-normalised correction to the backbone forecast, so it needs no change inside the
backbone. Kernel / neighbour / gate options are the ``--dlt_*`` arguments of models/DLT.py;
``--dlt_variant ci`` gives the backbone alone through the same code path.
"""
import importlib

import torch.nn as nn

from models import DLT


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
        if self.use_dlt:
            self.dlt = DLT.Model(configs, cross_only=True)

    def prepare(self, train_data):
        if self.use_dlt:
            self.dlt.prepare(train_data)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        y = self.base(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(y, tuple):                     # backbones that also return attention maps
            y = y[0]
        y = y[:, -self.pred_len:, :]
        if self.use_dlt:
            y = y + self.dlt.correction(x_enc)
        return y
