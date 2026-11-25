import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.DLT_layers import (
    GATES,
    IMPLS,
    LAGSET_VARIANTS,
    NEIGHBOUR_MODES,
    VARIANTS,
    lag_basis,
    lag_taps,
    pair_conv,
    select_lags,
    select_neighbours,
)


class Model(nn.Module):
    """DLT: RLinear-style self head plus a distributed-lag cross-variable branch (kernel family --dlt_variant, combination --dlt_gate).

    Stand-alone guards, as for the plug-in (src/model/DLTPlug.py): --dlt_train decoupled trains the self head on its own
    loss first and the branch on its error (stage 2 in exp_long_term_forecasting), --dlt_select variable keeps the
    correction only for the variables whose validation MSE it lowers, --dlt_lr sets the branch's learning rate. Without
    them the forecast takes the unchanged path of forecast().
    """

    def __init__(self, configs, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        self.seq_len = T = configs.seq_len
        self.pred_len = H = configs.pred_len
        self.C = C = configs.enc_in
        self.variant = getattr(configs, "dlt_variant", "dltdyn")
        if self.variant not in VARIANTS:
            raise ValueError("--dlt_variant must be one of {}, got {}".format(VARIANTS, self.variant))
        self.L = L = getattr(configs, "dlt_L", 48)
        self.R = R = getattr(configs, "dlt_R", 4)
        self.neighbours = getattr(configs, "dlt_neighbours", "ccf")
        if self.neighbours not in NEIGHBOUR_MODES:
            raise ValueError("--dlt_neighbours must be one of {}, got {}".format(NEIGHBOUR_MODES, self.neighbours))
        self.k = k = C - 1 if self.neighbours == "all" else max(0, min(getattr(configs, "dlt_k", 8), C - 1))
        self.neighbour_seed = getattr(configs, "seed", 2021)
        self.dyn_dim = d = getattr(configs, "dlt_dyn_dim", 16)
        self.ccf_len = getattr(configs, "dlt_ccf_len", 8000)
        self.K = K = getattr(configs, "dlt_topk", 5)
        self.gate = getattr(configs, "dlt_gate", "none")
        if self.gate not in GATES:
            raise ValueError("--dlt_gate must be one of {}, got {}".format(GATES, self.gate))
        if self.variant in LAGSET_VARIANTS and not 1 <= K <= L + 1:
            raise ValueError("--dlt_topk must be in 1..dlt_L+1 = {}, got {}".format(L + 1, K))
        self.impl = getattr(configs, "dlt_impl", "kernel")
        if self.impl not in IMPLS:
            raise ValueError("--dlt_impl must be one of {}, got {}".format(IMPLS, self.impl))
        if k == 0 and self.variant != "ci":
            print("[DLT] enc_in={}: no other variables, falling back to variant 'ci'".format(C))
            self.variant = "ci"
        if cross_only and self.variant == "ci":
            raise ValueError("cross_only=True needs a cross-variable variant (not 'ci')")
        # stand-alone guards (as the plug-in): self head and cross-variable correction as two parts of the forecast
        self.training_mode = getattr(configs, "dlt_train", "joint")
        self.select = getattr(configs, "dlt_select", "none")
        self.mae = getattr(configs, "loss", "MSE").upper() == "MAE"
        self.select_mask = None                      # (C,) per-variable keep mask, set by --dlt_select variable
        self._y_backbone = None
        guarded = not cross_only and (self.training_mode == "decoupled" or self.select != "none")
        if guarded and (self.variant == "ci" or self.gate == "moe"):
            raise ValueError("--dlt_train decoupled / --dlt_select variable need a cross-variable branch added to the self head "
                             "(not --dlt_variant ci, not --dlt_gate moe)")
        self.split_output = guarded
        if not cross_only:
            self.self_head = nn.Linear(T, H)
        if self.variant == "ci":
            return
        self.register_buffer("idx", torch.zeros(C, k, dtype=torch.long))
        self.register_buffer("lag", torch.zeros(C, k, dtype=torch.long))
        self.register_buffer("prepared", torch.zeros((), dtype=torch.bool))
        self.cross_head = nn.Linear(T, H)
        g = lambda *s: nn.Parameter(0.01 * torch.randn(*s))
        v = self.variant
        if v in ("scalar", "shift", "scalardyn"):
            self.a = g(C, k)
        elif v == "tap3":
            self.w = g(C, k, 3)
        elif v == "full":
            self.w = g(C, k, L + 1)
        elif v in ("dlt", "dltdyn"):
            self.alpha = g(C, k, R)
            self.phi = nn.Parameter(lag_basis(L, R))
        if v in ("scalardyn", "dltdyn"):
            # alpha_ij(X) = alpha_ij * (1 + tanh(MLP([s_i || s_j]))) with a zero-initialised last layer
            self.summ = nn.Linear(T, d)
            self.mod = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, R if v == "dltdyn" else 1))
            nn.init.zeros_(self.mod[-1].weight)
            nn.init.zeros_(self.mod[-1].bias)
        if v in LAGSET_VARIANTS:
            self.register_buffer("tlag", torch.zeros(C, k, K, dtype=torch.long))
            if v == "topk":
                self.w = g(C, k, K)
            else:
                # MillGNN decay K_ij(tau) = a_ij * ReLU(exp(-be tau) - (1 - att) exp(-bi tau)); a_ij signed amplitude
                logit = lambda p: float(np.log(p / (1 - p)))
                self.a = g(C, k)
                self.b_exc = nn.Parameter(logit(0.05) + 0.01 * torch.randn(C, k))
                self.b_inh = nn.Parameter(logit(0.2) + 0.01 * torch.randn(C, k))
                if v == "mill":
                    self.b_att = nn.Parameter(0.01 * torch.randn(C, k))
                else:
                    self.summ = nn.Linear(T, d)
                    self.q_proj = nn.Linear(d, d)
                    self.k_proj = nn.Linear(d, d)
        if self.gate == "moe":
            self.gate_net = nn.Linear(T, 2)
        elif self.gate == "static":
            self.gate_logit = nn.Parameter(torch.full((C,), float(getattr(configs, "dlt_gate_init", -3.0))))

    @torch.no_grad()
    def prepare(self, train_data):
        """Select neighbours and peak lags (and the lag sets of topk / mill / milldyn) on the training split."""
        if self.variant == "ci":
            return
        values = getattr(train_data, "data_x", train_data)
        idx, lag = select_neighbours(values, self.k, self.L, self.ccf_len, device=self.idx.device, mode=self.neighbours, seed=self.neighbour_seed)
        self.idx.copy_(idx)
        self.lag.copy_(lag)
        msg = ""
        if self.variant in LAGSET_VARIANTS:
            tl = select_lags(values, idx, self.K, self.L, self.ccf_len, device=self.idx.device)
            self.tlag.copy_(tl)
            msg = "; K={} lags per pair, lag 0 among them for {:.0%}".format(self.K, (tl == 0).any(-1).float().mean().item())
        self.prepared.fill_(True)
        print(
            "[DLT] variant={}: selected k={} neighbours per variable ({}) (peak lag 0 for {:.0%} of pairs){}".format(
                self.variant, self.k, self.neighbours, (lag == 0).float().mean().item(), msg
            )
        )

    def _pair_features(self, xn):
        s = self.summ(xn)
        # s: [B, C, d] -> [B, C, k, 2d]
        B, C, d = s.shape
        return torch.cat([s.unsqueeze(2).expand(B, C, self.k, d), s[:, self.idx]], dim=-1)

    def cross(self, xn, mod_input=None):
        """xn [B, C, T] -> z [B, C, T]; input-dependent coefficients come from mod_input (default xn)."""
        B, C, T = xn.shape
        k, L, idx, v = self.k, self.L, self.idx, self.variant
        m = xn if mod_input is None else mod_input
        if v == "scalar":
            return (xn[:, idx] * self.a[None, :, :, None]).sum(2)
        if v == "scalardyn":
            a = self.a[None] * (1 + torch.tanh(self.mod(self._pair_features(m)).squeeze(-1)))
            return (xn[:, idx] * a[..., None]).sum(2)
        if v == "shift":
            src = F.pad(xn, (L, 0))[:, idx]
            pos = torch.arange(T, device=xn.device)[None, None] + L - self.lag[..., None]
            g = torch.gather(src, 3, pos.unsqueeze(0).expand(B, -1, -1, -1))
            return (g * self.a[None, :, :, None]).sum(2)
        if v in ("dlt", "dltdyn"):
            alpha = self.alpha
            if v == "dltdyn":
                alpha = alpha[None] * (1 + torch.tanh(self.mod(self._pair_features(m))))
            if self.impl == "dense":
                xp = F.pad(xn, (L, 0)).reshape(B * C, 1, T + L)
                u = F.conv1d(xp, self.phi.flip(-1).unsqueeze(1)).view(B, C, self.R, T)
                if v == "dlt":
                    A = alpha.new_zeros(C, C, self.R).scatter(1, idx[..., None].expand(C, k, self.R), alpha)
                    return torch.einsum("ijr,bjrt->bit", A, u)
                A = alpha.new_zeros(B, C, C, self.R).scatter(2, idx[None, ..., None].expand(B, C, k, self.R), alpha)
                return torch.einsum("bijr,bjrt->bit", A, u)
            if self.impl == "basis":
                xp = F.pad(xn, (L, 0)).reshape(B * C, 1, T + L)
                ug = F.conv1d(xp, self.phi.flip(-1).unsqueeze(1)).view(B, C, self.R, T)[:, idx]
                eq = "bckrt,ckr->bct" if v == "dlt" else "bckrt,bckr->bct"
                return torch.einsum(eq, ug, alpha)
            if v == "dlt":
                return pair_conv(xn, idx, torch.einsum("ckr,rl->ckl", alpha, self.phi))
            K = torch.einsum("bckr,rl->bckl", alpha, self.phi)
            xg = F.pad(xn[:, idx], (L, 0)).reshape(1, B * C * k, T + L)
            z = F.conv1d(xg, K.reshape(B * C * k, 1, L + 1).flip(-1), groups=B * C * k)
            return z.view(B, C, k, T).sum(2)
        if v == "topk":
            return lag_taps(xn, idx, self.tlag, self.w, L)
        if v in ("mill", "milldyn"):
            return lag_taps(xn, idx, self.tlag, self.mill_weights(m), L)
        return pair_conv(xn, idx, self.w)

    def mill_weights(self, xn=None):
        """Tap weights a_ij * Lambda_ij(tau) on the selected lags: [C, k, K], or [B, C, k, K] for milldyn."""
        if self.variant == "mill":
            att = torch.sigmoid(self.b_att)
        else:
            s = self.summ(xn)
            q, kk = self.q_proj(s), self.k_proj(s)
            score = (q.unsqueeze(2) * kk[:, self.idx]).sum(-1) / np.sqrt(q.shape[-1])
            att = torch.softmax(score, dim=-1)
        tau = self.tlag.to(self.a.dtype)
        lam = F.relu(torch.exp(-torch.sigmoid(self.b_exc)[..., None] * tau) - (1 - att[..., None]) * torch.exp(-torch.sigmoid(self.b_inh)[..., None] * tau))
        return self.a[..., None] * lam

    def _check_ready(self):
        if not getattr(self, "_ready", False):
            if not bool(self.prepared):
                raise RuntimeError("DLT: neighbours not selected; call model.prepare(train_data) before training, or load a trained checkpoint")
            self._ready = True

    def correction(self, x_enc):
        """cross_only models: the de-normalised cross-variable term sigma_i * gate * W_c z_i: [B, H, C]."""
        x = x_enc.permute(0, 2, 1)
        sd = x.std(-1, keepdim=True) + 1e-5
        xn = (x - x.mean(-1, keepdim=True)) / sd
        self._check_ready()
        y = self.cross_head(self.cross(xn))
        if self.gate == "static":
            y = torch.sigmoid(self.gate_logit)[:, None] * y
        elif self.gate == "moe":
            raise ValueError("the MoVE-style gate mixes two forecasts; use --dlt_gate none or static for plug-ins")
        return (y * sd).permute(0, 2, 1)

    def forecast(self, x_enc):
        x = x_enc.permute(0, 2, 1)
        # x: [B, C, T]
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True) + 1e-5
        xn = (x - mu) / sd
        y = self.self_head(xn)
        if self.variant != "ci":
            self._check_ready()
            y_cross = self.cross_head(self.cross(xn))
            if self.gate == "none":
                y = y + y_cross
            elif self.gate == "moe":
                gw = torch.softmax(self.gate_net(xn), dim=-1)
                y = gw[..., :1] * y + gw[..., 1:] * y_cross
            else:
                y = y + torch.sigmoid(self.gate_logit)[:, None] * y_cross
        return (y * sd + mu).permute(0, 2, 1)

    def parts(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """(self forecast, cross-variable correction), both [B, H, C] on the data scale; their sum is forecast(x_enc)."""
        x = x_enc.permute(0, 2, 1)
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True) + 1e-5
        xn = (x - mu) / sd
        y = self.self_head(xn) * sd + mu
        self._check_ready()
        c = self.cross_head(self.cross(xn))
        if self.gate == "static":
            c = torch.sigmoid(self.gate_logit)[:, None] * c
        return y.permute(0, 2, 1), (c * sd).permute(0, 2, 1)

    def backbone_parameters(self):
        """the self head: the part trained first by --dlt_train decoupled"""
        return self.self_head.parameters()

    def branch_parameters(self):
        """the cross-variable branch: every parameter but the self head (its learning rate: --dlt_lr)"""
        own = {id(p) for p in self.self_head.parameters()} if hasattr(self, "self_head") else set()
        return [p for p in self.parameters() if id(p) not in own]

    def extra_loss(self, batch_x, outputs, batch_y):
        """--dlt_train decoupled: the self head's own loss (exp_long_term_forecasting adds it to the loss of the forecast)"""
        if not (self.split_output and self.training_mode == "decoupled" and self.training) or self._y_backbone is None:
            return outputs.new_zeros(())
        y = self._y_backbone[:, -batch_y.shape[1]:, -batch_y.shape[2]:]
        self._y_backbone = None
        return F.l1_loss(y, batch_y) if self.mae else F.mse_loss(y, batch_y)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if not self.split_output:
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        y, c = self.parts(x_enc)
        if self.select_mask is not None:
            c = c * self.select_mask.to(c.dtype)
        if self.training_mode == "decoupled" and self.training:
            self._y_backbone = y
            return y.detach() + c
        return y + c
