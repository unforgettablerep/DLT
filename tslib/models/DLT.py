"""DLT: Distributed-Lag Transfer for multivariate time series forecasting.

Two branches on RevIN-normalised inputs (B, C, T):

* intra-variable head   y_self = W_self x_i                 (shared Linear T -> H, RLinear-style)
* cross-variable branch z_i(t) = sum_{j in N(i)} sum_{tau=0}^{L} K_ij(tau) x_j(t - tau)
                        y_cross = W_cross z_i               (shared Linear T -> H)

The per-pair lag kernel K_ij is parameterised by ``--dlt_variant``:

  ci         no cross branch (channel-independent baseline)
  scalar     K = a_ij at tau = 0                    (iTransformer / VCformer-style mixing)
  shift      K = a_ij at tau = d_ij (CCF peak lag)  (LIFT-style single lag)
  tap3       K free for tau = 0..2                  (CrossLinear-style short kernel)
  full       K free for tau = 0..L                  (unconstrained per-pair kernel)
  dlt        K = sum_r alpha_ijr phi_r(tau)         (shared lag basis, R << L)
  scalardyn  scalar with input-adaptive a_ij(X)     (control: dynamics without a lag kernel)
  dltdyn     dlt with input-adaptive alpha_ijr(X)   (proposed)

Neighbours N(i) (top-k) and peak lags d_ij are chosen from the *training split only*
(max |CCF| of first differences over lags 0..L). ``prepare(train_dataset)`` is called by
Exp_Long_Term_Forecast.train() before training. They are stored as buffers, so they follow
``.to(device)`` / DataParallel and are saved in (and restored from) checkpoints.

All tensors created in ``forward`` use the input's device: the model runs unchanged on
CPU, CUDA (single / multi-GPU via DataParallel, AMP) and MPS.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("ci", "scalar", "shift", "tap3", "dlt", "full", "scalardyn", "dltdyn")


@torch.no_grad()
def select_neighbours(train_values, k, L, max_len=8000, device="cpu"):
    """Top-k sources per target by max_tau |corr(dx_i(t), dx_j(t - tau))|, tau = 0..L.

    train_values: (T, C) array of the (scaled) training split.
    Returns idx (C, k) and lag (C, k) as int64 tensors on ``device``.
    """
    dev = torch.device(device)
    if dev.type == "mps":            # float64 is not supported on MPS
        dev = torch.device("cpu")
    x = torch.as_tensor(np.asarray(train_values), dtype=torch.float64, device=dev)
    x = torch.diff(x, dim=0)[-max_len:]
    x = (x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-8)
    n, C = x.shape
    best = torch.full((C, C), -1.0, dtype=torch.float64, device=dev)
    blag = torch.zeros((C, C), dtype=torch.long, device=dev)
    for tau in range(L + 1):
        c = (x[tau:].T @ x[:n - tau]).abs() / (n - tau)    # (C_target, C_source)
        upd = c > best
        best[upd] = c[upd]
        blag[upd] = tau
    best.fill_diagonal_(-float("inf"))
    idx = torch.argsort(best, dim=1, descending=True)[:, :k]
    lag = torch.gather(blag, 1, idx)
    return idx, lag


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name not in ("long_term_forecast", "short_term_forecast"):
            raise NotImplementedError("DLT currently supports long_term_forecast / short_term_forecast only")
        self.seq_len = T = configs.seq_len
        self.pred_len = H = configs.pred_len
        self.C = C = configs.enc_in
        self.variant = getattr(configs, "dlt_variant", "dltdyn")
        if self.variant not in VARIANTS:
            raise ValueError(f"--dlt_variant must be one of {VARIANTS}, got {self.variant}")
        self.L = L = getattr(configs, "dlt_L", 48)
        self.R = R = getattr(configs, "dlt_R", 4)
        self.k = k = max(0, min(getattr(configs, "dlt_k", 8), C - 1))
        self.dyn_dim = d = getattr(configs, "dlt_dyn_dim", 16)
        self.ccf_len = getattr(configs, "dlt_ccf_len", 8000)
        if k == 0 and self.variant != "ci":
            print(f"[DLT] enc_in={C}: no other variables, falling back to variant 'ci'")
            self.variant = "ci"

        self.self_head = nn.Linear(T, H)
        if self.variant == "ci":
            return
        # neighbour indices / peak lags: filled by prepare(), saved in checkpoints
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
            tau = torch.arange(L + 1).float()
            # basis initialised as delayed, decaying exponentials (onsets 0..0.6L, scales 1..L/4)
            delays = torch.linspace(0, L * 0.6, R)
            scales = torch.logspace(0, np.log10(max(L / 4, 2)), R)
            phi = torch.exp(-F.relu(tau[None] - delays[:, None]) / scales[:, None]) * (tau[None] >= delays[:, None])
            phi = phi / phi.norm(dim=1, keepdim=True)
            self.phi = nn.Parameter(phi + 0.01 * torch.randn_like(phi))
        if v in ("scalardyn", "dltdyn"):
            # alpha_ij(X) = alpha_ij * (1 + tanh(MLP([s_i || s_j]))), s = Linear_T->d(x); zero-init last layer
            self.summ = nn.Linear(T, d)
            self.mod = nn.Sequential(nn.Linear(2 * d, 32), nn.GELU(), nn.Linear(32, R if v == "dltdyn" else 1))
            nn.init.zeros_(self.mod[-1].weight)
            nn.init.zeros_(self.mod[-1].bias)

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def prepare(self, train_data):
        """Select neighbours / peak lags from the training split (called once before training)."""
        if self.variant == "ci":
            return
        values = getattr(train_data, "data_x", train_data)
        idx, lag = select_neighbours(values, self.k, self.L, self.ccf_len, device=self.idx.device)
        self.idx.copy_(idx)
        self.lag.copy_(lag)
        self.prepared.fill_(True)
        print(f"[DLT] variant={self.variant}: selected k={self.k} neighbours per variable "
              f"(peak lag 0 for {(lag == 0).float().mean().item():.0%} of pairs)")

    # ------------------------------------------------------------------ model
    def _pair_features(self, xn):
        s = self.summ(xn)                                                  # (B, C, d)
        B, C, d = s.shape
        return torch.cat([s.unsqueeze(2).expand(B, C, self.k, d), s[:, self.idx]], dim=-1)  # (B, C, k, 2d)

    def cross(self, xn):
        """xn: (B, C, T) normalised input -> z: (B, C, T)."""
        B, C, T = xn.shape
        k, L, idx, v = self.k, self.L, self.idx, self.variant
        if v == "scalar":
            return (xn[:, idx] * self.a[None, :, :, None]).sum(2)
        if v == "scalardyn":
            a = self.a[None] * (1 + torch.tanh(self.mod(self._pair_features(xn)).squeeze(-1)))   # (B, C, k)
            return (xn[:, idx] * a[..., None]).sum(2)
        if v == "shift":
            src = F.pad(xn, (L, 0))[:, idx]                                     # (B, C, k, T + L)
            pos = torch.arange(T, device=xn.device)[None, None] + L - self.lag[..., None]   # (C, k, T)
            g = torch.gather(src, 3, pos.unsqueeze(0).expand(B, -1, -1, -1))
            return (g * self.a[None, :, :, None]).sum(2)
        if v in ("dlt", "dltdyn"):
            # filter every variable once with the R shared basis kernels, then mix per pair
            xp = F.pad(xn, (L, 0)).reshape(B * C, 1, T + L)
            u = F.conv1d(xp, self.phi.flip(-1).unsqueeze(1)).view(B, C, self.R, T)
            ug = u[:, idx]                                                       # (B, C, k, R, T)
            if v == "dlt":
                return torch.einsum("bckrt,ckr->bct", ug, self.alpha)
            alpha = self.alpha[None] * (1 + torch.tanh(self.mod(self._pair_features(xn))))  # (B, C, k, R)
            return torch.einsum("bckrt,bckr->bct", ug, alpha)
        # tap3 / full: explicit per-pair kernels as one grouped convolution
        taps = self.w.shape[-1]
        xg = F.pad(xn[:, idx.reshape(-1)], (taps - 1, 0))                       # (B, C*k, T + taps - 1)
        z = F.conv1d(xg, self.w.reshape(C * k, 1, taps).flip(-1), groups=C * k)
        return z.view(B, C, k, T).sum(2)

    def forecast(self, x_enc):
        x = x_enc.permute(0, 2, 1)                                              # (B, C, T)
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True) + 1e-5
        xn = (x - mu) / sd
        y = self.self_head(xn)
        if self.variant != "ci":
            if not getattr(self, "_ready", False):      # checked once (avoids a device sync per step)
                if not bool(self.prepared):
                    raise RuntimeError("DLT: neighbours not selected; call model.prepare(train_data) "
                                       "before training, or load a trained checkpoint")
                self._ready = True
            y = y + self.cross_head(self.cross(xn))
        return (y * sd + mu).permute(0, 2, 1)                                   # (B, H, C)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)[:, -self.pred_len:, :]
