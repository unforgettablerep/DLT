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
  topk       free weights on the K lags with the largest |CCF| per pair  (MillGNN-style lag sets)
  mill       a_ij * Lambda_ij(tau) on the same K lags, with MillGNN's excitatory / inhibitory decay
             Lambda_ij(tau) = ReLU(exp(-be_ij tau) - (1 - att_ij) exp(-bi_ij tau)), be, bi, att in (0, 1)
  milldyn    mill with input-dependent att_ij(X) = softmax_j(q_i . k_j / sqrt(d))  (MillGNN-style, dynamic)

``--dlt_gate`` combines the two branches (any variant except ci):

  none       y = y_self + y_cross                                   (default)
  moe        y = g_0 y_self + g_1 y_cross, g = softmax(W_g x_i)      (MoVE-style input-dependent gate)
  static     y = y_self + sigmoid(theta_i) y_cross, theta_i = --dlt_gate_init (per-variable gate, opens from ~0)

Neighbours N(i) (top-k), peak lags d_ij and the K-lag sets are chosen from the *training split only*
(|CCF| of first differences over lags 0..L). ``--dlt_neighbours`` replaces the CCF ranking by k random
sources (``random``, drawn with ``--fix_seed``) or by all other variables (``all``, k = C - 1); peak lags
still come from the CCF. ``--dlt_impl dense`` evaluates dlt / dltdyn with a dense C x C coefficient
tensor (cheapest when k is close to C - 1).

With ``cross_only=True`` the model has no intra-variable head and ``correction(x_enc)`` returns only the
de-normalised cross-variable term sigma_i * W_c z_i; ``models/DLTPlug.py`` adds it to any backbone. ``prepare(train_dataset)`` is called by
Exp_Long_Term_Forecast.train() before training. They are stored as buffers, so they follow
``.to(device)`` / DataParallel and are saved in (and restored from) checkpoints.

All tensors created in ``forward`` use the input's device: the model runs unchanged on
CPU, CUDA (single / multi-GPU via DataParallel, AMP) and MPS.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("ci", "scalar", "shift", "tap3", "dlt", "full", "scalardyn", "dltdyn", "topk", "mill", "milldyn")
LAGSET_VARIANTS = ("topk", "mill", "milldyn")      # use K CCF-selected lags per pair
GATES = ("none", "moe", "static")
NEIGHBOUR_MODES = ("ccf", "random", "all")


def _diff_standardise(train_values, max_len, device):
    """First differences of the last ``max_len`` training steps, standardised per variable (float64)."""
    dev = torch.device(device)
    if dev.type == "mps":            # float64 is not supported on MPS
        dev = torch.device("cpu")
    x = torch.as_tensor(np.asarray(train_values), dtype=torch.float64, device=dev)
    x = torch.diff(x, dim=0)[-max_len:]
    return (x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-8)


def _abs_ccf(x, tau):
    """|corr(dx_i(t), dx_j(t - tau))| for all pairs: (C_target, C_source)."""
    n = x.shape[0]
    return (x[tau:].T @ x[:n - tau]).abs() / (n - tau)


@torch.no_grad()
def select_neighbours(train_values, k, L, max_len=8000, device="cpu", mode="ccf", seed=0):
    """Top-k sources per target by max_tau |corr(dx_i(t), dx_j(t - tau))|, tau = 0..L.

    train_values: (T, C) array of the (scaled) training split.
    mode: "ccf" (default), "random" (k sources drawn uniformly without the target itself, generator
    seeded with ``seed``) or "all" (all other variables, ordered by CCF strength).
    Returns idx (C, k) and lag (C, k) as int64 tensors on ``device``; lag is the CCF peak lag of each pair.
    """
    x = _diff_standardise(train_values, max_len, device)
    n, C = x.shape
    best = torch.full((C, C), -1.0, dtype=torch.float64, device=x.device)
    blag = torch.zeros((C, C), dtype=torch.long, device=x.device)
    for tau in range(L + 1):
        c = _abs_ccf(x, tau)                                  # (C_target, C_source)
        upd = c > best
        best[upd] = c[upd]
        blag[upd] = tau
    best.fill_diagonal_(-float("inf"))
    if mode == "random":
        gen = torch.Generator().manual_seed(int(seed))
        others = [[j for j in range(C) if j != i] for i in range(C)]
        idx = torch.tensor([[others[i][j] for j in torch.randperm(C - 1, generator=gen)[:k].tolist()]
                            for i in range(C)], dtype=torch.long, device=x.device)
    else:                            # "ccf"; "all" is the same ranking with k = C - 1
        idx = torch.argsort(best, dim=1, descending=True)[:, :k]
    lag = torch.gather(blag, 1, idx)
    return idx, lag


@torch.no_grad()
def select_lags(train_values, idx, K, L, max_len=8000, device="cpu"):
    """For every selected pair (i, idx[i, j]): the K lags in 0..L with the largest |CCF|
    (the lag sets of MillGNN's cross-correlation graph). Returns (C, k, K) int64, ascending."""
    x = _diff_standardise(train_values, max_len, device)
    idx = idx.to(x.device)
    rows = torch.arange(x.shape[1], device=x.device)[:, None]
    r = torch.stack([_abs_ccf(x, tau)[rows, idx] for tau in range(L + 1)], dim=-1)   # (C, k, L + 1)
    return r.topk(K, dim=-1).indices.sort(dim=-1).values


class Model(nn.Module):
    def __init__(self, configs, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
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
        self.neighbours = getattr(configs, "dlt_neighbours", "ccf")
        if self.neighbours not in NEIGHBOUR_MODES:
            raise ValueError(f"--dlt_neighbours must be one of {NEIGHBOUR_MODES}, got {self.neighbours}")
        self.k = k = C - 1 if self.neighbours == "all" else max(0, min(getattr(configs, "dlt_k", 8), C - 1))
        self.neighbour_seed = getattr(configs, "fix_seed", 2021)
        self.dyn_dim = d = getattr(configs, "dlt_dyn_dim", 16)
        self.ccf_len = getattr(configs, "dlt_ccf_len", 8000)
        self.K = K = getattr(configs, "dlt_topk", 5)              # lags per pair for topk / mill / milldyn
        self.gate = getattr(configs, "dlt_gate", "none")
        if self.gate not in GATES:
            raise ValueError(f"--dlt_gate must be one of {GATES}, got {self.gate}")
        if self.variant in LAGSET_VARIANTS and not 1 <= K <= L + 1:
            raise ValueError(f"--dlt_topk must be in 1..dlt_L+1 = {L + 1}, got {K}")
        # dlt / dltdyn computation path (mathematically identical):
        #   kernel: build K_ij = sum_r alpha_ijr phi_r, one grouped conv  (less memory; default)
        #   basis : filter with the R basis kernels, gather, mix          (fewer FLOPs for R << L)
        self.impl = getattr(configs, "dlt_impl", "kernel")
        if self.impl not in ("kernel", "basis", "dense"):
            raise ValueError(f"--dlt_impl must be 'kernel', 'basis' or 'dense', got {self.impl}")
        if k == 0 and self.variant != "ci":
            print(f"[DLT] enc_in={C}: no other variables, falling back to variant 'ci'")
            self.variant = "ci"

        if cross_only and self.variant == "ci":
            raise ValueError("cross_only=True needs a cross-variable variant (not 'ci')")
        if not cross_only:
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
        if v in LAGSET_VARIANTS:
            # K lags per pair with the largest training-split |CCF|: filled by prepare()
            self.register_buffer("tlag", torch.zeros(C, k, K, dtype=torch.long))
            if v == "topk":
                self.w = g(C, k, K)                                   # one free weight per selected lag
            else:
                # MillGNN decay-aware weights on the selected lags (time-step resolution, time-invariant):
                #   K_ij(tau) = a_ij * Lambda_ij(tau),  Lambda = ReLU(exp(-be tau) - (1 - att) exp(-bi tau))
                # be, bi in (0, 1) per pair (MillGNN: softmax of embeddings, shared across samples);
                # a_ij is a signed per-pair amplitude, added so that negative coupling is representable
                # as in every other variant (MillGNN's ReLU weights are non-negative).
                logit = lambda p: float(np.log(p / (1 - p)))
                self.a = g(C, k)
                self.b_exc = nn.Parameter(logit(0.05) + 0.01 * torch.randn(C, k))   # slow excitatory decay
                self.b_inh = nn.Parameter(logit(0.2) + 0.01 * torch.randn(C, k))    # faster inhibitory decay
                if v == "mill":
                    self.b_att = nn.Parameter(0.01 * torch.randn(C, k))            # att = sigmoid -> 0.5
                else:
                    # att_ij(X) = softmax over the k sources of q_i . k_j / sqrt(d)   (MillGNN Eq. 8)
                    self.summ = nn.Linear(T, d)
                    self.q_proj = nn.Linear(d, d)
                    self.k_proj = nn.Linear(d, d)
        if self.gate == "moe":
            self.gate_net = nn.Linear(T, 2)                           # MoVE: G(X) = softmax(W_g X)
        elif self.gate == "static":
            self.gate_logit = nn.Parameter(torch.full((C,), float(getattr(configs, "dlt_gate_init", -3.0))))

    # ------------------------------------------------------------------ setup
    @torch.no_grad()
    def prepare(self, train_data):
        """Select neighbours / peak lags from the training split (called once before training)."""
        if self.variant == "ci":
            return
        values = getattr(train_data, "data_x", train_data)
        idx, lag = select_neighbours(values, self.k, self.L, self.ccf_len, device=self.idx.device,
                                     mode=self.neighbours, seed=self.neighbour_seed)
        self.idx.copy_(idx)
        self.lag.copy_(lag)
        msg = ""
        if self.variant in LAGSET_VARIANTS:
            tl = select_lags(values, idx, self.K, self.L, self.ccf_len, device=self.idx.device)
            self.tlag.copy_(tl)
            msg = f"; K={self.K} lags per pair, lag 0 among them for {(tl == 0).any(-1).float().mean().item():.0%}"
        self.prepared.fill_(True)
        print(f"[DLT] variant={self.variant}: selected k={self.k} neighbours per variable ({self.neighbours}) "
              f"(peak lag 0 for {(lag == 0).float().mean().item():.0%} of pairs){msg}")

    # ------------------------------------------------------------------ model
    def _pair_features(self, xn):
        s = self.summ(xn)                                                  # (B, C, d)
        B, C, d = s.shape
        return torch.cat([s.unsqueeze(2).expand(B, C, self.k, d), s[:, self.idx]], dim=-1)  # (B, C, k, 2d)

    def cross(self, xn, mod_input=None):
        """xn: (B, C, T) normalised input -> z: (B, C, T). The input-dependent coefficients (dltdyn, scalardyn,
        milldyn) are computed from mod_input (default: xn), so that the kernels can also run over a longer sequence
        than the look-back window (DLTPlug --dlt_fusion refine: past and forecast), with the coefficients of the
        window."""
        B, C, T = xn.shape
        k, L, idx, v = self.k, self.L, self.idx, self.variant
        m = xn if mod_input is None else mod_input
        if v == "scalar":
            return (xn[:, idx] * self.a[None, :, :, None]).sum(2)
        if v == "scalardyn":
            a = self.a[None] * (1 + torch.tanh(self.mod(self._pair_features(m)).squeeze(-1)))    # (B, C, k)
            return (xn[:, idx] * a[..., None]).sum(2)
        if v == "shift":
            src = F.pad(xn, (L, 0))[:, idx]                                     # (B, C, k, T + L)
            pos = torch.arange(T, device=xn.device)[None, None] + L - self.lag[..., None]   # (C, k, T)
            g = torch.gather(src, 3, pos.unsqueeze(0).expand(B, -1, -1, -1))
            return (g * self.a[None, :, :, None]).sum(2)
        if v in ("dlt", "dltdyn"):
            alpha = self.alpha
            if v == "dltdyn":
                alpha = alpha[None] * (1 + torch.tanh(self.mod(self._pair_features(m))))   # (B, C, k, R)
            if self.impl == "dense":
                # basis-filtered signals of all variables, mixed with a dense C x C x R coefficient tensor
                xp = F.pad(xn, (L, 0)).reshape(B * C, 1, T + L)
                u = F.conv1d(xp, self.phi.flip(-1).unsqueeze(1)).view(B, C, self.R, T)
                if v == "dlt":
                    A = alpha.new_zeros(C, C, self.R).scatter(1, idx[..., None].expand(C, k, self.R), alpha)
                    return torch.einsum("ijr,bjrt->bit", A, u)
                A = alpha.new_zeros(B, C, C, self.R).scatter(2, idx[None, ..., None].expand(B, C, k, self.R), alpha)
                return torch.einsum("bijr,bjrt->bit", A, u)
            if self.impl == "basis":
                # filter every variable once with the R shared basis kernels, then mix per pair
                xp = F.pad(xn, (L, 0)).reshape(B * C, 1, T + L)
                ug = F.conv1d(xp, self.phi.flip(-1).unsqueeze(1)).view(B, C, self.R, T)[:, idx]  # (B,C,k,R,T)
                eq = "bckrt,ckr->bct" if v == "dlt" else "bckrt,bckr->bct"
                return torch.einsum(eq, ug, alpha)
            # "kernel": materialise K_ij = sum_r alpha_ijr phi_r once, then one grouped convolution
            if v == "dlt":
                return self._pair_conv(xn, torch.einsum("ckr,rl->ckl", alpha, self.phi))       # K: (C, k, L+1)
            K = torch.einsum("bckr,rl->bckl", alpha, self.phi)                                    # (B, C, k, L+1)
            xg = F.pad(xn[:, idx], (L, 0)).reshape(1, B * C * k, T + L)
            z = F.conv1d(xg, K.reshape(B * C * k, 1, L + 1).flip(-1), groups=B * C * k)
            return z.view(B, C, k, T).sum(2)
        if v == "topk":
            return self._lag_taps(xn, self.w)
        if v in ("mill", "milldyn"):
            return self._lag_taps(xn, self.mill_weights(m))
        # tap3 / full: explicit per-pair kernels
        return self._pair_conv(xn, self.w)

    def mill_weights(self, xn=None):
        """Tap weights a_ij * Lambda_ij(tau) at the selected lags: (C, k, K), or (B, C, k, K) for milldyn."""
        if self.variant == "mill":
            att = torch.sigmoid(self.b_att)                                          # (C, k)
        else:
            s = self.summ(xn)                                                        # (B, C, d)
            q, kk = self.q_proj(s), self.k_proj(s)
            score = (q.unsqueeze(2) * kk[:, self.idx]).sum(-1) / np.sqrt(q.shape[-1])  # (B, C, k)
            att = torch.softmax(score, dim=-1)
        tau = self.tlag.to(self.a.dtype)                                             # (C, k, K)
        lam = F.relu(torch.exp(-torch.sigmoid(self.b_exc)[..., None] * tau)
                     - (1 - att[..., None]) * torch.exp(-torch.sigmoid(self.b_inh)[..., None] * tau))
        return self.a[..., None] * lam

    def _lag_taps(self, xn, w):
        """z_i(t) = sum_j sum_m w_ijm x_j(t - tau_ijm) on the K selected lags; w: (C, k, K) or (B, C, k, K)."""
        B, C, T = xn.shape
        L = self.L
        src = F.pad(xn, (L, 0))[:, self.idx]                                         # (B, C, k, T + L)
        base = torch.arange(T, device=xn.device) + L
        z = 0
        for m in range(self.K):
            pos = (base - self.tlag[..., m:m + 1]).unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, k, T)
            z = z + (torch.gather(src, 3, pos) * w[..., m, None]).sum(2)
        return z

    def _pair_conv(self, xn, K):
        """z_i = sum_j K_ij * x_j for static per-pair kernels K: (C, k, taps), as one grouped convolution."""
        B, C, T = xn.shape
        k, taps = K.shape[1], K.shape[-1]
        xg = F.pad(xn[:, self.idx.reshape(-1)], (taps - 1, 0))                   # (B, C*k, T + taps - 1)
        z = F.conv1d(xg, K.reshape(C * k, 1, taps).flip(-1), groups=C * k)
        return z.view(B, C, k, T).sum(2)

    def _check_ready(self):
        if not getattr(self, "_ready", False):          # checked once (avoids a device sync per step)
            if not bool(self.prepared):
                raise RuntimeError("DLT: neighbours not selected; call model.prepare(train_data) "
                                   "before training, or load a trained checkpoint")
            self._ready = True

    def correction(self, x_enc):
        """cross_only models: the de-normalised cross-variable term sigma_i * gate * W_c z_i, (B, H, C)."""
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
        x = x_enc.permute(0, 2, 1)                                              # (B, C, T)
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
                gw = torch.softmax(self.gate_net(xn), dim=-1)                       # (B, C, 2)
                y = gw[..., :1] * y + gw[..., 1:] * y_cross
            else:
                y = y + torch.sigmoid(self.gate_logit)[:, None] * y_cross
        return (y * sd + mu).permute(0, 2, 1)                                   # (B, H, C)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)[:, -self.pred_len:, :]
