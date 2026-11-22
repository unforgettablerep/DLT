import numpy as np
import torch
import torch.nn.functional as F

VARIANTS = ("ci", "scalar", "shift", "tap3", "dlt", "full", "scalardyn", "dltdyn", "topk", "mill", "milldyn")
LAGSET_VARIANTS = ("topk", "mill", "milldyn")
GATES = ("none", "moe", "static")
NEIGHBOUR_MODES = ("ccf", "random", "all")
IMPLS = ("kernel", "basis", "dense")


def diff_standardise(train_values, max_len, device):
    """First differences of the last max_len training steps, standardised per variable (float64)."""
    dev = torch.device(device)
    if dev.type == "mps":
        dev = torch.device("cpu")
    x = torch.as_tensor(np.asarray(train_values), dtype=torch.float64, device=dev)
    x = torch.diff(x, dim=0)[-max_len:]
    return (x - x.mean(0)) / (x.std(0, unbiased=False) + 1e-8)


def abs_ccf(x, tau):
    # |corr(dx_i(t), dx_j(t - tau))|: [C_target, C_source]
    n = x.shape[0]
    return (x[tau:].T @ x[:n - tau]).abs() / (n - tau)


@torch.no_grad()
def select_neighbours(train_values, k, L, max_len=8000, device="cpu", mode="ccf", seed=0):
    """Top-k sources per target by max_tau |CCF| (tau = 0..L) on the training split; returns idx [C, k] and peak lag [C, k]."""
    x = diff_standardise(train_values, max_len, device)
    n, C = x.shape
    best = torch.full((C, C), -1.0, dtype=torch.float64, device=x.device)
    blag = torch.zeros((C, C), dtype=torch.long, device=x.device)
    for tau in range(L + 1):
        c = abs_ccf(x, tau)
        upd = c > best
        best[upd] = c[upd]
        blag[upd] = tau
    best.fill_diagonal_(-float("inf"))
    if mode == "random":
        gen = torch.Generator().manual_seed(int(seed))
        others = [[j for j in range(C) if j != i] for i in range(C)]
        idx = torch.tensor([[others[i][j] for j in torch.randperm(C - 1, generator=gen)[:k].tolist()] for i in range(C)], dtype=torch.long, device=x.device)
    else:
        idx = torch.argsort(best, dim=1, descending=True)[:, :k]
    lag = torch.gather(blag, 1, idx)
    return idx, lag


@torch.no_grad()
def select_lags(train_values, idx, K, L, max_len=8000, device="cpu"):
    """The K lags in 0..L with the largest |CCF| for every selected pair (MillGNN lag sets): [C, k, K], ascending."""
    x = diff_standardise(train_values, max_len, device)
    idx = idx.to(x.device)
    rows = torch.arange(x.shape[1], device=x.device)[:, None]
    r = torch.stack([abs_ccf(x, tau)[rows, idx] for tau in range(L + 1)], dim=-1)
    return r.topk(K, dim=-1).indices.sort(dim=-1).values


def lag_basis(L, R):
    """Initial lag basis: delayed, decaying exponentials (onsets 0..0.6 L, scales 1..L/4), unit norm, plus 0.01 noise: [R, L + 1]."""
    tau = torch.arange(L + 1).float()
    delays = torch.linspace(0, L * 0.6, R)
    scales = torch.logspace(0, np.log10(max(L / 4, 2)), R)
    phi = torch.exp(-F.relu(tau[None] - delays[:, None]) / scales[:, None]) * (tau[None] >= delays[:, None])
    phi = phi / phi.norm(dim=1, keepdim=True)
    return phi + 0.01 * torch.randn_like(phi)


def pair_conv(xn, idx, K):
    """z_i = sum_j K_ij * x_j for static per-pair kernels K [C, k, taps], as one grouped convolution; xn [B, C, T]."""
    B, C, T = xn.shape
    k, taps = K.shape[1], K.shape[-1]
    xg = F.pad(xn[:, idx.reshape(-1)], (taps - 1, 0))
    z = F.conv1d(xg, K.reshape(C * k, 1, taps).flip(-1), groups=C * k)
    return z.view(B, C, k, T).sum(2)


def lag_taps(xn, idx, tlag, w, L):
    """z_i(t) = sum_j sum_m w_ijm x_j(t - tau_ijm) on the selected lags tlag [C, k, K]; w [C, k, K] or [B, C, k, K]."""
    B, C, T = xn.shape
    src = F.pad(xn, (L, 0))[:, idx]
    base = torch.arange(T, device=xn.device) + L
    z = 0
    for m in range(tlag.shape[-1]):
        pos = (base - tlag[..., m:m + 1]).unsqueeze(0).expand(B, -1, -1, -1)
        z = z + (torch.gather(src, 3, pos) * w[..., m, None]).sum(2)
    return z
