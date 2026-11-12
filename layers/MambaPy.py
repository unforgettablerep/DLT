"""Mamba block in pure PyTorch, a drop-in replacement of ``mamba_ssm.Mamba`` where that package is not installed.

``mamba_ssm`` needs CUDA and a compiled extension. This block has the same constructor, the same parameters
(names, shapes and initialisation of ``mamba_ssm.modules.mamba_simple.Mamba``: a state dict moves between the
two) and computes the same function; the selective scan is a Python loop over the sequence, so it is slower and
meant for CPU checks and for machines without ``mamba-ssm``.

    x, z   = split(in_proj(h))                         h: (B, L, d_model)
    x      = SiLU(causal depthwise conv(x))
    dt, B, C = split(x_proj(x));   dt = softplus(dt_proj(dt))
    s_t    = exp(dt_t A) s_{t-1} + dt_t B_t x_t        zero-order hold for A, Euler for B (as in mamba_ssm)
    y_t    = <s_t, C_t> + D x_t
    out    = out_proj(y * SiLU(z))
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt_proj: softplus(bias) starts log-uniform in [dt_min, dt_max], as in mamba_ssm
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(dt_init)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
                       ).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # inverse of softplus

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))                              # A = -exp(A_log) = -(1 .. d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):
        B, L, _ = hidden_states.shape
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)                  # (B, L, d_inner) each
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                    # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                                   # (d_inner, d_state)

        state = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):                                                   # selective scan
            dt_t = dt[:, t, :, None]                                         # (B, d_inner, 1)
            state = torch.exp(dt_t * A) * state + dt_t * Bm[:, t, None, :] * x[:, t, :, None]
            ys.append((state * Cm[:, t, None, :]).sum(-1))                   # (B, d_inner)
        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))
