"""LagTS (Liu et al., ICASSP 2025), re-implemented from the paper -- no official code is available.

    H^L   = iTransformer encoder(X)                       variate tokens, EmbeddingLayer = MLP   (Eq. 1-2)
    H_rel = Extractor([H_source, H_target])               MLP on all N^2 ordered pairs            (Eq. 3-6)
    Y_hat = projection([H^L, reshape(H_rel, N x N D')])   linear                                  (Sec. II-B)
    L     = MSE(Y, Y_hat) + lambda * L_relation                                                   (Eq. 7-10)

Lag relation loss (Sec. II-C): X, Y_hat and Y are instance-normalised (statistics of the look-back
window, as in LIFT); for every pair, the lag delta_ij = argmax_{1 <= tau <= S-1} R_ij(tau) of the FFT
cross-correlation of the *ground-truth* future (Eq. 7); the source prediction shifted by delta_ij
(history in front, Eq. 8) has to match the target prediction (Eq. 9, squared error).

Choices the paper leaves open (documented in DLT_CHANGES.md): EmbeddingLayer / Extractor are two-layer
MLPs with GELU; D' = --lagts_rel_dim (default 8); all N^2 ordered pairs incl. i = j, as the sums of Eq. 9
are written (for i = j the symmetric autocorrelation ties tau and S - tau; the smaller lag is used);
delta is clipped to the look-back length (Eq. 8 needs X before the window otherwise).
Memory and parameters grow with N^2 (the paper evaluates N <= 21), so use it on small datasets only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name not in ("long_term_forecast", "short_term_forecast"):
            raise NotImplementedError("LagTS supports forecasting only")
        self.seq_len = T = configs.seq_len
        self.pred_len = S = configs.pred_len
        self.N = N = configs.enc_in
        D = configs.d_model
        self.rel_dim = Dr = getattr(configs, "lagts_rel_dim", 8)
        self.lam = getattr(configs, "lagts_lambda", 0.1)
        if N > 64:
            print(f"[LagTS] warning: N = {N} variables -> {N * N} pairs; memory grows with N^2")
        self.embed = nn.Sequential(nn.Linear(T, D), nn.GELU(), nn.Linear(D, D), nn.Dropout(configs.dropout))
        self.encoder = Encoder(
            [EncoderLayer(AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), D, configs.n_heads),
                          D, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
             for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(D))
        self.extractor = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, Dr))
        self.projection = nn.Linear(D + N * Dr, S)

    def forecast(self, x_enc, x_mark_enc):
        mu = x_enc.mean(1, keepdim=True).detach()
        sd = torch.sqrt(torch.var(x_enc - mu, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = (x_enc - mu) / sd
        tokens = x.permute(0, 2, 1)                                          # (B, N, T)
        if x_mark_enc is not None:                                           # time marks as extra tokens
            tokens = torch.cat([tokens, x_mark_enc.permute(0, 2, 1)], 1)
        h, _ = self.encoder(self.embed(tokens), attn_mask=None)
        h = h[:, :self.N]                                                    # (B, N, D) variate tokens
        B, N, D = h.shape
        pair = torch.cat([h.unsqueeze(1).expand(B, N, N, D),                # [target j][source i]: h_i
                          h.unsqueeze(2).expand(B, N, N, D)], -1)            # h_j
        rel = self.extractor(pair).reshape(B, N, N * self.rel_dim)          # row j: relations to all sources
        y = self.projection(torch.cat([h, rel], -1)).permute(0, 2, 1)        # (B, S, N)
        return y * sd + mu

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc)[:, -self.pred_len:, :]

    def extra_loss(self, x_enc, pred, true):
        """lambda * L_relation (Eq. 9) for a batch; pred / true: (B, S, N) in the scaled data space."""
        if self.lam == 0:
            return pred.new_zeros(())
        mu = x_enc.mean(1, keepdim=True)
        sd = x_enc.std(1, keepdim=True) + 1e-5
        xh, yh, yt = ((v - mu) / sd for v in (x_enc, pred, true))
        B, S, N = yh.shape
        T = xh.shape[1]
        with torch.no_grad():                                                # lags from the ground truth
            f = torch.fft.rfft(yt.permute(0, 2, 1), n=S)                     # (B, N, S//2+1)
            r = torch.fft.irfft(f.unsqueeze(1) * torch.conj(f).unsqueeze(2), n=S) / S   # [b, i, j, tau]
            # for i = j the circular autocorrelation is symmetric, R(tau) = R(S - tau): take the smaller lag
            late = torch.arange(1, S, device=r.device) > S // 2
            diag = torch.eye(N, dtype=torch.bool, device=r.device)[..., None] & late
            delta = (r[..., 1:].masked_fill(diag, float("-inf")).argmax(-1) + 1).clamp(max=T)  # (B, N_i, N_j)
        z = torch.cat([xh, yh], 1).permute(0, 2, 1)                          # (B, N, T + S)
        pos = T + torch.arange(S, device=z.device) - delta.unsqueeze(-1)     # (B, N_i, N_j, S)
        shifted = torch.gather(z.unsqueeze(2).expand(B, N, N, T + S), 3, pos)  # source i shifted for target j
        target = yh.permute(0, 2, 1).unsqueeze(1)                            # (B, 1, N_j, S)
        return self.lam * F.mse_loss(shifted, target.expand_as(shifted))
