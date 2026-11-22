import torch
import torch.nn as nn

from src.layers.Embed import DataEmbedding_inverted
from src.layers.Mamba_EncDec import EncoderLayer, mamba_block


class Model(nn.Module):
    """S-Mamba (arXiv:2403.11144) after github.com/wzhwzhwzh0921/S-D-Mamba; unused blocks of the official layer are left out."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len, self.pred_len = configs.seq_len, configs.pred_len
        self.use_norm = getattr(configs, "use_norm", 1)
        on_gpu = bool(getattr(configs, "use_gpu", True)) and torch.cuda.is_available()
        Mamba, self.mamba_impl, note = mamba_block(getattr(configs, "mamba_impl", "auto"), on_gpu)
        print("[S-Mamba] Mamba block: {} ({})".format(self.mamba_impl, note))
        D = configs.d_model
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, D, configs.embed, configs.freq, configs.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(Mamba, D, configs.d_ff, getattr(configs, "d_state", 32), configs.dropout, configs.activation) for _ in range(configs.e_layers)]
        )
        self.norm = nn.LayerNorm(D)
        self.projector = nn.Linear(D, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        N = x_enc.shape[-1]
        h = self.enc_embedding(x_enc, x_mark_enc)
        # h: [B, N + marks, D]
        for layer in self.layers:
            h = layer(h)
        y = self.projector(self.norm(h)).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            y = y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc)[:, -self.pred_len:, :]
