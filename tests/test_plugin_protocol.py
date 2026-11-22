import contextlib
import io
import math
import shlex
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))

from mamba_simple_block import MambaBlock  # noqa: E402
from src.configs import baseline_profiles  # noqa: E402
from src.configs.dlt_profiles import DATASET_META, emit_cli_args, get_pred_lens, protocol  # noqa: E402
from src.layers import DLT_layers  # noqa: E402
from src.layers.MambaPy import Mamba  # noqa: E402
from src.model.DLinear import Model as DLinear  # noqa: E402
from src.model.DLT import Model as DLT  # noqa: E402
from src.model.DLTPlug import Model as DLTPlug  # noqa: E402
from src.model.iTransformer import Model as ITransformer  # noqa: E402
from src.model.RLinear import Model as RLinear  # noqa: E402
from src.model.S_Mamba import Model as SMamba  # noqa: E402
from src.model.TiDE import Model as TiDE  # noqa: E402


def cfg(**kw):
    c = dict(
        seq_len=96,
        label_len=48,
        pred_len=24,
        enc_in=9,
        dec_in=9,
        c_out=9,
        dlt_variant="dltdyn",
        dlt_k=4,
        dlt_L=10,
        dlt_R=3,
        dlt_dyn_dim=8,
        dlt_ccf_len=8000,
        dlt_impl="kernel",
        dlt_topk=3,
        dlt_gate="none",
        dlt_neighbours="ccf",
        seed=2021,
        moving_avg=25,
        individual=False,
        d_model=16,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        factor=1,
        dropout=0.0,
        activation="gelu",
        embed="timeF",
        freq="h",
        plug_base="DLinear",
    )
    c.update(kw)
    return types.SimpleNamespace(**c)


def reference_z(m, xn):
    """Explicit z_i(t) = sum_j sum_tau K_ij(tau) x_j(t - tau) with K_ij = sum_r alpha_ijr phi_r."""
    B, C, T = xn.shape
    al = m.alpha[None].expand(B, -1, -1, -1)
    if m.variant == "dltdyn":
        al = m.alpha[None] * (1 + torch.tanh(m.mod(m._pair_features(xn))))
    K = torch.einsum("bckr,rl->bckl", al, m.phi)
    z = torch.zeros(B, C, T)
    for i in range(C):
        for jj in range(m.k):
            j = m.idx[i, jj]
            for tau in range(m.L + 1):
                z[:, i, tau:] += K[:, i, jj, tau, None] * xn[:, j, :T - tau]
    return z


class Float64Case(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        cls.Xtr = torch.randn(3000, 9).cumsum(0).numpy()
        cls.xn = torch.randn(3, 9, 96)
        cls.x_enc = torch.randn(4, 96, 9)
        cls.mark = torch.randn(4, 96, 4)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.dtype)


class NeighbourAndDensePathTests(Float64Case):
    def test_dense_path_matches_reference(self):
        for nb in ("ccf", "all"):
            for v in ("dlt", "dltdyn"):
                torch.manual_seed(1)
                m = DLT(cfg(dlt_variant=v, dlt_impl="dense", dlt_neighbours=nb))
                m.prepare(self.Xtr)
                with torch.no_grad():
                    for p in m.parameters():
                        p.add_(0.3 * torch.randn_like(p))
                    err = (m.cross(self.xn) - reference_z(m, self.xn)).abs().max().item()
                self.assertLess(err, 1e-10, "{} {}".format(v, nb))

    def test_neighbour_sets(self):
        torch.manual_seed(2)
        for nb in ("random", "all"):
            for seed in (2021, 2022):
                m = DLT(cfg(dlt_variant="dlt", dlt_neighbours=nb, seed=seed))
                m.prepare(self.Xtr)
                idx = m.idx
                self.assertTrue(all(i not in idx[i].tolist() and len(set(idx[i].tolist())) == m.k for i in range(9)))
                m2 = DLT(cfg(dlt_variant="dlt", dlt_neighbours=nb, seed=seed))
                m2.prepare(self.Xtr)
                self.assertTrue(torch.equal(idx, m2.idx))
        r1 = DLT(cfg(dlt_variant="dlt", dlt_neighbours="random", seed=2021))
        r1.prepare(self.Xtr)
        r2 = DLT(cfg(dlt_variant="dlt", dlt_neighbours="random", seed=2022))
        r2.prepare(self.Xtr)
        c1 = DLT(cfg(dlt_variant="dlt"))
        c1.prepare(self.Xtr)
        self.assertFalse(torch.equal(r1.idx, r2.idx))
        self.assertFalse(torch.equal(r1.idx, c1.idx))
        idx_all, lag_all = DLT_layers.select_neighbours(self.Xtr, 8, 10, mode="all")
        for i in range(9):
            for j in range(4):
                self.assertEqual(int(r1.lag[i, j]), int(lag_all[i][idx_all[i].tolist().index(int(r1.idx[i, j]))]))


class PlugInTests(Float64Case):
    def test_output_fusion_and_alone(self):
        torch.manual_seed(3)
        plug = DLTPlug(cfg(dlt_variant="dltdyn"))
        plug.prepare(self.Xtr)
        with torch.no_grad():
            for p in plug.dlt.parameters():
                p.add_(0.3 * torch.randn_like(p))
            x = self.x_enc.permute(0, 2, 1)
            mu = x.mean(-1, keepdim=True)
            sd = x.std(-1, keepdim=True) + 1e-5
            ref = plug.base(self.x_enc, self.mark, None, None) + (plug.dlt.cross_head(plug.dlt.cross((x - mu) / sd)) * sd).permute(0, 2, 1)
            self.assertTrue(torch.allclose(plug(self.x_enc, self.mark, None, None), ref, atol=1e-12))
        torch.manual_seed(4)
        plug_ci = DLTPlug(cfg(dlt_variant="ci"))
        torch.manual_seed(4)
        base = DLinear(cfg())
        self.assertTrue(torch.equal(plug_ci(self.x_enc, self.mark, None, None), base(self.x_enc, self.mark, None, None)))
        self.assertFalse(hasattr(plug.dlt, "self_head"))

    def test_refine_and_input_start_as_backbone(self):
        torch.manual_seed(4)
        base0 = DLinear(cfg())
        with torch.no_grad():
            yb0 = base0(self.x_enc, self.mark, None, None)
        for fu in ("refine", "input"):
            torch.manual_seed(4)
            pf = DLTPlug(cfg(dlt_variant="dltdyn", dlt_fusion=fu))
            pf.prepare(self.Xtr)
            pf.base.load_state_dict(base0.state_dict())
            with torch.no_grad():
                self.assertTrue(torch.allclose(pf(self.x_enc, self.mark, None, None), yb0, atol=1e-12), fu)
            self.assertFalse(hasattr(pf.dlt, "cross_head"), fu)

    def test_refine_is_explicit_causal_sum(self):
        torch.manual_seed(5)
        pr = DLTPlug(cfg(dlt_variant="dlt", dlt_fusion="refine", dlt_impl="kernel"))
        pr.prepare(self.Xtr)
        with torch.no_grad():
            pr.dlt.alpha.normal_(0, 0.3)
            yb = pr._backbone(self.x_enc, self.mark, None, None)
            c = pr.refine(self.x_enc, yb)
            xx = self.x_enc.permute(0, 2, 1)
            mu = xx.mean(-1, keepdim=True)
            sn = torch.cat([xx - mu, yb.permute(0, 2, 1) - mu], -1)
            ref = reference_z(pr.dlt, sn)[..., -24:].permute(0, 2, 1)
            self.assertLess((c - ref).abs().max().item(), 1e-10)
            yb2 = yb.clone()
            yb2[:, 10:, :] += 1.0
            moved = (pr.refine(self.x_enc, yb2) - c).abs().amax((0, 2)).gt(1e-12).nonzero().flatten().tolist()
            self.assertEqual(moved, list(range(10, 24)))
            xf = self.x_enc.clone()
            xf[..., 0] = 0.3
            cf = pr.refine(xf, pr._backbone(xf, self.mark, None, None) + 5.0)
            self.assertTrue(bool(torch.isfinite(cf).all()) and cf.abs().max().item() < 100)

    def test_decoupled_training_gradients(self):
        yt = torch.randn(4, 24, 9)
        for fu in ("output", "refine"):
            torch.manual_seed(6)
            pd_ = DLTPlug(cfg(dlt_variant="dltdyn", dlt_fusion=fu, dlt_train="decoupled"))
            pd_.prepare(self.Xtr)
            torch.manual_seed(6)
            b_alone = DLinear(cfg())
            b_alone.load_state_dict(pd_.base.state_dict())
            with torch.no_grad():
                for q in pd_.dlt.parameters():
                    q.add_(0.05 * torch.randn_like(q))
            pd_.train()
            out = pd_(self.x_enc, self.mark, None, None)
            loss = torch.nn.functional.mse_loss(out, yt) + pd_.extra_loss(self.x_enc, out, yt)
            loss.backward()
            torch.nn.functional.mse_loss(b_alone(self.x_enc, self.mark, None, None), yt).backward()
            gb = max((p1.grad - p2.grad).abs().max().item() for p1, p2 in zip(pd_.base.parameters(), b_alone.parameters()))
            gd = sum(q.grad.abs().sum().item() for q in pd_.dlt.parameters() if q.grad is not None)
            self.assertLess(gb, 1e-12, fu)
            self.assertGreater(gd, 0, fu)

    def test_variable_selection_mask(self):
        torch.manual_seed(7)
        ps_ = DLTPlug(cfg(dlt_variant="dltdyn"))
        ps_.prepare(self.Xtr)
        ps_.eval()
        with torch.no_grad():
            yb_, c_ = ps_.parts(self.x_enc, self.mark, None, None)
            ps_.select_mask = torch.tensor([True, False] * 4 + [True])
            y_sel = ps_(self.x_enc, self.mark, None, None)
        self.assertTrue(torch.allclose(y_sel, yb_ + c_ * ps_.select_mask.to(c_.dtype), atol=1e-12))
        self.assertTrue(torch.allclose(y_sel[..., 1], yb_[..., 1], atol=1e-12))


class BackboneTests(Float64Case):
    def test_mamba_block_matches_reference(self):
        torch.manual_seed(5)
        worst = 0.0
        for d_model, d_state, d_conv, expand, Lq in ((32, 16, 2, 1, 11), (48, 2, 2, 1, 25), (64, 32, 4, 2, 9)):
            blk = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            ref = MambaBlock(types.SimpleNamespace(d_model=d_model, d_ff=d_state, d_conv=d_conv, expand=expand), d_inner=d_model * expand, dt_rank=math.ceil(d_model / 16))
            ref.load_state_dict(blk.state_dict(), strict=True)
            hq = torch.randn(3, Lq, d_model)
            with torch.no_grad():
                worst = max(worst, (blk(hq) - ref(hq)).abs().max().item())
        self.assertLess(worst, 1e-10)

    def test_mamba_block_causal_and_initialised(self):
        blk = Mamba(32, d_state=8, d_conv=2, expand=1)
        hq = torch.randn(2, 12, 32)
        h2 = hq.clone()
        h2[:, 7:] += 1.0
        with torch.no_grad():
            moved = (blk(hq) - blk(h2)).abs().amax(-1)[0].nonzero().flatten().tolist()
        self.assertEqual(moved, list(range(7, 12)))
        sp = torch.nn.functional.softplus(blk.dt_proj.bias)
        self.assertTrue(bool(sp.min() >= 1e-3 - 1e-9 and sp.max() <= 0.1 + 1e-9))
        self.assertTrue(torch.allclose(-torch.exp(blk.A_log[0]).double(), -torch.arange(1.0, 9.0).double(), atol=1e-6))
        self.assertTrue(bool((blk.D == 1).all()))

    def test_s_mamba(self):
        c = cfg(enc_in=6, dec_in=6, c_out=6, d_model=32, d_ff=32, e_layers=2, d_state=4, mamba_impl="torch", use_norm=1)
        with contextlib.redirect_stdout(io.StringIO()):
            sm = SMamba(c).eval()
        xq, mq = torch.randn(2, 96, 6), torch.randn(2, 96, 4)
        with torch.no_grad():
            yq = sm(xq, mq, None, None)
            x3 = xq.clone()
            x3[:, :, 3] += torch.randn(2, 96)
            dy = (sm(x3, mq, None, None) - yq).abs().amax((0, 1))
            ys = sm(xq * 3.0 + 5.0, mq, None, None)
        self.assertEqual(tuple(yq.shape), (2, 24, 6))
        self.assertTrue(bool((dy > 1e-9).all()))
        self.assertLess((ys - (yq * 3.0 + 5.0)).abs().max().item(), 1e-4)

    def test_rlinear_written_out(self):
        cr = cfg(enc_in=5, seq_len=16, pred_len=8, dropout=0.0, use_norm=1)
        torch.manual_seed(3)
        rl = RLinear(cr).eval()
        with torch.no_grad():
            rl.rev.affine_weight.uniform_(0.5, 1.5)
            rl.rev.affine_bias.uniform_(-0.3, 0.3)
        xr = torch.randn(3, 16, 5) * 2 + 1
        with torch.no_grad():
            yr = rl(xr, None, None, None)
            mu = xr.mean(1, keepdim=True)
            sd = torch.sqrt(xr.var(1, keepdim=True, unbiased=False) + 1e-5)
            w, b = rl.rev.affine_weight, rl.rev.affine_bias
            z = ((xr - mu) / sd) * w + b
            p = rl.Linear(z.transpose(1, 2)).transpose(1, 2)
            ref = (p - b) / (w + 1e-10) * sd + mu
            err2 = (rl(xr * 3.0 + 5.0, None, None, None) - (yr * 3.0 + 5.0)).abs().max().item()
        self.assertLess((yr - ref).abs().max().item(), 1e-10)
        self.assertLess(err2, 1e-4)

    def test_tide_checkpoint_identical(self):
        ct = cfg(enc_in=12, dec_in=12, c_out=8, seq_len=96, pred_len=24, d_model=32, d_ff=16, e_layers=2, d_layers=2, dropout=0.3, freq="h", embed="timeF")
        torch.manual_seed(1)
        t0 = TiDE(ct).train()
        ct.tide_checkpoint = True
        torch.manual_seed(1)
        t1 = TiDE(ct).train()
        xt, ytt = torch.randn(4, 96, 12), torch.randn(4, 24, 12)
        torch.manual_seed(7)
        o0 = t0(xt, None, None, None)
        ((o0 - ytt) ** 2).mean().backward()
        torch.manual_seed(7)
        o1 = t1(xt, None, None, None)
        ((o1 - ytt) ** 2).mean().backward()
        gdiff = max((p.grad - q.grad).abs().max().item() for p, q in zip(t0.parameters(), t1.parameters()))
        self.assertTrue(torch.equal(o0, o1))
        self.assertEqual(gdiff, 0.0)

    def test_itransformer_use_norm(self):
        ci = cfg(enc_in=6, d_model=16, d_ff=16, e_layers=1, n_heads=2)
        torch.manual_seed(0)
        it1 = ITransformer(ci).eval()
        ci.use_norm = 0
        torch.manual_seed(0)
        it0 = ITransformer(ci).eval()
        xq, mq = torch.randn(2, 96, 6), torch.randn(2, 96, 4)
        with torch.no_grad():
            y1, y0 = it1(xq, mq, None, None), it0(xq, mq, None, None)
            err = (it1(xq * 3.0 + 5.0, mq, None, None) - (y1 * 3.0 + 5.0)).abs().max().item()
        self.assertLess(err, 1e-4)
        self.assertGreater((y1 - y0).abs().max().item(), 1e-3)


class BaselineSettingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = {(m, ds, H): baseline_profiles.args(m, ds, H) for m in baseline_profiles.BASELINES for ds in DATASET_META for H in get_pred_lens(ds)}

    def a(self, m, ds, H):
        return self.A[(m, ds, H)][0]

    def test_every_setting_exists(self):
        self.assertEqual(len(self.A), 6 * 52)

    def test_official_values(self):
        self.assertIn("--d_model 512", self.a("iTransformer", "ETTh1", 336))
        self.assertIn("--train_epochs 1", self.a("iTransformer", "Exchange", 336))
        self.assertIn("--d_state 2", self.a("S_Mamba", "ETTh1", 96))
        self.assertIn("--learning_rate 0.00007", self.a("S_Mamba", "ETTh1", 96))
        self.assertTrue(all("--c_out 8" in self.a("TiDE", ds, 96) for ds in ("ETTh1", "ECL", "Weather")))
        self.assertIn("--batch_size 256", self.a("RLinear", "ETTh2", 96))
        self.assertIn("--train_epochs 15", self.a("RLinear", "ETTh2", 96))
        self.assertIn("--batch_size 128", self.a("PatchTST", "ETTm1", 192))

    def test_documented_fallbacks(self):
        self.assertIn("ECL configuration", self.A[("PatchTST", "PEMS08", 12)][1])
        self.assertIn("ETTh1 configuration", self.A[("TiDE", "Weather", 96)][1])
        self.assertIn("ETTh1 configuration", self.A[("RLinear", "Exchange", 96)][1])
        self.assertIn("ECL configuration", self.A[("RLinear", "PEMS04", 12)][1])
        self.assertEqual(self.a("DLinear", "Weather", 96), "")
        self.assertNotIn("c_out", self.a("iTransformer", "ECL", 96))

    def test_official_pems_and_solar_settings(self):
        self.assertIn("--e_layers 4", self.a("iTransformer", "PEMS08", 48))
        self.assertIn("--use_norm 0", self.a("iTransformer", "PEMS08", 48))
        self.assertIn("--use_norm 1", self.a("iTransformer", "PEMS08", 12))
        self.assertIn("--learning_rate 0.0007", self.a("S_Mamba", "PEMS08", 24))
        self.assertTrue(all("--use_norm 0" in self.a(m, "PEMS04", H) for m in ("iTransformer", "S_Mamba") for H in (12, 24, 48, 96)))
        self.assertIn("--train_epochs 5", self.a("S_Mamba", "PEMS03", 12))
        self.assertIn("--e_layers 5", self.a("S_Mamba", "PEMS03", 48))
        self.assertIn("--learning_rate 0.0001", self.a("S_Mamba", "PEMS03", 48))
        self.assertIn("--batch_size 16", self.a("iTransformer", "PEMS07", 96))
        self.assertIn("--use_norm 0", self.a("S_Mamba", "PEMS07", 24))
        self.assertIn("--learning_rate 0.0005", self.a("iTransformer", "Solar", 336))

    def test_protocol(self):
        self.assertEqual(" ".join(emit_cli_args(protocol("ETTm1"))), "--freq h --drop_last --test_batch_size 1")
        self.assertTrue(" ".join(emit_cli_args(protocol("Solar"))).endswith("--no_time_marks"))
        self.assertIn("--pems_split itransformer", " ".join(emit_cli_args(protocol("PEMS07"))))

    def test_benchmark_jobs_parse_with_the_protocol(self):
        import importlib

        from run import parse_args

        make_jobs = importlib.import_module("benchmark.make_jobs")
        argv = sys.argv
        try:
            sys.argv = ["make_jobs.py", "baselines", "--seeds", "2021", "--list", "_test_baselines"]
            with contextlib.redirect_stdout(io.StringIO()):
                make_jobs.main()
        finally:
            sys.argv = argv
        from src.utils.paths import jobs_file

        jf = jobs_file("_test_baselines")
        n, bad = 0, []
        for line in open(jf):
            toks = shlex.split(line.split(" > ")[0])
            args = parse_args(toks[toks.index("scripts/run.py") + 1:])
            n += 1
            if args.freq != "h" or not args.drop_last or args.test_batch_size != 1:
                bad.append(args.des)
        jf.unlink()
        self.assertEqual(n, 312)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
