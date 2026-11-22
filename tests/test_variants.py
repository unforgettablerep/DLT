import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.layers import DLT_layers  # noqa: E402
from src.model.DLT import Model  # noqa: E402
from src.utils import paths  # noqa: E402

PREVIOUS = "af398bd:tslib/models/DLT.py"


def cfg(**kw):
    c = dict(
        task_name="long_term_forecast",
        seq_len=96,
        pred_len=24,
        enc_in=12,
        dlt_variant="dltdyn",
        dlt_k=4,
        dlt_L=10,
        dlt_R=3,
        dlt_dyn_dim=8,
        dlt_ccf_len=8000,
        dlt_impl="kernel",
        dlt_topk=3,
        dlt_gate="none",
    )
    c.update(kw)
    return types.SimpleNamespace(**c)


def previous_module():
    """The DLT model of commit af398bd (the reference of the 8 original variants), or None without git history."""
    res = subprocess.run(["git", "-C", str(REPO), "show", PREVIOUS], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(res.stdout)
    tmp.close()
    spec = importlib.util.spec_from_file_location("DLT_previous", tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.unlink(tmp.name)
    return module


def dense_kernel(m, xn):
    """Per-sample dense kernels [B, C, k, L + 1] of every variant, built independently of cross()."""
    B = xn.shape[0]
    C, k, L = m.C, m.k, m.L
    K = torch.zeros(B, C, k, L + 1)
    v = m.variant
    if v == "scalar":
        K[..., 0] = m.a
    elif v == "scalardyn":
        K[..., 0] = m.a[None] * (1 + torch.tanh(m.mod(m._pair_features(xn)).squeeze(-1)))
    elif v == "shift":
        for i in range(C):
            for j in range(k):
                K[:, i, j, m.lag[i, j]] = m.a[i, j]
    elif v in ("tap3", "full"):
        K[..., :m.w.shape[-1]] = m.w
    elif v in ("dlt", "dltdyn"):
        al = m.alpha[None].expand(B, -1, -1, -1)
        if v == "dltdyn":
            al = m.alpha[None] * (1 + torch.tanh(m.mod(m._pair_features(xn))))
        K = torch.einsum("bckr,rl->bckl", al, m.phi)
    elif v in ("topk", "mill", "milldyn"):
        if v == "topk":
            w = m.w[None].expand(B, -1, -1, -1)
        else:
            w = torch.zeros(B, C, k, m.K)
            for b in range(B):
                if v == "milldyn":
                    s = m.summ(xn[b])
                    q = m.q_proj(s)
                    kk = m.k_proj(s)
                for i in range(C):
                    if v == "milldyn":
                        sc = torch.stack([q[i] @ kk[m.idx[i, j]] for j in range(k)]) / np.sqrt(q.shape[-1])
                        att = torch.softmax(sc, 0)
                    for j in range(k):
                        a = torch.sigmoid(m.b_att[i, j]) if v == "mill" else att[j]
                        be, bi = torch.sigmoid(m.b_exc[i, j]), torch.sigmoid(m.b_inh[i, j])
                        for mm in range(m.K):
                            t = m.tlag[i, j, mm].double()
                            w[b, i, j, mm] = m.a[i, j] * torch.relu(torch.exp(-be * t) - (1 - a) * torch.exp(-bi * t))
        for i in range(C):
            for j in range(k):
                for mm in range(m.K):
                    K[:, i, j, m.tlag[i, j, mm]] += w[:, i, j, mm]
    return K


def reference_z(m, xn):
    """Explicit z_i(t) = sum_j sum_tau K_ij(tau) x_j(t - tau)."""
    K = dense_kernel(m, xn)
    B, C, T = xn.shape
    z = torch.zeros(B, C, T)
    for i in range(C):
        for jj in range(m.k):
            j = m.idx[i, jj]
            for tau in range(m.L + 1):
                if K[:, i, jj, tau].abs().max() == 0:
                    continue
                z[:, i, tau:] += K[:, i, jj, tau, None] * xn[:, j, :T - tau]
    return z


class DLTVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        cls.C = 12
        cls.Xtr = torch.randn(3000, cls.C).cumsum(0).numpy()
        cls.xn = torch.randn(3, cls.C, 96)
        cls.xin = torch.randn(4, 96, cls.C)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.dtype)

    def build(self, seed, **kw):
        torch.manual_seed(seed)
        m = Model(cfg(**kw))
        m.prepare(self.Xtr)
        return m

    def test_neighbour_and_lag_selection(self):
        if not paths.exists("ETTh1.csv"):
            self.skipTest("ETTh1.csv is not in the data directory")
        ett = pd.read_csv(paths.path_of("ETTh1.csv")).values[:, 1:].astype(float)[:8545]
        ett = (ett - ett.mean(0)) / ett.std(0)
        i1, l1 = DLT_layers.select_neighbours(ett, 6, 48)
        previous = previous_module()
        if previous is not None:
            i0, l0 = previous.select_neighbours(ett, 6, 48)
            self.assertTrue(torch.equal(i0, i1) and torch.equal(l0, l1), "select_neighbours differs from the previous version")
        tl = DLT_layers.select_lags(ett, i1, 5, 48)
        x = np.diff(ett, axis=0)[-8000:]
        x = (x - x.mean(0)) / x.std(0)
        n = len(x)
        for i in range(7):
            for j in range(6):
                s = i1[i, j].item()
                r = np.array([abs((x[t:, i] * x[:n - t, s]).sum()) / (n - t) for t in range(49)])
                self.assertEqual(sorted(np.argsort(-r)[:5].tolist()), tl[i, j].tolist())
                self.assertEqual(tl[i, j, :].tolist().count(l1[i, j].item()), 1)

    def test_cross_matches_reference_loop(self):
        for v in DLT_layers.VARIANTS[1:]:
            for impl in ("kernel", "basis") if v in ("dlt", "dltdyn") else ("kernel",):
                m = self.build(1, dlt_variant=v, dlt_impl=impl)
                with torch.no_grad():
                    for p in m.parameters():
                        p.add_(0.3 * torch.randn_like(p))
                    err = (m.cross(self.xn) - reference_z(m, self.xn)).abs().max().item()
                self.assertLess(err, 1e-10, "{} [{}]".format(v, impl))

    def test_topk1_is_shift(self):
        ms = self.build(2, dlt_variant="shift")
        mt = Model(cfg(dlt_variant="topk", dlt_topk=1))
        mt.prepare(self.Xtr)
        with torch.no_grad():
            mt.w.copy_(ms.a[..., None])
        self.assertTrue(torch.equal(mt.tlag[..., 0], ms.lag))
        self.assertTrue(torch.allclose(mt.cross(self.xn), ms.cross(self.xn), atol=1e-12))

    def test_original_variants_unchanged(self):
        previous = previous_module()
        if previous is None:
            self.skipTest("git history with {} is not available".format(PREVIOUS))
        for v in previous.VARIANTS:
            torch.manual_seed(3)
            mo = previous.Model(cfg(dlt_variant=v))
            mo.prepare(self.Xtr)
            mn = self.build(3, dlt_variant=v)
            self.assertEqual(list(mo.state_dict()), list(mn.state_dict()), v)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(mo.state_dict().values(), mn.state_dict().values())), v)
            self.assertTrue(torch.equal(mo(self.xin, None, None, None), mn(self.xin, None, None, None)), v)

    def test_gates(self):
        base = self.build(4, dlt_variant="dltdyn")
        for gate in ("moe", "static"):
            mg = self.build(4, dlt_variant="dltdyn", dlt_gate=gate, dlt_gate_init=-3.0)
            with torch.no_grad():
                x = self.xin.permute(0, 2, 1)
                mu = x.mean(-1, keepdim=True)
                sd = x.std(-1, keepdim=True) + 1e-5
                xx = (x - mu) / sd
                ys, yc = mg.self_head(xx), mg.cross_head(mg.cross(xx))
                if gate == "moe":
                    gw = torch.softmax(xx @ mg.gate_net.weight.T + mg.gate_net.bias, -1)
                    y = gw[..., :1] * ys + gw[..., 1:] * yc
                else:
                    y = ys + 1 / (1 + np.exp(3.0)) * yc
                ref = (y * sd + mu).permute(0, 2, 1)
            shared = all(torch.equal(base.state_dict()[k], v) for k, v in mg.state_dict().items() if k in base.state_dict())
            self.assertTrue(shared, gate)
            self.assertTrue(torch.allclose(mg(self.xin, None, None, None), ref, atol=1e-12), gate)

    def test_finite_and_checkpoint_round_trip(self):
        for v in DLT_layers.VARIANTS:
            for gate in ("none",) if v == "ci" else DLT_layers.GATES:
                m = self.build(5, dlt_variant=v, dlt_gate=gate)
                y = m(self.xin, None, None, None)
                y.pow(2).mean().backward()
                self.assertTrue(bool(torch.isfinite(y).all()), "{}/{}".format(v, gate))
                self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters()), "{}/{}".format(v, gate))
                buf = io.BytesIO()
                torch.save(m.state_dict(), buf)
                buf.seek(0)
                m2 = Model(cfg(dlt_variant=v, dlt_gate=gate))
                m2.load_state_dict(torch.load(buf))
                self.assertTrue(torch.equal(m2(self.xin, None, None, None), y.detach()), "{}/{}".format(v, gate))

    def test_gradients_reach_mill_and_gate_parameters(self):
        m = self.build(6, dlt_variant="milldyn", dlt_gate="static")
        m(self.xin, None, None, None).pow(2).mean().backward()
        for p in (m.b_exc, m.b_inh, m.q_proj.weight, m.gate_logit):
            self.assertTrue(p.grad is not None and p.grad.abs().sum() > 0)


if __name__ == "__main__":
    unittest.main()
