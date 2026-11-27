import numpy as np

MASK_EPS = 1e-3


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true), MAPE(pred, true), MSPE(pred, true)


def mape_clip5(pred, true):
    """MAPE of the TimeMixer PEMS protocol: ratios above 5 and 0 / 0 count as 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs((pred - true) / true)
    r = np.nan_to_num(r, nan=0.0, posinf=np.inf)
    return float(np.mean(np.where(r > 5, 0, r)))


def mape_masked(pred, true):
    """MAPE over |true| > MASK_EPS (zeros come back as ~1e-5 after the float32 scale round trip)."""
    m = np.abs(true) > MASK_EPS
    return float(np.mean(np.abs((pred[m] - true[m]) / true[m]))) if m.any() else float("nan")


class StreamMetrics:
    """Running float64 sums of the test metrics, so that the test split need not be kept in memory (same definitions as above)."""

    def __init__(self):
        self.n = self.n_masked = self.nonfinite = 0
        self.s = dict.fromkeys(("ae", "se", "ape", "spe", "ape_clip5", "ape_masked"), 0.0)
        self.se_step, self.n_step = None, 0                    # squared error per forecast step (axis 1 of [B, H, C])

    def update(self, pred, true):
        self.nonfinite += int(pred.size - np.isfinite(pred).sum())
        d = pred.astype(np.float64) - true.astype(np.float64)
        t = true.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = d / t
        a = np.abs(r)
        self.n += d.size
        self.s["ae"] += np.abs(d).sum()
        se = np.square(d)
        self.s["se"] += se.sum()
        step = se.sum(axis=tuple(i for i in range(se.ndim) if i != 1))
        self.se_step = step if self.se_step is None else self.se_step + step
        self.n_step += d.size // d.shape[1]
        self.s["ape"] += a.sum()
        self.s["spe"] += np.square(r).sum()
        tm = np.nan_to_num(a, nan=0.0, posinf=np.inf)
        self.s["ape_clip5"] += np.where(tm > 5, 0.0, tm).sum()
        m = np.abs(t) > MASK_EPS
        self.n_masked += int(m.sum())
        self.s["ape_masked"] += a[m].sum()

    def mse_steps(self):
        """MSE of every forecast step, averaged over the windows and variables seen so far."""
        return [] if self.se_step is None else (self.se_step / self.n_step).tolist()

    def result(self):
        n, s = self.n, self.s
        mae, mse = s["ae"] / n, s["se"] / n
        mape_m = s["ape_masked"] / self.n_masked if self.n_masked else float("nan")
        return mae, mse, float(np.sqrt(mse)), s["ape"] / n, s["spe"] / n, s["ape_clip5"] / n, mape_m
