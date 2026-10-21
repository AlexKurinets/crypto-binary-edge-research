"""The probability that a contract settles YES.

Three pieces, each fitted only on a fold's training window:

1. HAR-RV volatility. Next-hour variance per minute is forecast as a linear blend of realized
   variance over the last 15 minutes, hour and day (Corsi's heterogeneous autoregressive model).
2. Settlement-aware probability. The contract settles on the average of the index over its
   final 60 seconds, and averaging removes two thirds of the variance of that last minute. The
   log distance to the strike is scaled by the remaining volatility and read off a Student-t
   with fat tails, rescaled to the same variance.
3. Recalibration. A logistic fit of the outcome on logit(p) corrects any steady over- or
   under-confidence that the first two pieces leave behind.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t as student_t

HAR_FEATURES = ["rv15", "rv60", "rv1440"]
P_CLIP = (1e-4, 1 - 1e-4)


def har_target(bars: pd.DataFrame) -> pd.DataFrame:
    """Realized variance per minute over the NEXT 60 minutes. A training target, never a feature."""
    frames = []
    for asset, g in bars.groupby("asset", sort=False):
        g = g.sort_values("ts")
        r2 = np.log(g["close"]).diff() ** 2
        fwd = r2.rolling(60).mean().shift(-60)
        frames.append(pd.DataFrame({"asset": asset, "ts": g["ts"].to_numpy(), "fwd_rv60": fwd.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


@dataclass
class HarModel:
    intercept: float
    b15: float
    b60: float
    b1440: float

    def predict(self, rv15, rv60, rv1440):
        pred = self.intercept + self.b15 * np.asarray(rv15) + self.b60 * np.asarray(rv60) + self.b1440 * np.asarray(rv1440)
        return np.maximum(pred, 0.25 * np.asarray(rv1440))


def fit_har(rows: pd.DataFrame) -> HarModel:
    """Ordinary least squares on rows with rv15, rv60, rv1440 and fwd_rv60."""
    rows = rows.dropna(subset=[*HAR_FEATURES, "fwd_rv60"])
    x = np.column_stack([np.ones(len(rows)), rows[HAR_FEATURES].to_numpy()])
    coef, *_ = np.linalg.lstsq(x, rows["fwd_rv60"].to_numpy(), rcond=None)
    return HarModel(*map(float, coef))


def effective_seconds(tau_s, window_s: int = 60):
    """Variance-equivalent seconds left when settlement averages the final window_s seconds.

    For Brownian motion, the average over the last w seconds has the variance of w/3 seconds
    of the path, so a contract with tau seconds left carries tau - w + w/3 seconds of risk.
    """
    tau = np.asarray(tau_s, dtype=float)
    if np.any(tau < window_s):
        raise ValueError("decisions inside the settlement window need the seconds already averaged")
    return tau - window_s + window_s / 3.0


def prob_at_or_above(spot, strike, tau_s, var_per_min, dof: float, window_s: int = 60):
    var = np.asarray(var_per_min) / 60.0 * effective_seconds(tau_s, window_s)
    scale = np.sqrt(var * (dof - 2.0) / dof)
    return student_t.sf(np.log(np.asarray(strike) / np.asarray(spot)) / scale, dof)


@dataclass
class Recalibration:
    a: float = 0.0
    b: float = 1.0

    def apply(self, p):
        x = logit(np.clip(np.asarray(p, dtype=float), *P_CLIP))
        return np.clip(expit(self.a + self.b * x), 0.001, 0.999)


def fit_recalibration(p, y, iters: int = 50) -> Recalibration:
    """Two-parameter logistic regression of y on logit(p), by iteratively reweighted least squares."""
    x = logit(np.clip(np.asarray(p, dtype=float), *P_CLIP))
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        eta = design @ beta
        mu = expit(eta)
        w = mu * (1 - mu) + 1e-9
        z = eta + (y - mu) / w
        new = np.linalg.solve(design.T @ (w[:, None] * design), design.T @ (w * z))
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    return Recalibration(float(beta[0]), float(beta[1]))


@dataclass
class FittedModel:
    har: HarModel
    recal: Recalibration
    dof: float
    window_s: int = 60

    def raw(self, spot, strike, tau_s, rv15, rv60, rv1440):
        var = self.har.predict(rv15, rv60, rv1440)
        return prob_at_or_above(spot, strike, tau_s, var, self.dof, self.window_s)

    def prob(self, spot, strike, tau_s, rv15, rv60, rv1440):
        return self.recal.apply(self.raw(spot, strike, tau_s, rv15, rv60, rv1440))

    def to_json(self) -> dict:
        return {"har": asdict(self.har), "recal": asdict(self.recal), "dof": self.dof, "window_s": self.window_s}

    @classmethod
    def from_json(cls, d: dict) -> FittedModel:
        return cls(HarModel(**d["har"]), Recalibration(**d["recal"]), d["dof"], d["window_s"])


def fit_model(train_bars_rows: pd.DataFrame, train_snapshots: pd.DataFrame, dof: float, window_s: int = 60) -> FittedModel:
    """Fit HAR on per-minute rows, then recalibrate on the same window's settled snapshots."""
    har = fit_har(train_bars_rows)
    model = FittedModel(har, Recalibration(), dof, window_s)
    s = train_snapshots
    p_raw = model.raw(s["close"], s["strike"], s["tau_s"], s["rv15"], s["rv60"], s["rv1440"])
    model.recal = fit_recalibration(p_raw, s["result"])
    return model
