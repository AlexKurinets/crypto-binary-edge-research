"""Inference that respects how few independent outcomes there really are.

Every strike and every decision in one asset-hour settles off the same 60-second average, so
those trades are one outcome seen several times, not several outcomes. Confidence intervals
therefore resample whole asset-hours (a cluster bootstrap), and the p-values for the full set
of registered hypotheses are adjusted together with Benjamini-Hochberg, so testing twelve ideas
does not hand one of them a lucky pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class Estimate:
    mean: float
    ci_lo: float
    ci_hi: float
    p_value: float
    n_clusters: int


def cluster_bootstrap(values, weights, clusters, resamples: int = 2000, seed: int = 7) -> Estimate:
    """Weighted mean with a percentile CI and a one-sided p-value for mean <= 0.

    The p-value comes from the bootstrap distribution shifted to the null: how often a resampled
    mean minus the estimate is at least as large as the estimate.
    """
    df = pd.DataFrame({"v": np.asarray(values, float) * np.asarray(weights, float), "w": np.asarray(weights, float), "c": clusters})
    agg = df.groupby("c", sort=True)[["v", "w"]].sum()  # sorted, so row order never changes the resamples
    s, w = agg["v"].to_numpy(), agg["w"].to_numpy()
    n = len(agg)
    if n == 0 or w.sum() == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    est = s.sum() / w.sum()
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 250):  # chunks keep memory flat for large cluster counts
        stop = min(resamples, start + 250)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = s[idx].sum(axis=1) / np.maximum(w[idx].sum(axis=1), 1e-12)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = (1 + np.sum(means - est >= est)) / (resamples + 1)
    return Estimate(float(est), float(lo), float(hi), float(p), int(n))


def benjamini_hochberg(p_values) -> np.ndarray:
    """Adjusted p-values (q-values) controlling the false discovery rate."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def min_detectable_edge(n_independent, sd: float = 0.5, alpha: float = 0.05, power: float = 0.8):
    """Smallest per-contract edge a one-sided test detects with the given power.

    A binary contract bought near 50 cents pays 0 or 1, so one outcome has a standard deviation
    of about 50 cents. With n independent outcomes the standard error is 0.5 / sqrt(n).
    """
    return (norm.ppf(1 - alpha) + norm.ppf(power)) * sd / np.sqrt(np.asarray(n_independent, dtype=float))


def outcomes_needed(edge: float, sd: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> int:
    return int(np.ceil(((norm.ppf(1 - alpha) + norm.ppf(power)) * sd / edge) ** 2))


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    """Mean forecast and observed frequency in equal-width probability bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(p_mean=("p", "mean"), freq=("y", "mean"), n=("y", "size"))
    return df.reset_index()
