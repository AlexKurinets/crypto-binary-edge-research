"""Features at a decision time, from 1-minute bars that closed at or before it.

There are two implementations of the same arithmetic. `bar_features` runs vectorized over the
whole history for research, and `features_at` runs on the tail of a live buffer for paper
trading. One test holds them to the same numbers. Another changes every bar after a decision
time and checks that the features at that time do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgelab.synthetic import in_release_window

LOOKBACK_BARS = 1441  # 1,440 one-minute returns for the daily realized variance
RANGE_BARS = 240  # the prior four hours, for range and breakout
BREAKOUT_RECENT = 15  # a breakout must clear the range as it stood 15 minutes ago
SHOCK_RATIO = 4.0  # 15-minute variance at four times the daily rate
TREND_ER = 0.35  # efficiency ratio over the last hour that counts as trending

FEATURE_COLUMNS = ["close", "rv15", "rv60", "rv1440", "ret_1m_z", "ret_15m_z", "er60", "vol_ratio", "breakout"]


def classify_regime(vol_ratio, breakout, er60):
    """shock > breakout > trend > range, by priority."""
    return np.select(
        [np.asarray(vol_ratio) >= SHOCK_RATIO, np.asarray(breakout) != 0, np.asarray(er60) >= TREND_ER],
        ["shock", "breakout", "trend"],
        default="range",
    )


def bar_features(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per asset and bar close time. Rows without a full day of history are dropped."""
    frames = []
    for asset, g in bars.groupby("asset", sort=False):
        g = g.sort_values("ts")
        logc = np.log(g["close"])
        r = logc.diff()
        r2 = r * r
        rv60 = r2.rolling(60).mean()
        rv1440 = r2.rolling(1440).mean()
        prior_high = g["high"].shift(BREAKOUT_RECENT).rolling(RANGE_BARS - BREAKOUT_RECENT).max()
        prior_low = g["low"].shift(BREAKOUT_RECENT).rolling(RANGE_BARS - BREAKOUT_RECENT).min()
        f = pd.DataFrame(
            {
                "asset": asset,
                "ts": g["ts"].to_numpy(),
                "close": g["close"].to_numpy(),
                "rv15": r2.rolling(15).mean().to_numpy(),
                "rv60": rv60.to_numpy(),
                "rv1440": rv1440.to_numpy(),
                "ret_1m_z": (r / np.sqrt(rv60)).to_numpy(),
                "ret_15m_z": ((logc - logc.shift(15)) / np.sqrt(15 * rv60)).to_numpy(),
                "er60": ((logc - logc.shift(60)).abs() / r.abs().rolling(60).sum()).to_numpy(),
                "breakout": np.where(g["close"] > prior_high, 1, np.where(g["close"] < prior_low, -1, 0)),
            }
        )
        f["vol_ratio"] = f["rv15"] / f["rv1440"]
        frames.append(f.dropna(subset=["rv1440"]))
    out = pd.concat(frames, ignore_index=True)
    out["regime"] = classify_regime(out["vol_ratio"], out["breakout"], out["er60"])
    return out


def features_at(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> dict:
    """The same features for the newest bar of a live buffer."""
    if len(closes) < LOOKBACK_BARS or len(highs) < RANGE_BARS:
        raise ValueError("need a full day of 1-minute bars")
    logc = np.log(np.asarray(closes[-LOOKBACK_BARS:], dtype=float))
    r = np.diff(logc)
    r2 = r * r
    rv15, rv60, rv1440 = r2[-15:].mean(), r2[-60:].mean(), r2.mean()
    prior_high = np.max(highs[-RANGE_BARS:-BREAKOUT_RECENT])
    prior_low = np.min(lows[-RANGE_BARS:-BREAKOUT_RECENT])
    close = float(closes[-1])
    feats = {
        "close": close,
        "rv15": rv15,
        "rv60": rv60,
        "rv1440": rv1440,
        "ret_1m_z": r[-1] / np.sqrt(rv60),
        "ret_15m_z": (logc[-1] - logc[-16]) / np.sqrt(15 * rv60),
        "er60": abs(logc[-1] - logc[-61]) / np.abs(r[-60:]).sum(),
        "breakout": 1 if close > prior_high else (-1 if close < prior_low else 0),
        "vol_ratio": rv15 / rv1440,
    }
    feats["regime"] = str(classify_regime(feats["vol_ratio"], feats["breakout"], feats["er60"]))
    return feats


def release_window(ts: pd.Series | pd.DatetimeIndex, releases: pd.DataFrame) -> np.ndarray:
    """Within 15 minutes before or 30 after a scheduled release. The calendar is public in advance."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    t = ((pd.Series(ts) - epoch).dt.total_seconds()).to_numpy().astype(np.int64)
    rel = np.sort(((releases["ts"] - epoch).dt.total_seconds()).to_numpy().astype(np.int64))
    return in_release_window(t, rel)
