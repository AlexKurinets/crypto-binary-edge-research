"""When not to trade: a rule's net result split by the conditions at decision time.

No-trade filters come from two places. Some are registered with the hypothesis in advance,
from reasoning the data is too thin to test (scheduled releases come twice a week). The rest
must be earned: a bucket is marked "avoid" only when its whole 95% interval is below zero over
at least min_clusters independent asset-hours. A bucket that is merely negative is "within
noise". Cutting every negative-looking bucket fits the filters to luck, and the paper period
would then pay for it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgelab.stats import cluster_bootstrap

REGIME_ORDER = ["range", "trend", "breakout", "shock"]


def _ranged(values: pd.Series, spec: list) -> tuple[pd.Series, pd.Series]:
    """Half-open buckets [lo, hi) from the config."""
    label = pd.Series(None, index=values.index, dtype="object")
    order = pd.Series(-1, index=values.index)
    for i, (lo, hi, name) in enumerate(spec):
        hit = (values >= lo) & (values < hi)
        label[hit] = name
        order[hit] = i
    return label, order


def labels(frame: pd.DataFrame, cfg: dict) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Bucket label and sort order for every dimension, per row."""
    c = cfg["conditions"]
    regime = frame["regime"].astype(str)
    release = np.where(frame["release"], "inside release window", "outside")
    return {
        "Spread at decision": _ranged(frame["yes_ask"] - frame["yes_bid"], c["spread_cents"]),
        "Minutes to close": _ranged(frame["tau_min"], c["minutes_to_close"]),
        "Contract price": _ranged(frame["decision_price"], c["contract_price"]),
        "Market regime": (regime, regime.map({r: i for i, r in enumerate(REGIME_ORDER)})),
        "Scheduled release": (pd.Series(release, index=frame.index), pd.Series(np.where(frame["release"], 1, 0), index=frame.index)),
        "Asset": (frame["asset"], frame["asset"].map({"BTC": 0, "ETH": 1})),
    }


def excluded_mask(frame: pd.DataFrame, no_trade: list, cfg: dict) -> np.ndarray:
    """Rows a hypothesis's registered no-trade filters block."""
    mask = np.zeros(len(frame), dtype=bool)
    if not no_trade or frame.empty:
        return mask
    labs = labels(frame, cfg)
    for dim, bucket in no_trade:
        mask |= (labs[dim][0] == bucket).to_numpy()
    return mask


def labels_for_row(row: dict, cfg: dict) -> dict[str, str]:
    """The same buckets for a single live decision."""
    frame = pd.DataFrame([row])
    return {dim: str(lab.iloc[0]) for dim, (lab, _) in labels(frame, cfg).items()}


def analyse(trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    inf = cfg["inference"]
    rows = []
    for dim, (lab, order) in labels(trades, cfg).items():
        for bucket in lab.dropna().unique():
            t = trades[lab == bucket]
            est = cluster_bootstrap(t["net"], t["contracts"], t["cluster"], inf["bootstrap_resamples"], inf["bootstrap_seed"])
            if est.n_clusters < inf["min_clusters"]:
                verdict = "too few"
            elif est.ci_hi < 0:
                verdict = "avoid"
            elif est.mean < 0:
                verdict = "within noise"
            else:
                verdict = "keep"
            rows.append(
                {
                    "dimension": dim,
                    "bucket": str(bucket),
                    "sort_order": int(order[lab == bucket].iloc[0]),
                    "n_trades": len(t),
                    "n_clusters": est.n_clusters,
                    "net": est.mean,
                    "ci_lo": est.ci_lo,
                    "ci_hi": est.ci_hi,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)
