"""Turn each registered hypothesis into at most one order per asset and decision time.

Every rule sees one row per strike with the model probability, the decision-time quote and the
features. It picks a side and a score, and the best-scoring strike in each snapshot is the
order. Rules that are about direction rather than price buy the contract nearest 50 cents,
since that is where a directional view has the most leverage on the payout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgelab.costs import taker_fee

SNAPSHOT_KEY = ["asset", "ts"]
NEAR_MONEY = 0.25  # directional rules only trade contracts with a mid between 25 and 75 cents


def add_edges(frame: pd.DataFrame, order_contracts: int, coefficient: float) -> pd.DataFrame:
    """Decision-time price of each side and the model's edge over it after the estimated fee."""
    f = frame.copy()
    f["yes_price"] = f["yes_ask"] / 100
    f["no_price"] = (100 - f["yes_bid"]) / 100
    f["mid_yes"] = (f["yes_bid"] + f["yes_ask"]) / 200
    fee_yes = taker_fee(f["yes_price"], order_contracts, coefficient) / order_contracts
    fee_no = taker_fee(f["no_price"], order_contracts, coefficient) / order_contracts
    f["edge_yes"] = f["p_model"] - f["yes_price"] - fee_yes
    f["edge_no"] = (1 - f["p_model"]) - f["no_price"] - fee_no
    return f


def _pick(cand: pd.DataFrame) -> pd.DataFrame:
    if cand.empty:
        return cand
    return cand.sort_values("score", kind="stable").groupby(SNAPSHOT_KEY, sort=False).tail(1).sort_values(SNAPSHOT_KEY)


def _priced_side(f: pd.DataFrame, lo: float, hi: float, target: float) -> pd.DataFrame:
    yes = f[(f["yes_price"] >= lo) & (f["yes_price"] <= hi)].assign(side="yes")
    yes["score"] = -(yes["yes_price"] - target).abs()
    no = f[(f["no_price"] >= lo) & (f["no_price"] <= hi)].assign(side="no")
    no["score"] = -(no["no_price"] - target).abs()
    return pd.concat([yes, no])


def _directional(f: pd.DataFrame, direction: pd.Series) -> pd.DataFrame:
    near = (f["mid_yes"] - 0.5).abs() <= NEAR_MONEY
    sel = f[(direction != 0) & near].copy()
    sel["side"] = np.where(direction[sel.index] > 0, "yes", "no")
    sel["score"] = -(sel["mid_yes"] - 0.5).abs()
    return sel


def signals(frame: pd.DataFrame, hyp: dict) -> pd.DataFrame:
    """The orders one hypothesis would send: the chosen rows with a `side` column."""
    f = frame
    if hyp.get("weekend") is not None:
        f = f[f["weekend"] == bool(hyp["weekend"])]
    if hyp.get("min_tau_min"):
        f = f[f["tau_min"] >= hyp["min_tau_min"]]
    if hyp.get("regime"):
        f = f[f["regime"] == hyp["regime"]]
    kind = hyp["kind"]

    if kind == "fair_value":
        best_yes = f["edge_yes"] >= f["edge_no"]
        edge = np.where(best_yes, f["edge_yes"], f["edge_no"])
        cand = f[edge >= hyp["min_edge"]].copy()
        cand["side"] = np.where(best_yes[edge >= hyp["min_edge"]], "yes", "no")
        cand["score"] = edge[edge >= hyp["min_edge"]]
    elif kind == "fast_move":
        z = f["ret_1m_z"]
        up, down = z >= hyp["min_abs_z"], z <= -hyp["min_abs_z"]
        edge = np.where(up, f["edge_yes"], f["edge_no"])
        mask = (up | down) & (edge >= hyp["min_edge"])
        cand = f[mask].copy()
        cand["side"] = np.where(up[mask], "yes", "no")
        cand["score"] = edge[mask]
    elif kind == "favourite":
        lo, hi = hyp["price_range"]
        cand = _priced_side(f, lo, hi, target=lo)
    elif kind == "tails":
        lo, hi = hyp["price_range"]
        cand = _priced_side(f[f["vol_ratio"] >= hyp["min_vol_ratio"]], lo, hi, target=(lo + hi) / 2)
    elif kind == "directional":
        s = f[hyp["signal"]]
        thr = hyp["threshold"]
        direction = pd.Series(np.where(s >= thr, 1, np.where(s <= -thr, -1, 0)), index=f.index)
        if not hyp["follow"]:
            direction = -direction
        cand = _directional(f, direction)
    elif kind == "cross_asset":
        lead = frame[frame["asset"] == hyp["leader"]].groupby("ts")["ret_1m_z"].first()
        fol = f[f["asset"] == hyp["follower"]]
        lead_z = fol["ts"].map(lead)
        move = (lead_z.abs() >= hyp["leader_min_z"]) & (fol["ret_1m_z"].abs() <= hyp["follower_max_z"])
        direction = pd.Series(np.where(move, np.sign(lead_z), 0), index=fol.index)
        cand = _directional(fol, direction)
    else:
        raise ValueError(f"unknown hypothesis kind {kind!r}")
    return _pick(cand)
