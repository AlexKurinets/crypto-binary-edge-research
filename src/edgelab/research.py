"""Purged walk-forward research over the history up to the research cutoff.

Each test week is predicted by a model fitted only on the four weeks before it, with an
embargo so no training target reaches into the test week. Every registered hypothesis is then
traded on the out-of-sample predictions at the decision-time quote, filled against the quote
two seconds later, and charged the fee. The one rule that survives, with its no-trade filters
and a model refitted on the final four weeks, is frozen for the paper period.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import psycopg

from edgelab.conditions import analyse, excluded_mask
from edgelab.config import content_hash, load_registry, load_research_config
from edgelab.costs import fill
from edgelab.db import copy_frame, meta, read_frame
from edgelab.features import bar_features, release_window
from edgelab.hypotheses import add_edges, signals
from edgelab.model import fit_model, har_target
from edgelab.stats import benjamini_hochberg, brier, cluster_bootstrap, log_loss, reliability

CONTESTED = (0.10, 0.90)  # calibration is scored where the market had a real opinion


def load_market(conn: psycopg.Connection) -> dict[str, pd.DataFrame]:
    return {
        "bars": read_frame(conn, "select asset, ts, open, high, low, close from market.bars order by asset, ts", ["ts"]),
        "contracts": read_frame(conn, "select ticker, asset, open_ts, close_ts, strike, result from market.contracts", ["open_ts", "close_ts"]),
        "quotes": read_frame(conn, "select * from market.quotes", ["ts", "fill_ts"]),
        "releases": read_frame(conn, "select ts, name from market.releases order by ts", ["ts"]),
    }


def build_snapshots(quotes, contracts, feats, releases) -> pd.DataFrame:
    """One row per strike and decision time, with the features as of that time."""
    q = quotes.merge(contracts[["ticker", "asset", "close_ts", "strike", "result"]], on="ticker", how="inner")
    q = q.merge(feats, on=["asset", "ts"], how="inner")
    q["tau_s"] = (q["close_ts"] - q["ts"]).dt.total_seconds()
    q["tau_min"] = (q["tau_s"] // 60).astype(int)
    q["release"] = release_window(q["ts"], releases)
    q["weekend"] = q["ts"].dt.dayofweek >= 5
    q["cluster"] = q["asset"] + "|" + q["close_ts"].dt.strftime("%Y-%m-%dT%H")
    q["mid"] = (q["yes_bid"] + q["yes_ask"]) / 200
    return q.sort_values(["ts", "asset", "strike"]).reset_index(drop=True)


def fold_windows(start: pd.Timestamp, cutoff: pd.Timestamp, wf: dict):
    test_start = start + pd.Timedelta(days=wf["warmup_days"] + wf["train_days"])
    k = 0
    while test_start < cutoff:
        test_end = min(test_start + pd.Timedelta(days=wf["test_days"]), cutoff)
        yield k, test_start - pd.Timedelta(days=wf["train_days"]), test_start, test_start, test_end
        k += 1
        test_start = test_end


def fit_window(har_rows, snaps, train_start, train_end, cfg):
    wf, m = cfg["walk_forward"], cfg["model"]
    embargo = pd.Timedelta(minutes=wf["embargo_minutes"])
    rows = har_rows[(har_rows["ts"] >= train_start) & (har_rows["ts"] <= train_end - embargo)]
    rows = rows[rows["ts"].dt.minute % wf["har_sample_every_min"] == 0]
    train = snaps[(snaps["ts"] >= train_start) & (snaps["close_ts"] <= train_end)]
    return fit_model(rows, train, m["student_t_dof"], m["settlement_window_s"])


def evaluate(oos: pd.DataFrame, registry: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ex, inf = cfg["execution"], cfg["inference"]
    framed = add_edges(oos, ex["order_contracts"], ex["fee_coefficient"])
    results, all_trades = [], []
    for h in registry["hypotheses"]:
        sig = signals(framed, h)
        sim = fill(sig, ex["order_contracts"], ex["limit_slip_cents"], ex["fee_coefficient"])
        sim["excluded"] = excluded_mask(sim, h.get("no_trade", []), cfg)
        sim["hypothesis_id"] = h["id"]
        every = sim[sim["filled"] & (sim["contracts"] > 0)]
        all_trades.append(every)  # blocked signals are kept and settled, so the filters can be judged
        live = sim[~sim["excluded"]]
        t = every[~every["excluded"]]
        net = cluster_bootstrap(t["net"], t["contracts"], t["cluster"], inf["bootstrap_resamples"], inf["bootstrap_seed"])
        gross = cluster_bootstrap(t["gross"], t["contracts"], t["cluster"], inf["bootstrap_resamples"], inf["bootstrap_seed"])
        w = t["contracts"].sum()

        def avg(col, t=t, w=w):
            return float((t[col] * t["contracts"]).sum() / w) if w else float("nan")

        results.append(
            {
                "hypothesis_id": h["id"],
                "name": h["name"],
                "family": h["family"],
                "description": h["description"],
                "signals": len(sig),
                "excluded": int(sim["excluded"].sum()),
                "n_trades": len(t),
                "n_clusters": net.n_clusters,
                "contracts": int(w),
                "fill_rate": len(t) / len(live) if len(live) else float("nan"),
                "hit_rate": avg("payout"),
                "gross": avg("gross"),
                "spread": avg("spread"),
                "latency": avg("latency"),
                "fee": avg("fee_per"),
                "net": net.mean,
                "net_ci_lo": net.ci_lo,
                "net_ci_hi": net.ci_hi,
                "gross_ci_lo": gross.ci_lo,
                "gross_ci_hi": gross.ci_hi,
                "p_value": net.p_value,
                "pnl_usd": float(t["pnl_usd"].sum()),
            }
        )
    res = pd.DataFrame(results)
    res["q_value"] = benjamini_hochberg(res["p_value"].fillna(1.0))
    res["verdict"] = [_verdict(r, cfg) for r in res.itertuples()]
    return res, pd.concat(all_trades, ignore_index=True)


def _verdict(r, cfg) -> str:
    inf = cfg["inference"]
    if r.n_clusters < inf["min_clusters"]:
        return "too few to tell"
    if r.net > 0 and r.q_value <= inf["fdr_q"]:
        return "survives"
    if r.gross_ci_lo > 0:
        return "edge erased by costs"
    return "no edge"


def run_research(conn: psycopg.Connection, cfg: dict | None = None, registry: dict | None = None) -> dict:
    cfg = cfg or load_research_config()
    registry = registry or load_registry()
    info = meta(conn)
    start, cutoff = pd.Timestamp(info["start"]), pd.Timestamp(info["research_cutoff"])
    market = load_market(conn)
    bars = market["bars"][market["bars"]["ts"] <= cutoff]
    feats = bar_features(bars)
    har_rows = feats.merge(har_target(bars), on=["asset", "ts"])
    quotes = market["quotes"][market["quotes"]["ts"] < cutoff]
    snaps = build_snapshots(quotes, market["contracts"], feats, market["releases"])

    folds, preds = [], []
    for k, train_start, train_end, test_start, test_end in fold_windows(start, cutoff, cfg["walk_forward"]):
        model = fit_window(har_rows, snaps, train_start, train_end, cfg)
        test = snaps[(snaps["ts"] >= test_start) & (snaps["ts"] < test_end)].copy()
        test["p_raw"] = model.raw(test["close"], test["strike"], test["tau_s"], test["rv15"], test["rv60"], test["rv1440"])
        test["p_model"] = model.recal.apply(test["p_raw"])
        test["fold"] = k
        preds.append(test)
        folds.append({"fold": k, "train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end, "model": json.dumps(model.to_json())})
    if not preds:
        raise RuntimeError("not enough history before the cutoff for one walk-forward fold")
    oos = pd.concat(preds, ignore_index=True)

    results, trades = evaluate(oos, registry, cfg)
    conds = []
    for hid, t in trades.groupby("hypothesis_id"):
        c = analyse(t, cfg)
        c["hypothesis_id"] = hid
        conds.append(c)
    conditions = pd.concat(conds, ignore_index=True) if conds else pd.DataFrame()

    contested = oos[(oos["mid"] >= CONTESTED[0]) & (oos["mid"] <= CONTESTED[1])]
    calib = []
    for source, col in (("model", "p_model"), ("market", "mid")):
        rel = reliability(contested[col], contested["result"])
        rel["source"], rel["period"] = source, "research"
        calib.append(rel)
    calibration = pd.concat(calib, ignore_index=True)

    survivors = results[results["verdict"] == "survives"].sort_values("net", ascending=False)
    frozen_model = fit_window(har_rows, snaps, cutoff - pd.Timedelta(days=cfg["walk_forward"]["train_days"]), cutoff, cfg)
    frozen_id = survivors["hypothesis_id"].iloc[0] if len(survivors) else None
    frozen_params = next((h for h in registry["hypotheses"] if h["id"] == frozen_id), None)
    avoid = []
    if frozen_params is not None:
        avoid = [{"dimension": d, "bucket": b, "source": "registered"} for d, b in frozen_params.get("no_trade", [])]
        if not conditions.empty:
            bad = conditions[(conditions["hypothesis_id"] == frozen_id) & (conditions["verdict"] == "avoid")]
            avoid += [
                {"dimension": r.dimension, "bucket": r.bucket, "source": "walk-forward"}
                for r in bad.itertuples()
                if (r.dimension, r.bucket) not in {(a["dimension"], a["bucket"]) for a in avoid}
            ]
    frozen = {"hypothesis": frozen_params, "avoid": avoid, "model": frozen_model.to_json()}

    hyp_hash = content_hash(registry["hypotheses"])
    survivor = survivors.iloc[0] if len(survivors) else None
    summary = {
        "folds": len(folds),
        "oos_start": oos["ts"].min().isoformat(),
        "oos_end": cutoff.isoformat(),
        "oos_days": round((cutoff - oos["ts"].min()).total_seconds() / 86400, 1),
        "oos_snapshots": int(len(oos)),
        "oos_asset_hours": int(oos["cluster"].nunique()),
        "contested_snapshots": int(len(contested)),
        "brier_model": brier(contested["p_model"], contested["result"]),
        "brier_market": brier(contested["mid"], contested["result"]),
        "logloss_model": log_loss(contested["p_model"], contested["result"]),
        "logloss_market": log_loss(contested["mid"], contested["result"]),
        "hypotheses": len(results),
        "survivors": survivors["hypothesis_id"].tolist(),
        "erased_by_costs": results.loc[results["verdict"] == "edge erased by costs", "hypothesis_id"].tolist(),
        "frozen_hypothesis": frozen_id,
        "avoid": avoid,
        "backtest": None
        if survivor is None
        else {
            "net": float(survivor["net"]),
            "net_ci": [float(survivor["net_ci_lo"]), float(survivor["net_ci_hi"])],
            "gross": float(survivor["gross"]),
            "trades": int(survivor["n_trades"]),
            "asset_hours": int(survivor["n_clusters"]),
            "q_value": float(survivor["q_value"]),
            "pnl_usd": float(survivor["pnl_usd"]),
        },
    }

    with conn.transaction():
        conn.execute(
            "insert into research.registrations (hypotheses_hash, hypothesis_ids) values (%s, %s) on conflict do nothing",
            (hyp_hash, [h["id"] for h in registry["hypotheses"]]),
        )
        summary["registrations"] = conn.execute("select count(*) from research.registrations").fetchone()[0]
        run_id = conn.execute(
            "insert into research.runs (research_start, cutoff, config, hypotheses_hash, summary) values (%s, %s, %s, %s, %s) returning run_id",
            (start, cutoff, json.dumps(cfg), hyp_hash, json.dumps(summary)),
        ).fetchone()[0]
        _store(conn, run_id, folds, oos, results, trades, conditions, calibration)
        conn.execute(
            "insert into research.frozen_rules (run_id, hypothesis_id, params, avoid, model, frozen_at, rule_hash) values (%s, %s, %s, %s, %s, %s, %s)",
            (run_id, frozen_id, json.dumps(frozen_params), json.dumps(avoid), json.dumps(frozen["model"]), cutoff, content_hash(frozen)),
        )
    conn.commit()
    return {"run_id": run_id, "created_at": datetime.now(UTC).isoformat(timespec="seconds"), **summary}


def _store(conn, run_id, folds, oos, results, trades, conditions, calibration):
    f = pd.DataFrame(folds).assign(run_id=run_id)
    copy_frame(conn, "research.folds", f, ["run_id", "fold", "train_start", "train_end", "test_start", "test_end", "model"])
    p = oos.rename(columns={"result": "outcome"}).assign(run_id=run_id)
    copy_frame(conn, "research.predictions", p, ["run_id", "fold", "ticker", "ts", "asset", "tau_min", "p_raw", "p_model", "mid", "outcome"])
    copy_frame(
        conn,
        "research.hypotheses",
        results.assign(run_id=run_id),
        ["run_id", "hypothesis_id", "name", "family", "description", "signals", "excluded", "n_trades", "n_clusters", "contracts", "fill_rate", "hit_rate",
         "gross", "spread", "latency", "fee", "net", "net_ci_lo", "net_ci_hi", "gross_ci_lo", "gross_ci_hi", "p_value", "q_value", "verdict", "pnl_usd"],
    )
    t = trades.rename(columns={"ts": "decided_at"}).assign(run_id=run_id)
    copy_frame(
        conn,
        "research.trades",
        t,
        ["run_id", "hypothesis_id", "ticker", "asset", "decided_at", "cluster", "side", "decision_price", "fill_price", "contracts", "fee", "payout",
         "gross", "spread", "latency", "fee_per", "net", "pnl_usd", "excluded"],
    )
    if not conditions.empty:
        copy_frame(
            conn,
            "research.conditions",
            conditions.assign(run_id=run_id),
            ["run_id", "hypothesis_id", "dimension", "bucket", "sort_order", "n_trades", "n_clusters", "net", "ci_lo", "ci_hi", "verdict"],
        )
    copy_frame(conn, "research.calibration", calibration.assign(run_id=run_id), ["run_id", "period", "source", "bin", "p_mean", "freq", "n"])


def latest_run(conn: psycopg.Connection) -> int | None:
    row = conn.execute("select max(run_id) from research.runs").fetchone()
    return row[0] if row else None


def summary_line(value: float) -> str:
    return f"{value * 100:+.2f}c per contract" if np.isfinite(value) else "n/a"
