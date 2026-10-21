"""Forward paper trading with the frozen rule.

The trader talks to a feed with the same four calls the live adapters exposed: bars up to now,
the quote snapshot at a decision time, the quote an order meets after the latency, and the
contracts that have settled by now. In production the feed wrapped the exchange and index
connections. Here `ReplayFeed` serves the synthetic paper period from Postgres, and it will not
hand out a bar, a fill quote or a settlement before its time.

The clock advances a minute at a time. At each decision time the trader computes features from
the bars it has been given so far. Orders for an hour's contracts are decided together when the
hour ends, before those contracts settle. The rule keeps no state between snapshots, so batching
gives exactly the orders that deciding each snapshot on arrival would, and it uses the same
`signals` and `fill` code as the backtest.

Every decision is written to paper.decisions before its contract settles, and a trigger refuses
any later change to what was decided. Signals the no-trade filters block are logged as skipped
and settled too, so the filters themselves are tested.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import psycopg

from edgelab.conditions import labels_for_row
from edgelab.config import load_research_config
from edgelab.costs import fill, taker_fee
from edgelab.db import copy_frame, meta, read_frame
from edgelab.features import LOOKBACK_BARS, RANGE_BARS, features_at
from edgelab.hypotheses import add_edges, signals
from edgelab.model import FittedModel
from edgelab.research import CONTESTED
from edgelab.stats import brier, cluster_bootstrap, reliability
from edgelab.synthetic import in_release_window

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
NUMERIC_FEATURES = ["close", "rv15", "rv60", "rv1440", "ret_1m_z", "ret_15m_z", "er60", "breakout", "vol_ratio"]
SNAPSHOT_COLUMNS = ["ticker", "asset", "strike", "close_ts", "yes_bid", "yes_ask", "bid_size", "ask_size"]


def _iso(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S+00")


def _epoch_s(values: pd.Series) -> np.ndarray:
    return ((values - EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def _ts(seconds) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(seconds, dtype=np.int64), unit="s", utc=True)


class ReplayFeed:
    """Point-in-time access to the paper period. Times are epoch seconds internally."""

    def __init__(self, conn: psycopg.Connection, start: pd.Timestamp, end: pd.Timestamp, latency_s: int):
        warm = start - pd.Timedelta(minutes=LOOKBACK_BARS)
        bars = read_frame(
            conn, f"select asset, ts, high, low, close from market.bars where ts > '{_iso(warm)}' and ts <= '{_iso(end)}' order by asset, ts", ["ts"]
        )
        self._bars = {
            a: {"ts": _epoch_s(g["ts"]), "close": g["close"].to_numpy(), "high": g["high"].to_numpy(), "low": g["low"].to_numpy()}
            for a, g in bars.groupby("asset")
        }
        q = read_frame(
            conn,
            "select q.*, c.asset, c.strike, c.close_ts from market.quotes q join market.contracts c using (ticker) "
            f"where q.ts >= '{_iso(start)}' and q.ts < '{_iso(end)}' order by q.ts, c.asset, c.strike",
            ["ts", "fill_ts", "close_ts"],
        )
        ts_s, fill_s, close_s = _epoch_s(q["ts"]), _epoch_s(q["fill_ts"]), _epoch_s(q["close_ts"])
        tickers = q["ticker"].to_numpy(dtype=object)
        self._fills = {
            (t, s): (f, b, a) for t, s, f, b, a in zip(tickers, ts_s, fill_s, q["fill_yes_bid"].to_numpy(), q["fill_yes_ask"].to_numpy(), strict=True)
        }
        cols = {
            "ticker": tickers,
            "asset": q["asset"].to_numpy(dtype=object),
            "strike": q["strike"].to_numpy(),
            "close_ts": close_s,
            "yes_bid": q["yes_bid"].to_numpy(),
            "yes_ask": q["yes_ask"].to_numpy(),
            "bid_size": q["bid_size"].to_numpy(),
            "ask_size": q["ask_size"].to_numpy(),
        }
        bounds = np.flatnonzero(np.diff(ts_s)) + 1
        starts, stops = np.r_[0, bounds], np.r_[bounds, len(ts_s)]
        self._snapshots = {int(ts_s[a]): {k: v[a:b] for k, v in cols.items()} for a, b in zip(starts, stops, strict=True)}
        settle = read_frame(
            conn, f"select ticker, close_ts, result from market.contracts where close_ts > '{_iso(start)}' and close_ts <= '{_iso(end)}' order by close_ts", ["close_ts"]
        )
        self._settle = settle
        self._settle_s = _epoch_s(settle["close_ts"])
        self._settled_upto = 0
        self.latency_s = latency_s
        self.start, self.end = start, end

    def clock(self) -> range:
        """Every minute of the paper period, as epoch seconds."""
        return range(int(self.start.timestamp()), int(self.end.timestamp()) + 1, 60)

    def history(self, asset: str, now_s: int, n: int) -> dict[str, np.ndarray]:
        """The last n bars that closed at or before now."""
        b = self._bars[asset]
        stop = int(np.searchsorted(b["ts"], now_s, side="right"))
        lo = max(0, stop - n)
        return {k: v[lo:stop] for k, v in b.items()}

    def snapshot(self, now_s: int) -> dict[str, np.ndarray] | None:
        return self._snapshots.get(now_s)

    def send(self, ticker: str, decided_s: int, now_s: int) -> dict:
        """The quote an order sent at decided_s meets, available only once the latency has passed."""
        fill_s, bid, ask = self._fills[(ticker, decided_s)]
        if now_s < fill_s:
            raise RuntimeError("fill quote requested before the order could have reached the book")
        return {"fill_yes_bid": int(bid), "fill_yes_ask": int(ask)}

    def settled(self, now_s: int) -> pd.DataFrame:
        """Contracts that settled since the last call, up to now."""
        stop = int(np.searchsorted(self._settle_s, now_s, side="right"))
        out = self._settle.iloc[self._settled_upto : stop]
        self._settled_upto = stop
        return out


class PaperTrader:
    def __init__(self, conn: psycopg.Connection, run_id: int, frozen: dict, cfg: dict, releases: pd.DataFrame):
        self.conn, self.run_id, self.cfg = conn, run_id, cfg
        self.rule = frozen["params"]
        self.avoid = {(a["dimension"], a["bucket"]) for a in frozen["avoid"]}
        self.model = FittedModel.from_json(frozen["model"])
        self.release_s = np.sort(_epoch_s(releases["ts"]))
        self.pending: list[dict] = []
        self.open: dict[int, dict] = {}
        self.predictions: list[pd.DataFrame] = []

    def step(self, feed: ReplayFeed, now_s: int) -> None:
        if now_s % 3600 == 0:
            self.decide_pending(feed, now_s)
        self.settle(feed.settled(now_s))
        snap = feed.snapshot(now_s)
        if snap is not None:
            self.observe(feed, snap, now_s)

    def observe(self, feed: ReplayFeed, snap: dict, now_s: int) -> None:
        """Features as of now, from bars the feed has released so far."""
        n = len(snap["ticker"])
        cols = {k: np.empty(n) for k in NUMERIC_FEATURES}
        regime = np.empty(n, dtype=object)
        for asset in np.unique(snap["asset"]):
            h = feed.history(asset, now_s, LOOKBACK_BARS)
            f = features_at(h["close"], h["high"][-RANGE_BARS:], h["low"][-RANGE_BARS:])
            m = snap["asset"] == asset
            for k in NUMERIC_FEATURES:
                cols[k][m] = f[k]
            regime[m] = f["regime"]
        release = bool(in_release_window(np.array([now_s]), self.release_s)[0])
        self.pending.append({**snap, **cols, "regime": regime, "ts_s": np.full(n, now_s), "release": np.full(n, release)})

    def decide_pending(self, feed: ReplayFeed, now_s: int) -> None:
        if not self.pending:
            return
        f = pd.DataFrame({k: np.concatenate([p[k] for p in self.pending]) for k in self.pending[0]})
        self.pending = []
        f["ts"] = _ts(f["ts_s"])
        f["tau_s"] = (f["close_ts"] - f["ts_s"]).astype(float)
        f["tau_min"] = (f["close_ts"] - f["ts_s"]) // 60
        f["weekend"] = f["ts"].dt.dayofweek >= 5
        f["mid"] = (f["yes_bid"] + f["yes_ask"]) / 200
        f["cluster"] = f["ticker"].str.rsplit("-", n=1).str[0]
        f["p_model"] = self.model.prob(f["close"], f["strike"], f["tau_s"], f["rv15"], f["rv60"], f["rv1440"])
        self.predictions.append(f[["ticker", "ts", "asset", "tau_min", "p_model", "mid"]])
        if self.rule is None:
            return
        ex = self.cfg["execution"]
        for pick in signals(add_edges(f, ex["order_contracts"], ex["fee_coefficient"]), self.rule).to_dict("records"):
            self.decide(feed, pick, now_s)

    def decide(self, feed: ReplayFeed, pick: dict, now_s: int) -> None:
        ex = self.cfg["execution"]
        side = pick["side"]
        price = pick["yes_price"] if side == "yes" else pick["no_price"]
        edge = pick["edge_yes"] if side == "yes" else pick["edge_no"]
        buckets = labels_for_row({**pick, "decision_price": price}, self.cfg)
        blocked = [f"{dim}: {b}" for dim, b in buckets.items() if (dim, b) in self.avoid]
        decided_s = int(pick["ts_s"])
        record = {
            "decided_at": pick["ts"], "asset": pick["asset"], "ticker": pick["ticker"], "strike": float(pick["strike"]), "tau_min": int(pick["tau_min"]),
            "index_price": float(pick["close"]), "p_model": float(pick["p_model"]), "yes_bid": int(pick["yes_bid"]), "yes_ask": int(pick["yes_ask"]),
            "side": side, "price": float(price), "edge": float(edge), "regime": pick["regime"],
            "fill_price": None, "contracts": 0, "fee": 0.0, "skip_reason": None,
        }
        est_fee_per = float(taker_fee(price, ex["order_contracts"], ex["fee_coefficient"]) / ex["order_contracts"])
        if blocked:
            record.update(action="skipped", skip_reason="; ".join(blocked))
        else:
            quote = feed.send(pick["ticker"], decided_s, decided_s + feed.latency_s)
            sim = fill(pd.DataFrame([{**pick, **quote}]), ex["order_contracts"], ex["limit_slip_cents"], ex["fee_coefficient"]).iloc[0]
            if sim["filled"] and sim["contracts"] > 0:
                record.update(action="filled", fill_price=float(sim["fill_price"]), contracts=int(sim["contracts"]), fee=float(sim["fee"]))
            else:
                record.update(action="missed")
        decision_id = self.conn.execute(
            """insert into paper.decisions (run_id, decided_at, logged_at, asset, ticker, strike, tau_min, index_price, p_model, yes_bid, yes_ask,
                 side, price, edge, regime, action, skip_reason, fill_price, contracts, fee)
               values (%(run_id)s, %(decided_at)s, %(decided_at)s, %(asset)s, %(ticker)s, %(strike)s, %(tau_min)s, %(index_price)s, %(p_model)s,
                 %(yes_bid)s, %(yes_ask)s, %(side)s, %(price)s, %(edge)s, %(regime)s, %(action)s, %(skip_reason)s, %(fill_price)s, %(contracts)s, %(fee)s)
               returning decision_id""",
            {**record, "run_id": self.run_id},
        ).fetchone()[0]
        self.open[decision_id] = {**record, "est_fee_per": est_fee_per}

    def settle(self, settled: pd.DataFrame) -> None:
        if settled.empty or not self.open:
            return
        result = dict(zip(settled["ticker"], settled["result"], strict=True))
        closes = dict(zip(settled["ticker"], settled["close_ts"], strict=True))
        updates = []
        for decision_id, d in list(self.open.items()):
            if d["ticker"] not in result:
                continue
            payout = int(bool(result[d["ticker"]]) == (d["side"] == "yes"))
            net = pnl = None
            if d["action"] == "filled":
                net = payout - d["fill_price"] - d["fee"] / d["contracts"]
                pnl = d["contracts"] * net
            elif d["action"] == "skipped":
                net = payout - d["price"] - d["est_fee_per"]  # what the blocked order would have made at the decision price
            updates.append((closes[d["ticker"]], bool(result[d["ticker"]]), payout, net, pnl, decision_id))
            del self.open[decision_id]
        if updates:
            with self.conn.cursor() as cur:
                cur.executemany("update paper.decisions set settled_at = %s, outcome = %s, payout = %s, net = %s, pnl_usd = %s where decision_id = %s", updates)

    def flush_predictions(self) -> None:
        if self.predictions:
            df = pd.concat(self.predictions, ignore_index=True).rename(columns={"ts": "decided_at"}).assign(run_id=self.run_id)
            copy_frame(self.conn, "paper.predictions", df, ["run_id", "ticker", "decided_at", "asset", "tau_min", "p_model", "mid"])
            self.predictions = []


def run_paper(conn: psycopg.Connection, run_id: int | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_research_config()
    info = meta(conn)
    run_id = run_id or conn.execute("select max(run_id) from research.runs").fetchone()[0]
    if run_id is None:
        raise RuntimeError("no research run yet; run `edgelab research` first")
    fr = conn.execute("select params, avoid, model, rule_hash from research.frozen_rules where run_id = %s", (run_id,)).fetchone()
    frozen = {"params": fr[0], "avoid": fr[1], "model": fr[2]}
    start, end = pd.Timestamp(info["research_cutoff"]), pd.Timestamp(info["end"])
    conn.execute("delete from paper.decisions where run_id = %s", (run_id,))
    conn.execute("delete from paper.predictions where run_id = %s", (run_id,))
    conn.execute("delete from research.calibration where run_id = %s and period = 'paper'", (run_id,))
    releases = read_frame(conn, "select ts, name from market.releases order by ts", ["ts"])
    feed = ReplayFeed(conn, start, end, cfg["execution"]["latency_s"])
    trader = PaperTrader(conn, run_id, frozen, cfg, releases)
    for now_s in feed.clock():
        trader.step(feed, now_s)
        if now_s % 86_400 == 0:
            trader.flush_predictions()
            conn.commit()
    trader.decide_pending(feed, int(end.timestamp()))
    trader.flush_predictions()
    conn.execute(
        "update paper.predictions p set outcome = c.result from market.contracts c where c.ticker = p.ticker and p.run_id = %s and c.close_ts <= %s",
        (run_id, end),
    )
    conn.commit()
    summary = paper_summary(conn, run_id, cfg)
    summary["rule_hash"] = fr[3]
    conn.execute("update research.runs set summary = summary || jsonb_build_object('paper', %s::jsonb) where run_id = %s", (json.dumps(summary), run_id))
    conn.commit()
    return {"run_id": run_id, "finished_at": datetime.now(UTC).isoformat(timespec="seconds"), **summary}


def paper_summary(conn: psycopg.Connection, run_id: int, cfg: dict) -> dict:
    inf = cfg["inference"]
    d = read_frame(conn, f"select * from paper.decisions where run_id = {int(run_id)} order by decided_at", ["decided_at", "settled_at"])
    p = read_frame(conn, f"select * from paper.predictions where run_id = {int(run_id)} and outcome is not null", ["decided_at"])
    out = {"signals": int(len(d)), "filled": 0, "missed": 0, "skipped": 0}
    if len(d):
        out.update({k: int(v) for k, v in d["action"].value_counts().items()})
        d["cluster"] = d["ticker"].str.rsplit("-", n=1).str[0]
        filled = d[(d["action"] == "filled") & d["net"].notna()]
        est = cluster_bootstrap(filled["net"], filled["contracts"], filled["cluster"], inf["bootstrap_resamples"], inf["bootstrap_seed"])
        out.update(
            net=est.mean, net_ci=[est.ci_lo, est.ci_hi], p_value=est.p_value, asset_hours=est.n_clusters,
            pnl_usd=float(filled["pnl_usd"].sum()), contracts=int(filled["contracts"].sum()),
            hit_rate=float((filled["payout"] * filled["contracts"]).sum() / max(filled["contracts"].sum(), 1)),
        )
        skipped = d[(d["action"] == "skipped") & d["net"].notna()]
        if len(skipped):
            out["skipped_net"] = float(skipped["net"].mean())
    contested = p[(p["mid"] >= CONTESTED[0]) & (p["mid"] <= CONTESTED[1])]
    if len(contested):
        out["brier_model"] = brier(contested["p_model"], contested["outcome"])
        out["brier_market"] = brier(contested["mid"], contested["outcome"])
        rows = []
        for source, col in (("model", "p_model"), ("market", "mid")):
            rows.append(reliability(contested[col], contested["outcome"]).assign(source=source, period="paper", run_id=run_id))
        copy_frame(conn, "research.calibration", pd.concat(rows), ["run_id", "period", "source", "bin", "p_mean", "freq", "n"])
    out["predictions"] = int(len(p))
    return out
