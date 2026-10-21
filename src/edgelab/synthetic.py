"""The synthetic market the research runs against.

A 1-second reference index for BTC and ETH, hourly above/below contracts on a five-strike ladder
that settle on the 60-second average of the index before the hour, a calendar of scheduled
releases, and one market maker quoting every contract.

The index is a martingale. Volatility switches between calm, normal and volatile regimes, rises
in US hours and after releases, and jumps now and then, but direction is never predictable, so
every trend, range and breakout rule should find nothing. The maker is good: it prices off the
true volatility, blended a little with a six-hour average, and it is about as well calibrated
as the research model. It has three known weaknesses, so the research has something real to
find and plenty to reject:

* during a burst (a one-minute move of two true standard deviations or more) its updates queue
  up and it quotes off an index print about 45 seconds old; in the paper period a faster
  competitor halves that, the way an edge decays once others find it;
* its volatility blend lags when volatility expands;
* it shades every price 2.5% toward 50 cents (the favourite-longshot bias).

It widens its spread after fast moves, in volatile regimes and around scheduled releases. It
watches releases closely, so there it neither lags nor misjudges volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

YEAR_SECONDS = 365 * 86_400
DECISION_OFFSETS_MIN = (50, 45, 40, 35, 30, 25, 20, 15, 10, 5)
STRIKE_OFFSETS = (-2, -1, 0, 1, 2)
SETTLE_WINDOW_S = 60
RELEASE_WINDOW_MIN = (15, 30)  # the maker treats 15 minutes before to 30 after a release as one window


@dataclass(frozen=True)
class AssetSpec:
    start_price: float
    vols: tuple[float, float, float]  # annualized, in the calm, normal and volatile regimes
    strike_step: float
    jump_sd: float  # unscheduled jump, log terms
    release_jump_sd: float  # jump at a scheduled release, log terms


ASSETS = {
    "BTC": AssetSpec(64_000.0, (0.30, 0.48, 0.90), 250.0, 0.0025, 0.006),
    "ETH": AssetSpec(2_450.0, (0.40, 0.62, 1.15), 15.0, 0.0032, 0.008),
}
ETH_BTC_CORRELATION = 0.8


@dataclass(frozen=True)
class MakerSpec:
    base_lag_mean_s: float = 2.0  # normal conditions: the maker reprices within a couple of seconds
    burst_lag_mean_s: float = 45.0  # after a fast move its updates queue up behind the burst
    burst_lag_mean_late_s: float = 22.0  # paper period: a faster competitor halves that
    release_lag_mean_s: float = 0.0  # around scheduled releases it is watching the index print by print
    burst_z: float = 2.0  # a one-minute move of this many true standard deviations is a burst
    max_lag_s: int = 90
    true_vol_weight: float = 0.8  # maker variance = true^0.8 x (six-hour average)^0.2
    vol_half_life_min: float = 360.0
    longshot_shrink: float = 0.025
    quote_noise: float = 0.005
    half_spread: float = 0.01
    fast_extra: float = 0.01
    volatile_extra: float = 0.01
    release_extra: float = 0.02
    latency_s: int = 2


@dataclass
class SyntheticMarket:
    bars: pd.DataFrame
    contracts: pd.DataFrame
    quotes: pd.DataFrame
    releases: pd.DataFrame
    meta: dict


def _regimes(rng: np.random.Generator, minutes: int) -> np.ndarray:
    """Market-wide volatility regime per minute: 0 calm, 1 normal, 2 volatile."""
    mean_len = np.array([360, 480, 90])
    nxt = np.array([[0.0, 0.8, 0.2], [0.6, 0.0, 0.4], [0.2, 0.8, 0.0]])
    out = np.empty(minutes, dtype=np.int8)
    i, state = 0, 1
    while i < minutes:
        n = int(rng.geometric(1 / mean_len[state]))
        out[i : i + n] = state
        i += n
        state = int(rng.choice(3, p=nxt[state]))
    return out


def release_calendar(start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Two scheduled releases a week, Tuesday and Thursday at 12:30 UTC."""
    rows = []
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        if day.weekday() in (1, 3):
            rows.append((day + pd.Timedelta(hours=12, minutes=30), "Scheduled macro release"))
    return pd.DataFrame(rows, columns=["ts", "name"])


def in_release_window(ts_s: np.ndarray, release_s: np.ndarray) -> np.ndarray:
    """True where a time (seconds) falls 15 minutes before to 30 minutes after a release."""
    before, after = RELEASE_WINDOW_MIN[0] * 60, RELEASE_WINDOW_MIN[1] * 60
    if len(release_s) == 0:
        return np.zeros(len(ts_s), dtype=bool)
    nxt = np.searchsorted(release_s, ts_s, side="left")  # first release at or after t
    prv = nxt - 1
    upcoming = (nxt < len(release_s)) & (release_s[np.minimum(nxt, len(release_s) - 1)] - ts_s <= before)
    recent = (prv >= 0) & (ts_s - release_s[np.maximum(prv, 0)] <= after)
    return upcoming | recent


def _quote(spot, tau_s, var_m, strikes, half, noise, shrink) -> tuple[np.ndarray, np.ndarray]:
    """The maker's YES bid and ask in cents: a normal probability on its own volatility, shaded toward 50."""
    v = (var_m / 60.0) * (tau_s - SETTLE_WINDOW_S + SETTLE_WINDOW_S / 3.0)
    pm = norm.sf(np.log(strikes / spot[:, None]) / np.sqrt(v)[:, None])
    pq = 0.5 + (pm - 0.5) * (1 - shrink) + noise
    bid = np.clip(np.round(100 * (pq - half[:, None])), 1, 98).astype(np.int16)
    ask = np.clip(np.round(100 * (pq + half[:, None])), bid + 1, 99).astype(np.int16)
    return bid, ask


def generate(
    start: str | pd.Timestamp = "2026-05-04",
    research_days: int = 112,
    paper_days: int = 42,
    seed: int = 20260518,
    maker: MakerSpec | None = None,
) -> SyntheticMarket:
    maker = maker or MakerSpec()
    start = pd.Timestamp(start, tz="UTC")
    days = research_days + paper_days
    minutes, hours = days * 1440, days * 24
    n_sec = minutes * 60
    paper_start_s = research_days * 86_400
    rng = np.random.default_rng(seed)

    releases = release_calendar(start, days)
    release_s = ((releases["ts"] - start).dt.total_seconds().astype(np.int64)).to_numpy()

    regime = _regimes(rng, minutes)
    minute_idx = np.arange(minutes)
    hour_of_day = (minute_idx // 60) % 24
    weekday = (start.weekday() + minute_idx // 1440) % 7
    mult = np.where((hour_of_day >= 13) & (hour_of_day < 21), 1.25, 1.0) * np.where(weekday >= 5, 0.75, 1.0)
    for r in release_s // 60:
        regime[r : r + 45] = 2
        mult[r : r + 15] *= 1.5

    n_jumps = int(rng.poisson(1.5 * days))
    jump_at = np.sort(rng.integers(0, n_sec, size=n_jumps))
    jump_z = rng.standard_normal(n_jumps)
    release_z = rng.standard_normal(len(release_s))
    z_common = rng.standard_normal(n_sec)

    bars, contracts, quotes = [], [], []
    hour_open_s = np.arange(hours, dtype=np.int64) * 3600
    hour_close_s = hour_open_s + 3600
    for k, (asset, spec) in enumerate(ASSETS.items()):
        sig = np.asarray(spec.vols)[regime] * mult / np.sqrt(YEAR_SECONDS)
        r = np.repeat(sig, 60)
        if k == 0:
            r *= z_common
        else:
            z = rng.standard_normal(n_sec)
            z *= np.sqrt(1 - ETH_BTC_CORRELATION**2)
            z += ETH_BTC_CORRELATION * z_common
            r *= z
            del z
        np.add.at(r, jump_at, jump_z * spec.jump_sd)
        np.add.at(r, release_s - 1, release_z * spec.release_jump_sd)
        logp = np.empty(n_sec + 1)
        logp[0] = 0.0
        np.cumsum(r, out=logp[1:])
        del r
        p = spec.start_price * np.exp(logp)
        del logp

        # 1-minute bars labelled by their close time; bar m covers seconds (60m, 60(m+1)]
        seg = p[1:].reshape(minutes, 60)
        open_ = p[0:n_sec:60]
        close = seg[:, -1].copy()
        bars.append(
            pd.DataFrame(
                {
                    "asset": asset,
                    "ts": start + pd.to_timedelta((minute_idx + 1) * 60, unit="s"),
                    "open": open_,
                    "high": np.maximum(seg.max(axis=1), open_),
                    "low": np.minimum(seg.min(axis=1), open_),
                    "close": close,
                }
            )
        )

        # Hourly ladders: five strikes around the hour's opening print
        ref = p[hour_open_s]
        center = np.round(ref / spec.strike_step) * spec.strike_step
        strikes = center[:, None] + spec.strike_step * np.asarray(STRIKE_OFFSETS)[None, :]
        settle_idx = hour_close_s[:, None] + np.arange(-SETTLE_WINDOW_S + 1, 1)[None, :]
        settle_avg = p[settle_idx].mean(axis=1)
        close_ts = start + pd.to_timedelta(hour_close_s, unit="s")
        tickers = np.array(
            [
                [f"{asset}-{ts:%y%m%d}-{ts:%H}Z-K{s:.0f}" for s in row]
                for ts, row in zip(close_ts, strikes, strict=True)
            ]
        )
        contracts.append(
            pd.DataFrame(
                {
                    "ticker": tickers.ravel(),
                    "asset": asset,
                    "open_ts": np.repeat(start + pd.to_timedelta(hour_open_s, unit="s"), 5),
                    "close_ts": np.repeat(close_ts, 5),
                    "strike": strikes.ravel(),
                    "settle_avg": np.repeat(settle_avg, 5),
                    "result": (settle_avg[:, None] >= strikes).ravel(),
                }
            )
        )

        # The maker's quote at each decision time, and again latency_s seconds later
        offsets = np.asarray(DECISION_OFFSETS_MIN) * 60
        t = (hour_close_s[:, None] - offsets[None, :]).ravel()
        tau = np.tile(offsets, hours)
        hour_of = np.repeat(np.arange(hours), len(offsets))
        released = in_release_window(t, release_s)
        var_true = sig**2 * 60  # true variance per minute
        burst = np.abs(np.log(p[t] / p[t - 60])) > maker.burst_z * np.sqrt(var_true[t // 60 - 1])
        lag_mean = np.where(
            burst, np.where(t >= paper_start_s, maker.burst_lag_mean_late_s, maker.burst_lag_mean_s), maker.base_lag_mean_s
        )
        lag_mean = np.where(released, maker.release_lag_mean_s, lag_mean)
        lag = np.minimum(np.round(rng.exponential(lag_mean)), maker.max_lag_s).astype(np.int64)

        r_min = np.diff(np.log(close), prepend=np.log(spec.start_price))
        var_avg = pd.Series(r_min**2).ewm(halflife=maker.vol_half_life_min, adjust=False).mean().to_numpy()
        seen = (t - lag) // 60 - 1  # last bar the maker has fully seen
        w = np.where(released, 1.0, maker.true_vol_weight)  # around releases it prices off the true volatility
        var_m = var_true[seen] ** w * var_avg[seen] ** (1 - w)
        fast = np.abs(r_min[seen]) > 2.5 * np.sqrt(var_m)
        volatile = var_true[seen] > 1.5 * var_avg[seen]
        half = (
            maker.half_spread
            + maker.fast_extra * fast
            + maker.volatile_extra * volatile
            + maker.release_extra * released
        )

        k_strikes = strikes[hour_of]  # (snapshots, 5)
        noise = rng.normal(0.0, maker.quote_noise, size=k_strikes.shape)

        bid, ask = _quote(p[t - lag], tau.astype(float), var_m, k_strikes, half, noise, maker.longshot_shrink)
        fill_bid, fill_ask = _quote(p[t + maker.latency_s - lag], (tau - maker.latency_s).astype(float), var_m, k_strikes, half, noise, maker.longshot_shrink)
        thin = np.where(fast | released, 0.4, 1.0)[:, None]
        bid_size = np.maximum(5, np.round(rng.lognormal(np.log(120), 0.6, size=k_strikes.shape) * thin)).astype(int)
        ask_size = np.maximum(5, np.round(rng.lognormal(np.log(120), 0.6, size=k_strikes.shape) * thin)).astype(int)
        ts = start + pd.to_timedelta(np.repeat(t, 5), unit="s")
        quotes.append(
            pd.DataFrame(
                {
                    "ticker": tickers[hour_of].ravel(),
                    "ts": ts,
                    "yes_bid": bid.ravel(),
                    "yes_ask": ask.ravel(),
                    "bid_size": bid_size.ravel(),
                    "ask_size": ask_size.ravel(),
                    "fill_ts": ts + pd.Timedelta(seconds=maker.latency_s),
                    "fill_yes_bid": fill_bid.ravel(),
                    "fill_yes_ask": fill_ask.ravel(),
                }
            )
        )
        del p, seg

    meta = {
        "seed": seed,
        "start": start.isoformat(),
        "research_cutoff": (start + pd.Timedelta(days=research_days)).isoformat(),
        "end": (start + pd.Timedelta(days=days)).isoformat(),
        "research_days": research_days,
        "paper_days": paper_days,
        "assets": list(ASSETS),
        "decision_offsets_min": list(DECISION_OFFSETS_MIN),
        "strikes_per_ladder": len(STRIKE_OFFSETS),
        "latency_s": maker.latency_s,
    }
    return SyntheticMarket(
        bars=pd.concat(bars, ignore_index=True),
        contracts=pd.concat(contracts, ignore_index=True),
        quotes=pd.concat(quotes, ignore_index=True),
        releases=releases,
        meta=meta,
    )
