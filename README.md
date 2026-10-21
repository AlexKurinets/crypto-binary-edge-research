# Crypto binary edge research

A research and paper-trading pipeline for short-dated crypto event contracts: hourly BTC and ETH above/below ladders that settle on the 60-second average of a reference index. It prices every strike with a settlement-aware probability model, tests a registry of trading hypotheses walk-forward against the quote it would really have hit and the fee it would really have paid, judges them with intervals that count independent hours rather than trades, freezes whatever survives, and paper-trades that rule forward in a log that cannot be edited after the fact. A Streamlit dashboard shows calibration, the hypothesis scorecard, the conditions where a rule should not trade, and the paper log.

This is a reference implementation of the architecture behind a platform I built for a small crypto-focused proprietary trading firm. The production build is under NDA, so a few integrations are abstracted behind adapters: the order-book recorder, the index feed and the paper trader's live connections. It runs on synthetic data, and every company, place, person, phone number and figure in it is made up.

[![ci](https://github.com/AlexKurinets/crypto-binary-edge-research/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexKurinets/crypto-binary-edge-research/actions/workflows/ci.yml)

## Architecture

```mermaid
flowchart LR
  subgraph Sources["Recorded market (adapters)"]
    IDX["Index: 1-minute bars"]
    BOOK["Quotes at each decision, and 2 s later"]
    SET["Settlements: 60 s average"]
    REL["Release calendar"]
  end
  REG[("config/hypotheses.yml\nregistered rules")]
  subgraph Research["edgelab research"]
    FEAT["Point-in-time features"]
    MOD["HAR-RV + settlement-aware\nStudent-t + recalibration"]
    WF["Purged walk-forward\nweekly refit, 60 min embargo"]
    COST["Fill at quote + 2 s,\nIOC limit, size cap, taker fee"]
    INF["Cluster bootstrap by asset-hour,\nBenjamini-Hochberg"]
    FRZ["Freeze survivor, filters, model"]
  end
  subgraph PG["Postgres 16"]
    MKT[("market")]
    RES[("research")]
    PAP[("paper: append-only\ndecisions")]
  end
  FEED["ReplayFeed: bars, snapshot,\nsend, settled"]
  TRADER["edgelab paper"]
  DASH["Streamlit dashboard"]
  Sources --> MKT
  MKT --> FEAT --> MOD --> WF --> COST --> INF --> FRZ
  REG --> COST
  FRZ --> RES
  MKT --> FEED --> TRADER
  RES --> TRADER --> PAP
  RES --> DASH
  PAP --> DASH
```

The synthetic market runs from 4 May to 5 October 2026: two assets, five strikes around each hour's opening print, decisions 50, 45, ... 5 minutes before each close (369,600 quote snapshots), and two scheduled releases a week. The first 16 weeks are research history and the last 6 are the paper period.

## What the run finds

These are the figures the Compose stack produces from the synthetic market.

| | |
|---|---|
| Walk-forward test | 12 test weeks, 3,984 independent asset-hours, 199,200 out-of-sample strike snapshots |
| Calibration, contested snapshots (10 to 90 cents) | Brier 0.1757 for the model, 0.1750 for the market mid |
| Hypotheses registered | 12: one survives, two have gross edge that costs erase, eight have none, one trades too rarely to judge |
| Survivor, H02 stale quotes after a fast move | +6.2 cents a contract net (95% CI +3.3 to +9.0), 879 trades, 781 asset-hours, q = 0.006 |
| Where its gross 8.7 cents goes | 1.2 cents of half-spread, 0.0 of latency, 1.3 of fee |
| Paper period, 6 weeks, frozen rule | 444 signals: 333 filled, 87 missed, 24 skipped by the registered filter; +4.3 cents a contract (95% CI +0.05 to +8.4), $354 on 8,319 contracts |
| Signals the filter blocked | would have lost 7.3 cents a contract at their decision price |

The synthetic market is a known-answer test. Its market maker is about as well calibrated as the model, but it plants three weaknesses (stale quotes during bursts, a volatility estimate that lags expansions, and a small favourite-longshot bias), and its index has no predictable direction. A sound pipeline should find the first, see the other two eaten by costs, and reject every trend, range and breakout rule. It does.

## Design decisions

**Probabilities that know how the contract settles.** A contract settles on the average of the index over its final 60 seconds, and for a random walk that average carries the variance of only 20 seconds of path. [src/edgelab/model.py](src/edgelab/model.py) counts the risk left as `tau - 60 + 60/3` seconds, scales the log distance to the strike by the forecast volatility, and reads the probability off a Student-t with five degrees of freedom rescaled to the same variance. Without the averaging term, contracts near expiry look mispriced when they are not.

**Volatility from three horizons, refitted every week.** Next-hour variance is a HAR-RV regression on realized variance over the last 15 minutes, hour and day. A two-parameter logistic fit of the outcome on `logit(p)` then corrects any steady over- or under-confidence. [src/edgelab/research.py](src/edgelab/research.py) predicts each test week with a model fitted only on the four weeks before it, and drops training rows whose 60-minute target would reach into the test week.

**Features that cannot see the future, computed the same way live.** [src/edgelab/features.py](src/edgelab/features.py) has a vectorized version for research and a scalar version that runs on the paper trader's buffer. One test holds them to the same numbers, and another changes every bar after a decision time and checks the features at that time do not move.

**Hypotheses are registered before the data is scored.** [config/hypotheses.yml](config/hypotheses.yml) fixes every rule and threshold, and the run tests them exactly as written. Each run stores the registry's hash, and `research.registrations` counts how many versions the same data has seen, because every edit is another look.

**Costs charged the way the exchange charges them.** [src/edgelab/costs.py](src/edgelab/costs.py) buys at the ask, or at 100 minus the YES bid for NO. The order reaches the book two seconds later and is matched against that quote with an immediate-or-cancel limit one cent through the decision price, so a price that moved counts as a miss, not a worse fill. Size is capped at what was displayed. The fee is the published taker schedule, `round_up(0.07 x C x P x (1 - P))` per order, which peaks at 1.75 cents a contract at 50 cents, so a contract bought at 51 cents must win 52.8% of the time to break even. Gross edge, spread, latency and fee are stored separately and add up exactly to the net.

**Intervals that count hours, not trades.** Every strike and every decision in one asset-hour settles off the same average, so they are one outcome seen several times. [src/edgelab/stats.py](src/edgelab/stats.py) resamples whole asset-hours, and a test shows that repeating a trade inside its hour leaves the interval unchanged where a naive bootstrap would shrink it. The twelve p-values are adjusted together with Benjamini-Hochberg, and a hypothesis survives at q of 0.10 or less with a positive net.

**How much data an edge needs.** A contract near 50 cents has about 50 cents of standard deviation per outcome, so `edgelab power` and the dashboard show the smallest edge a sample can detect: about 2.3 cents for a month of 15-minute windows, and 3,865 independent outcomes for 2 cents.

**No-trade filters are either registered or earned.** [src/edgelab/conditions.py](src/edgelab/conditions.py) splits a rule's result by spread, minutes to close, contract price, regime, release window and asset. A bucket earns "avoid" only when its whole 95% interval is below zero over at least 30 asset-hours. Two releases a week are too few to learn from, so H02's release-window filter is registered with the hypothesis instead, and the paper period settles the signals it blocked so the filter is tested anyway.

**The paper trader cannot see the future or rewrite the past.** [src/edgelab/paper.py](src/edgelab/paper.py) talks to a feed with the four calls the live adapters had: bars up to now, the quote snapshot at a decision time, the quote an order meets after the latency, and settlements up to now. `ReplayFeed` refuses a fill quote before the order could have arrived. Orders for an hour are decided together before that hour settles, with the same `signals` and `fill` code as the backtest, and a test checks the paper trader sends exactly the orders the research path would. A trigger in [sql/001_schema.sql](sql/001_schema.sql) refuses any change to a logged decision and any second settlement.

## Running it

```bash
cp .env.example .env              # optional: change the host ports
docker compose up --build -d --wait
```

The `pipeline` service migrates, seeds the synthetic market, runs the walk-forward research and the six-week paper period, and exits; the first run takes about four minutes. Then open the dashboard at http://localhost:8501 (or your `EDGELAB_DASHBOARD_PORT`).

```bash
docker compose run --rm pipeline edgelab status                # dataset and the latest run's summary
docker compose run --rm pipeline edgelab report                # the findings as Markdown
docker compose run --rm pipeline edgelab power --edge 0.02     # outcomes needed for a 2-cent edge, fee and breakeven at 50 cents
```

To see the paper log refuse an edit:

```bash
docker compose exec db psql -U edgelab -d edgelab -c "update paper.decisions set p_model = 0.99 where decision_id = 1"
# ERROR:  paper decision 1 was logged at ... and cannot be changed
```

Running `docker compose up` again reuses the data and the run. `docker compose down -v` starts from scratch.

### Without Docker

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
export EDGELAB_DSN=postgresql://edgelab:edgelab@localhost:5432/edgelab
edgelab migrate && edgelab seed && edgelab research && edgelab paper
streamlit run dashboard/app.py
```

`edgelab seed --research-days 36 --paper-days 3` builds a small market in seconds; one walk-forward week is too little for anything to survive, which is the point of the power calculation.

## Tests

```bash
pytest          # needs Postgres at EDGELAB_TEST_DSN or EDGELAB_DSN; the suite drops and rebuilds the schemas
ruff check .
```

The suite runs against real Postgres on a small synthetic market. It covers the fee schedule and its rounding, NO pricing, the cost decomposition, misses past the limit and the size cap; the averaging term, the Student-t scaling against simulation, recalibration and HAR recovering known coefficients; lookahead and the live-versus-research feature match; the cluster bootstrap, Benjamini-Hochberg and the power figures; walk-forward folds never training on their test week; the paper feed refusing early fills, the trigger refusing edits, and the paper trader matching the backtest's orders; and every dashboard page rendering. CI runs the suite, the command-line path without Docker, then brings the Compose stack up and runs [scripts/smoke.sh](scripts/smoke.sh), which executes every command block in this README.

## Layout

```
config/            research.yml (walk-forward, costs, inference), hypotheses.yml (the registry)
sql/               001_schema.sql: market, research and paper schemas, the append-only trigger
src/edgelab/       synthetic, features, model, costs, hypotheses, stats, conditions, research, paper, report, cli
dashboard/         Streamlit app, one file per page under views/
scripts/smoke.sh   every README sequence against a running stack
tests/             pytest against Postgres, plus Streamlit AppTest for each page
```

## What is deliberately left out

- **Live connectors.** The production recorder pulled order books from the exchange and index prints from its constituent venues. Here the synthetic generator fills the same `market` tables, and `ReplayFeed` stands in for the live feed.
- **Order placement.** Nothing here sends an order; the paper trader decides, fills against the recorded quote and logs.
- **Queue position and hidden liquidity.** Fills assume the displayed size at the touch was ours to take.
- **Resting maker orders.** Every simulated order takes liquidity and pays the taker fee. A maker strategy has a different fee schedule and an adverse-selection problem this does not model.
- **Position sizing and risk limits.** Orders are a flat 25 contracts. Bankroll sizing belongs after an edge is established, not before.
- **Decisions inside the settlement window.** Pricing a contract in its final minute needs the seconds already averaged; the model refuses those inputs rather than guessing.

## License

MIT, Alex Kurinets.
