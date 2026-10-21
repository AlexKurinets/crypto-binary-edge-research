"""The findings as a short Markdown report, written from the database like the dashboard."""

from __future__ import annotations

import psycopg

from edgelab.db import meta, query_df
from edgelab.stats import min_detectable_edge


def _c(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f}c"


def render(conn: psycopg.Connection, run_id: int | None = None) -> str:
    run_id = run_id or conn.execute("select max(run_id) from research.runs").fetchone()[0]
    if run_id is None:
        raise RuntimeError("no research run yet")
    info = meta(conn)
    s = conn.execute("select summary from research.runs where run_id = %s", (run_id,)).fetchone()[0]
    h = query_df(conn, "select * from research.hypotheses where run_id = %s order by hypothesis_id", (run_id,))
    c = query_df(
        conn,
        "select * from research.conditions where run_id = %s and hypothesis_id = %s order by dimension, sort_order",
        (run_id, s.get("frozen_hypothesis") or ""),
    )
    paper = s.get("paper") or {}
    lines = [
        f"# Research run {run_id}",
        "",
        "## Data",
        "",
        f"Synthetic market from {info['start'][:10]} to {info['end'][:10]}: {', '.join(info['assets'])} hourly ladders of "
        f"{info['strikes_per_ladder']} strikes, decisions at {len(info['decision_offsets_min'])} times an hour. "
        f"Research ends {info['research_cutoff'][:10]}; the {info['paper_days']} days after it are the paper period.",
        "",
        "## Method",
        "",
        f"Purged walk-forward over {s['folds']} test weeks ({s['oos_days']} days, {s['oos_asset_hours']} independent asset-hours). "
        "Each week is predicted by HAR-RV volatility and a settlement-aware Student-t probability fitted on the four weeks before it, then "
        "recalibrated. Every registered hypothesis trades the out-of-sample predictions at the decision quote, fills against the quote two "
        "seconds later, and pays the taker fee. Intervals resample whole asset-hours; q-values are Benjamini-Hochberg across all "
        f"{s['hypotheses']} hypotheses.",
        "",
        f"On contested snapshots (market between 10 and 90 cents) the model's Brier score is {s['brier_model']:.4f} and the market's is "
        f"{s['brier_market']:.4f}.",
        "",
        "## Hypotheses",
        "",
        "| id | name | trades | asset-hours | gross | spread | fee | net | 95% CI | q | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for r in h.itertuples():
        lines.append(
            f"| {r.hypothesis_id} | {r.name} | {r.n_trades} | {r.n_clusters} | {_c(r.gross)} | {_c(r.spread)} | {_c(r.fee)} | {_c(r.net)} | "
            f"{_c(r.net_ci_lo)} to {_c(r.net_ci_hi)} | {r.q_value:.3f} | {r.verdict} |"
        )
    lines += ["", "## When not to trade", ""]
    if c.empty:
        lines.append("Nothing survived, so there is no rule to filter.")
    else:
        for r in c.itertuples():
            lines.append(f"- {r.dimension}, {r.bucket}: {_c(r.net)} over {r.n_clusters} asset-hours ({r.verdict})")
        lines.append("")
        lines.append("Frozen no-trade filters: " + ("; ".join(f"{a['dimension']}: {a['bucket']} ({a['source']})" for a in s["avoid"]) or "none"))
    lines += ["", "## Paper period", ""]
    if paper:
        lines.append(
            f"{paper.get('filled', 0)} filled, {paper.get('missed', 0)} missed and {paper.get('skipped', 0)} skipped signals. "
            f"Net {_c(paper.get('net'))} per contract, 95% CI {_c((paper.get('net_ci') or [None])[0])} to {_c((paper.get('net_ci') or [None, None])[1])}, "
            f"over {paper.get('asset_hours', 0)} asset-hours, against {_c((s.get('backtest') or {}).get('net'))} in the walk-forward test."
        )
        if paper.get("asset_hours"):
            lines.append(
                f"With {paper['asset_hours']} independent outcomes the smallest edge a one-sided 5% test detects with 80% power is about "
                f"{min_detectable_edge(paper['asset_hours']) * 100:.1f} cents."
            )
    else:
        lines.append("Not run yet.")
    lines += [
        "",
        "## Limitations",
        "",
        "- Fills assume the displayed size at the touch is available to us; queue position and hidden competition are not modeled.",
        "- No-trade filters chosen on walk-forward results are fitted to that data; only the paper period tests them cleanly.",
        "- One venue, two assets, one fee schedule. A fee change moves every net figure above.",
        "",
    ]
    return "\n".join(lines)
