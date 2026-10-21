import altair as alt
import pandas as pd
import streamlit as st

from common import BLUE, ORANGE, VERDICT_ICON, cents, md, q, require_run, usd

run = require_run()
s, meta = run["summary"], run["meta"]
paper = s.get("paper") or {}
bt = s.get("backtest") or {}
frozen = s.get("frozen_hypothesis")

st.title("Short-dated crypto contracts: research and paper trading")
st.caption(
    md(
        f"Hourly BTC and ETH above/below ladders, {meta['strikes_per_ladder']} strikes each, decisions {len(meta['decision_offsets_min'])} times an hour. "
        f"Walk-forward over {s['folds']} test weeks ending {meta['research_cutoff'][:10]}; paper trading from then to {meta['end'][:10]}."
    )
)

h = q("select hypothesis_id, name, verdict from research.hypotheses where run_id = %s order by hypothesis_id", (run["run_id"],))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hypotheses registered", s["hypotheses"], help="Written down before the run looked at outcomes, and tested exactly as written.")
c2.metric("Survived costs and correction", len(s["survivors"]), help="Net of spread, latency and fee, with a Benjamini-Hochberg q-value of 0.10 or less.")
c3.metric(
    f"Walk-forward net, {frozen or 'none'}",
    cents(bt.get("net")) if bt else "n/a",
    help="Per contract, across every out-of-sample trade.",
)
c4.metric(
    f"Paper net, {frozen or 'none'}",
    cents(paper.get("net")) if paper else "not run",
    delta=f"{paper.get('filled', 0)} fills over {paper.get('asset_hours', 0)} asset-hours" if paper else None,
    delta_color="off",
    help="Per contract, filled orders only, settled against the 60-second average.",
)

if frozen:
    bt_trades = q(
        "select decided_at as ts, pnl_usd from research.trades where run_id = %s and hypothesis_id = %s and not excluded order by decided_at",
        (run["run_id"], frozen),
    ).assign(phase="Walk-forward test")
    pp = q(
        "select decided_at as ts, pnl_usd from paper.decisions where run_id = %s and action = 'filled' and pnl_usd is not null order by decided_at",
        (run["run_id"],),
    ).assign(phase="Paper trading")
    curve = pd.concat([bt_trades, pp], ignore_index=True)
    if not curve.empty:
        curve["cumulative"] = curve["pnl_usd"].cumsum()
        st.subheader(f"Cumulative P&L of the surviving rule, {frozen}")
        st.caption(md(f"Orders of up to 25 contracts, capped at displayed size. The rule and model were frozen at {meta['research_cutoff'][:10]}."))
        line = (
            alt.Chart(curve)
            .mark_line(strokeWidth=2, interpolate="step-after")
            .encode(
                x=alt.X("ts:T", title=None),
                y=alt.Y("cumulative:Q", title="Cumulative P&L (USD)"),
                color=alt.Color("phase:N", scale=alt.Scale(domain=["Walk-forward test", "Paper trading"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top-left")),
                tooltip=[alt.Tooltip("ts:T", title="Decided", format="%Y-%m-%d %H:%M"), alt.Tooltip("phase:N"), alt.Tooltip("cumulative:Q", title="Cumulative USD", format=",.0f")],
            )
        )
        cutoff = alt.Chart(pd.DataFrame({"ts": [pd.Timestamp(meta["research_cutoff"])]})).mark_rule(strokeDash=[4, 4], color="#8a8984").encode(x="ts:T")
        st.altair_chart((line + cutoff).properties(height=320), use_container_width=True)
        a, b = st.columns(2)
        a.markdown(
            md(
                f"**Walk-forward:** {bt.get('trades', 0)} trades over {bt.get('asset_hours', 0)} independent asset-hours, "
                f"{cents(bt.get('net'))} per contract (95% CI {cents(bt['net_ci'][0])} to {cents(bt['net_ci'][1])}), {usd(bt.get('pnl_usd'))} in total."
            )
        )
        if paper:
            b.markdown(
                md(
                    f"**Paper:** {paper.get('filled', 0)} fills, {paper.get('missed', 0)} missed, {paper.get('skipped', 0)} skipped by the no-trade filter. "
                    f"{cents(paper.get('net'))} per contract (95% CI {cents(paper['net_ci'][0])} to {cents(paper['net_ci'][1])}), {usd(paper.get('pnl_usd'))} in total."
                )
            )

st.subheader("What happened to each idea")
counts = h["verdict"].value_counts()
cols = st.columns(4)
for col, verdict in zip(cols, ["survives", "edge erased by costs", "no edge", "too few to tell"], strict=True):
    names = ", ".join(h.loc[h["verdict"] == verdict, "hypothesis_id"])
    col.metric(f"{VERDICT_ICON[verdict]} {verdict.capitalize()}", int(counts.get(verdict, 0)), delta=names or "none", delta_color="off")
