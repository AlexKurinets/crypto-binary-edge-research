import altair as alt
import streamlit as st

from common import BLUE, cents, md, q, require_run, usd

run = require_run()
s = run["summary"]
paper = s.get("paper") or {}
meta = run["meta"]

st.title("Paper log")
st.caption(
    md(
        f"Every signal of the frozen rule ({s.get('frozen_hypothesis') or 'none'}) from {meta['research_cutoff'][:10]} to {meta['end'][:10]}, "
        "written when it was decided and settled later against the 60-second average. A database trigger refuses any change to a "
        "decision once it is logged, so the log cannot be tidied after the fact."
    )
)
if not paper:
    st.info("The paper period has not been run yet. Run `edgelab paper`.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Signals", paper["signals"])
c2.metric("Filled", paper["filled"], help="Immediate-or-cancel at most one cent through the decision price, matched against the quote two seconds later.")
c3.metric("Missed", paper["missed"], help="The price moved past the limit before the order arrived.")
c4.metric("Skipped by filters", paper["skipped"])
c5.metric("Net per contract", cents(paper.get("net")), delta=f"95% CI {cents(paper['net_ci'][0])} to {cents(paper['net_ci'][1])}", delta_color="off")

d = q("select * from paper.decisions where run_id = %s order by decided_at", (run["run_id"],))
filled = d[(d["action"] == "filled") & d["pnl_usd"].notna()].copy()
if not filled.empty:
    filled["cumulative"] = filled["pnl_usd"].cumsum()
    st.subheader(md(f"Cumulative paper P&L: {usd(paper.get('pnl_usd'))} on {paper.get('contracts', 0):,} contracts"))
    st.altair_chart(
        alt.Chart(filled)
        .mark_line(strokeWidth=2, color=BLUE, interpolate="step-after")
        .encode(
            x=alt.X("decided_at:T", title=None),
            y=alt.Y("cumulative:Q", title="Cumulative P&L (USD)"),
            tooltip=[alt.Tooltip("decided_at:T", title="Decided", format="%Y-%m-%d %H:%M"), alt.Tooltip("ticker:N"), alt.Tooltip("cumulative:Q", title="USD", format=",.0f")],
        )
        .properties(height=280),
        use_container_width=True,
    )

st.subheader("Decisions")
actions = st.multiselect("Show", ["filled", "missed", "skipped"], default=["filled", "missed", "skipped"])
view = d[d["action"].isin(actions)].sort_values("decided_at", ascending=False)
st.dataframe(
    view[["decided_at", "ticker", "tau_min", "regime", "p_model", "side", "price", "edge", "action", "fill_price", "contracts", "fee",
          "outcome", "net", "pnl_usd", "skip_reason"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "decided_at": st.column_config.DatetimeColumn("decided (UTC)", format="YYYY-MM-DD HH:mm"),
        "tau_min": st.column_config.NumberColumn("min to close"),
        "p_model": st.column_config.NumberColumn("model P(yes)", format="%.3f"),
        "price": st.column_config.NumberColumn("price", format="%.2f"),
        "edge": st.column_config.NumberColumn("edge after fee", format="%.3f"),
        "fill_price": st.column_config.NumberColumn("fill", format="%.2f"),
        "fee": st.column_config.NumberColumn("fee", format="%.2f"),
        "net": st.column_config.NumberColumn("net per contract", format="%.3f"),
        "pnl_usd": st.column_config.NumberColumn("P&L", format="%.2f"),
    },
)
st.caption(f"{len(view):,} rows. Prices and the edge are in dollars per contract. Skipped rows are settled at their decision price so the filter can be judged.")
