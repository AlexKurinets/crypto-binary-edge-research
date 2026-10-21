import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import CONDITION_ICON, NEGATIVE, POSITIVE, cents, md, q, require_run

run = require_run()
s = run["summary"]
h = q("select hypothesis_id, name from research.hypotheses where run_id = %s order by hypothesis_id", (run["run_id"],))
labels = list(h["hypothesis_id"] + "  " + h["name"])
frozen = s.get("frozen_hypothesis")

st.title("When not to trade")
st.caption(
    "A rule's walk-forward result split by the conditions at decision time. A bucket earns \"avoid\" only when its whole 95% "
    "interval is below zero over at least 30 independent asset-hours; merely negative is \"within noise\". Filters that the "
    "data is too thin to test are registered with the hypothesis in advance instead."
)

default = next((i for i, lab in enumerate(labels) if frozen and lab.startswith(frozen)), 0)
choice = st.selectbox("Hypothesis", labels, index=default)
hid = choice.split()[0]

if hid == frozen:
    avoid = s.get("avoid") or []
    if avoid:
        st.markdown(
            "**Frozen no-trade filters for the paper period:** "
            + "; ".join(
                f"{a['dimension']}, {a['bucket']} ({'registered in advance' if a['source'] == 'registered' else 'earned in the walk-forward test'})"
                for a in avoid
            )
        )
    else:
        st.markdown("**Frozen no-trade filters for the paper period:** none")

c = q("select * from research.conditions where run_id = %s and hypothesis_id = %s order by dimension, sort_order", (run["run_id"], hid))
if c.empty:
    st.info("This hypothesis made no trades, so there is nothing to split.")
    st.stop()
c = c.assign(net_c=c["net"] * 100, lo=c["ci_lo"] * 100, hi=c["ci_hi"] * 100, sign=np.where(c["net"] >= 0, "positive", "negative"))
c["bucket_label"] = c["bucket"] + "  " + c["verdict"].map(CONDITION_ICON)
order = ["Minutes to close", "Contract price", "Spread at decision", "Market regime", "Scheduled release", "Asset"]
cols = st.columns(2)
for i, dim in enumerate(order):
    d = c[c["dimension"] == dim]
    if d.empty:
        continue
    y = alt.Y("bucket_label:N", sort=list(d.sort_values("sort_order")["bucket_label"]), title=None, axis=alt.Axis(labelLimit=220))
    tip = [alt.Tooltip("bucket:N"), alt.Tooltip("net_c:Q", title="Net ¢", format="+.2f"), alt.Tooltip("lo:Q", title="CI low ¢", format="+.2f"),
           alt.Tooltip("hi:Q", title="CI high ¢", format="+.2f"), alt.Tooltip("n_trades:Q", title="Trades"), alt.Tooltip("n_clusters:Q", title="Asset-hours"),
           alt.Tooltip("verdict:N")]
    chart = (
        alt.Chart(d).mark_rule(strokeWidth=2, color="#8a8984").encode(y=y, x=alt.X("lo:Q", title="Net per contract (cents)"), x2="hi:Q", tooltip=tip)
        + alt.Chart(d).mark_point(size=90, filled=True).encode(
            y=y, x="net_c:Q", color=alt.Color("sign:N", scale=alt.Scale(domain=["positive", "negative"], range=[POSITIVE, NEGATIVE]), legend=None), tooltip=tip
        )
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#52514e").encode(x="x:Q")
    )
    with cols[i % 2]:
        st.markdown(f"**{dim}**")
        st.altair_chart(chart.properties(height=48 * len(d) + 50), use_container_width=True)

st.caption("✅ keep · ⛔ avoid · ➖ within noise · ❔ too few to tell")

if hid == frozen:
    p = q(
        "select action, count(*) as n, avg(net) as net from paper.decisions where run_id = %s and outcome is not null group by action order by action",
        (run["run_id"],),
    )
    if not p.empty:
        st.subheader("Did the filters pay for themselves in the paper period?")
        lines = []
        for r in p.itertuples():
            if r.action == "filled":
                lines.append(f"{r.n} filled orders made {cents(r.net)} per contract on average (unweighted by size).")
            elif r.action == "skipped":
                lines.append(f"{r.n} signals blocked by the filters would have made {cents(r.net)} per contract at their decision price.")
        st.markdown(md(" ".join(lines)))
