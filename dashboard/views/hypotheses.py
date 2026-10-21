import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import BLUE, NEGATIVE, ORANGE, POSITIVE, VERDICT_ICON, cents, md, q, require_run
from edgelab.stats import min_detectable_edge

run = require_run()
s = run["summary"]
h = q("select * from research.hypotheses where run_id = %s order by hypothesis_id", (run["run_id"],))

st.title("Hypotheses: what survived the costs")
st.caption(
    md(
        f"{len(h)} rules registered in advance, each traded on out-of-sample predictions only, filled against the quote two seconds "
        "after the decision, charged the taker fee, and judged with intervals that resample whole asset-hours. q-values are "
        "Benjamini-Hochberg across all of them."
    )
)

table = pd.DataFrame(
    {
        "": h["verdict"].map(VERDICT_ICON),
        "id": h["hypothesis_id"],
        "name": h["name"],
        "trades": h["n_trades"],
        "asset-hours": h["n_clusters"],
        "gross ¢": h["gross"] * 100,
        "spread ¢": h["spread"] * 100,
        "fee ¢": h["fee"] * 100,
        "net ¢": h["net"] * 100,
        "net 95% CI ¢": [f"{lo * 100:+.2f} to {hi * 100:+.2f}" if pd.notna(lo) else "" for lo, hi in zip(h["net_ci_lo"], h["net_ci_hi"], strict=True)],
        "q": h["q_value"],
        "verdict": h["verdict"].replace({"edge erased by costs": "erased by costs"}),
    }
)
fmt = st.column_config.NumberColumn
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    height=35 * (len(table) + 1) + 3,
    column_config={c: fmt(format="%.2f") for c in ["gross ¢", "spread ¢", "fee ¢", "net ¢"]}
    | {"q": fmt(format="%.3f"), "": st.column_config.TextColumn(width=36), "name": st.column_config.TextColumn(width=230)},
)

st.subheader("Net per contract, with 95% intervals")
plot = h[h["verdict"] != "too few to tell"].assign(
    net_c=lambda d: d["net"] * 100, lo=lambda d: d["net_ci_lo"] * 100, hi=lambda d: d["net_ci_hi"] * 100,
    sign=lambda d: np.where(d["net"] >= 0, "positive", "negative"), label=lambda d: d["hypothesis_id"] + "  " + d["name"],
)
y = alt.Y("label:N", title=None, sort=list(plot["label"]), axis=alt.Axis(labelLimit=320))
rules = alt.Chart(plot).mark_rule(strokeWidth=2, color="#8a8984").encode(y=y, x=alt.X("lo:Q", title="Net per contract (cents)"), x2="hi:Q")
dots = (
    alt.Chart(plot)
    .mark_point(size=90, filled=True)
    .encode(
        y=y,
        x="net_c:Q",
        color=alt.Color("sign:N", scale=alt.Scale(domain=["positive", "negative"], range=[POSITIVE, NEGATIVE]), legend=None),
        tooltip=[alt.Tooltip("label:N", title="Hypothesis"), alt.Tooltip("net_c:Q", title="Net ¢", format="+.2f"),
                 alt.Tooltip("lo:Q", title="CI low ¢", format="+.2f"), alt.Tooltip("hi:Q", title="CI high ¢", format="+.2f"),
                 alt.Tooltip("n_clusters:Q", title="Asset-hours"), alt.Tooltip("verdict:N")],
    )
)
zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#52514e").encode(x="x:Q")
st.altair_chart((rules + dots + zero).properties(height=34 * len(plot) + 40), use_container_width=True)
rare = h[h["verdict"] == "too few to tell"]
st.caption(
    "Rules that trade rarely have wide intervals. Not plotted: "
    + "; ".join(f"{r.hypothesis_id} traded {r.n_trades} times over {r.n_clusters} asset-hours, too few to judge" for r in rare.itertuples())
)

st.subheader("Where the gross edge goes")
survivors = np.flatnonzero(h["verdict"] == "survives")
choice = st.selectbox("Hypothesis", list(h["hypothesis_id"] + "  " + h["name"]), index=int(survivors[0]) if len(survivors) else 0)
r = h[h["hypothesis_id"] == choice.split()[0]].iloc[0]
steps = pd.DataFrame(
    {
        "step": ["Gross, against the mid", "Half-spread paid", "Price move during latency", "Taker fee", "Net"],
        "cents": [r["gross"] * 100, -r["spread"] * 100, -r["latency"] * 100, -r["fee"] * 100, r["net"] * 100],
    }
)
steps["end"] = steps["cents"].cumsum().where(steps.index < 4, steps["cents"])
steps["start"] = (steps["end"] - steps["cents"]).where(steps.index < 4, 0.0)
steps["kind"] = np.where(steps.index == 0, "gross", np.where(steps.index == 4, "net", "cost"))
bars = (
    alt.Chart(steps)
    .mark_bar(size=56, cornerRadius=4)
    .encode(
        x=alt.X("step:N", sort=list(steps["step"]), title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
        y=alt.Y("start:Q", title="Cents per contract"),
        y2="end:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["gross", "cost", "net"], range=[BLUE, "#b8b7b0", ORANGE]), legend=None),
        tooltip=[alt.Tooltip("step:N"), alt.Tooltip("cents:Q", title="Cents", format="+.2f")],
    )
)
st.altair_chart(bars.properties(height=300), use_container_width=True)
st.markdown(
    md(
        f"**{r['hypothesis_id']}:** {r['description']} Gross {cents(r['gross'])}, then {cents(-r['spread'])} of spread, "
        f"{cents(-r['latency'])} of latency and {cents(-r['fee'])} of fee leave {cents(r['net'])}. Hit rate {r['hit_rate']:.1%}, "
        f"fill rate {r['fill_rate']:.1%}."
    )
)

st.subheader("How much data an edge needs")
n = np.unique(np.geomspace(100, 20_000, 60).astype(int))
power = pd.DataFrame({"n": n, "mde": min_detectable_edge(n) * 100})
marks = []
if s.get("backtest"):
    marks.append({"n": s["backtest"]["asset_hours"], "label": "walk-forward"})
if (s.get("paper") or {}).get("asset_hours"):
    marks.append({"n": s["paper"]["asset_hours"], "label": "paper"})
m = pd.DataFrame(marks)
if not m.empty:
    m["mde"] = min_detectable_edge(m["n"]) * 100
curve = (
    alt.Chart(power)
    .mark_line(strokeWidth=2, color=BLUE)
    .encode(
        x=alt.X("n:Q", scale=alt.Scale(type="log"), title="Independent outcomes (asset-hours, or 15-minute windows)"),
        y=alt.Y("mde:Q", title="Smallest detectable edge (cents)"),
        tooltip=[alt.Tooltip("n:Q", title="Outcomes", format=","), alt.Tooltip("mde:Q", title="Edge ¢", format=".2f")],
    )
)
layer = curve
if not m.empty:
    layer = curve + alt.Chart(m).mark_point(size=110, filled=True, color=ORANGE).encode(
        x="n:Q", y="mde:Q", tooltip=[alt.Tooltip("label:N"), alt.Tooltip("n:Q", title="Outcomes"), alt.Tooltip("mde:Q", title="Edge ¢", format=".2f")]
    ) + alt.Chart(m).mark_text(dx=10, dy=-10, align="left", color="#52514e").encode(x="n:Q", y="mde:Q", text="label:N")
st.altair_chart(layer.properties(height=300), use_container_width=True)
st.caption(
    "One-sided test at 5%, 80% power, and a 50-cent standard deviation per outcome, which is what a contract near 50 cents carries. "
    "A month of 15-minute windows is about 2,900 outcomes, enough to see an edge of about 2.3 cents and no smaller."
)
