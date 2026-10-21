import altair as alt
import pandas as pd
import streamlit as st

from common import BLUE, ORANGE, q, require_run

run = require_run()
s = run["summary"]
paper = s.get("paper") or {}

st.title("Calibration: model against market")
st.caption(
    "Scored on contested snapshots, where the market's mid was between 10 and 90 cents. A forecast is calibrated when "
    "things it calls 70% happen about 70% of the time, which puts its dots on the diagonal."
)

period = st.segmented_control("Period", ["Walk-forward test", "Paper trading"], default="Walk-forward test")
key = "paper" if period == "Paper trading" else "research"
c1, c2, c3 = st.columns(3)
if key == "research":
    bm, bk = s["brier_model"], s["brier_market"]
    n = s["contested_snapshots"]
else:
    bm, bk = paper.get("brier_model"), paper.get("brier_market")
    n = None
c1.metric("Brier score, model", f"{bm:.4f}" if bm is not None else "n/a", help="Mean squared error of the probability against the 0/1 outcome. Lower is better.")
c2.metric("Brier score, market mid", f"{bk:.4f}" if bk is not None else "n/a")
if bm is not None and bk is not None:
    c3.metric("Model minus market", f"{bm - bk:+.4f}", help="Positive means the market forecast better overall.")

rel = q("select source, bin, p_mean, freq, n from research.calibration where run_id = %s and period = %s order by source, bin", (run["run_id"], key))
if rel.empty:
    st.info("No calibration rows for this period yet.")
    st.stop()
rel["source"] = rel["source"].map({"model": "Model", "market": "Market mid"})
diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#8a8984", strokeDash=[4, 4], strokeWidth=1).encode(x="x:Q", y="y:Q")
enc = {
    "x": alt.X("p_mean:Q", title="Forecast probability", scale=alt.Scale(domain=[0, 1])),
    "y": alt.Y("freq:Q", title="Observed frequency", scale=alt.Scale(domain=[0, 1])),
    "color": alt.Color("source:N", scale=alt.Scale(domain=["Model", "Market mid"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top-left")),
    "tooltip": [
        alt.Tooltip("source:N"),
        alt.Tooltip("p_mean:Q", title="Mean forecast", format=".3f"),
        alt.Tooltip("freq:Q", title="Observed", format=".3f"),
        alt.Tooltip("n:Q", title="Snapshots", format=","),
    ],
}
chart = alt.Chart(rel).mark_line(strokeWidth=2).encode(**enc) + alt.Chart(rel).mark_point(size=70, filled=True).encode(**enc)
st.altair_chart((diag + chart).properties(height=420), use_container_width=True)
st.caption(
    (f"{n:,} contested snapshots. " if n else "")
    + "The market maker in this data prices off the true volatility, so it is hard to beat on "
    "calibration alone. The edge the research found is about timing during bursts, not about a better probability on average."
)
with st.expander("Table view"):
    st.dataframe(rel.pivot(index="bin", columns="source", values=["p_mean", "freq", "n"]).round(3), use_container_width=True)
