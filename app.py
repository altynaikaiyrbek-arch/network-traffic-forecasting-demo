"""Streamlit demo: forecast, inject an event, see whether and when it is detected.

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trafficdemo.data import list_nodes, load_abilene
from trafficdemo.injection import EVENT_TYPES
from trafficdemo.pipeline import DETECTORS, FORECASTERS, DetectorSettings, Pipeline

RAW_DIR = "data/raw"

st.set_page_config(page_title="Traffic forecasting demo", layout="wide")
st.title("Network traffic forecasting and early anomaly warning")
st.caption(
    "Abilene backbone traffic matrices, weeks of 29 May to 9 July 2004. "
    "The model forecasts the load one step ahead, the detector scores the forecast error, "
    "and you inject a synthetic event to see whether and when it is detected."
)


@st.cache_resource(show_spinner="Loading data and training the forecaster...")
def get_pipeline(node, freq, forecaster):
    series = load_abilene(RAW_DIR, node=None if node == "All routers (network total)" else node, freq=freq)
    return Pipeline(series, freq, forecaster)


try:
    nodes = list_nodes(RAW_DIR)
except FileNotFoundError:
    st.error("Data not found. Run:  python scripts/download_data.py")
    st.stop()

with st.sidebar:
    st.header("Data and forecaster")
    node = st.selectbox("Series", ["All routers (network total)"] + nodes,
                        help="Sum of the origin-destination demands that start at the chosen router.")
    freq = st.selectbox("Resolution", ["1h", "15min", "5min"], index=0)
    forecaster = st.selectbox("Forecaster", FORECASTERS, index=1)

    st.header("Injected event")
    kind = st.selectbox("Event type", EVENT_TYPES)

pipe = get_pipeline(node, freq, forecaster)
hours = pipe.step_hours

with st.sidebar:
    amplitude = st.select_slider(
        "Amplitude (share of expected load)", options=[0.10, 0.25, 0.50, 0.75, 1.00], value=0.50,
        format_func=lambda v: f"{int(v * 100)} %", disabled=(kind == "outage"),
    )
    duration_hours = st.select_slider(
        "Duration (hours)", options=[h for h in [0.25, 0.5, 1, 2, 6, 12, 24, 48] if h >= hours], value=6 if hours <= 6 else hours,
        disabled=(kind == "level shift"),
    )
    test_index = pipe.series.index[pipe.test]
    first_start = test_index[0] + pd.Timedelta("1D")
    last_start = test_index[-1] - pd.Timedelta(hours=duration_hours + 3)
    start_time = st.slider(
        "Event start", min_value=first_start.to_pydatetime(), max_value=last_start.to_pydatetime(),
        value=(first_start + pd.Timedelta("3D 14h")).to_pydatetime(), step=pd.Timedelta(freq).to_pytimedelta(),
        format="ddd D MMM, HH:mm",
    )

    st.header("Detector")
    method = st.selectbox("Method", DETECTORS)
    if method == "Robust Z-score":
        k = st.slider("Threshold k (in residual scales)", 2.0, 8.0, 4.0, 0.25)
        min_consecutive = st.slider("Consecutive steps above k", 1, 6, 1)
        settings = DetectorSettings(method=method, k=k, min_consecutive=min_consecutive)
    else:
        drift = st.slider("Drift", 0.0, 2.0, 0.5, 0.1)
        h = st.slider("Threshold h", 2.0, 20.0, 8.0, 0.5)
        settings = DetectorSettings(method=method, drift=drift, h=h)
    settings.refractory_hours = st.slider("Merge alarms closer than (hours)", 1, 24, 6)
    settings.tolerance_hours = st.slider("Tolerance after event end (hours)", 0, 12, 2)

start = int(np.searchsorted(pipe.series.index, pd.Timestamp(start_time)))
live, window, det, result = pipe.run_event(kind, start, amplitude, duration_hours, settings)
idx = pipe.series.index
t0 = pipe.test_start

# Summary line.
c1, c2, c3, c4 = st.columns(4)
if result["detected"]:
    c1.metric("Event detected", "Yes")
    c2.metric("Detection latency", f"{result['latency_hours']:.2f} h")
else:
    c1.metric("Event detected", "No")
    c2.metric("Detection latency", "-")
c3.metric("False alarms (outside the event)", result["false_alarms"],
          help="Merged alarm events on the test block outside the injected event and its tolerance window.")
acc = pipe.accuracy()
c4.metric("Forecast MAE on clean test (Mbit/s)", f"{acc['MAE']:.0f}",
          help=f"RMSE {acc['RMSE']:.0f} Mbit/s, MASE {acc['MASE']:.2f} against weekly seasonal naive.")

# Plot 1: series, forecast, event and alarms.
fig = go.Figure()
fig.add_vrect(x0=idx[window[0]], x1=idx[min(window[1], len(idx)) - 1], fillcolor="#f5a623", opacity=0.18,
              line_width=0, annotation_text="injected event", annotation_position="top left")
fig.add_trace(go.Scatter(x=idx[t0:], y=live.iloc[t0:], name="observed load", line=dict(color="#2b6cb0", width=1.6)))
fig.add_trace(go.Scatter(x=idx[t0:], y=det.forecast.iloc[t0:], name=f"forecast ({forecaster})",
                         line=dict(color="#7a7a7a", width=1.2, dash="dot")))
if det.alarm_starts:
    fig.add_trace(go.Scatter(
        x=idx[det.alarm_starts], y=live.iloc[det.alarm_starts], mode="markers", name="alarm",
        marker=dict(color="#d62728", size=11, symbol="x"),
    ))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Load (Mbit/s)",
                  legend=dict(orientation="h", y=1.12), hovermode="x unified")
st.plotly_chart(fig, width='stretch')

# Plot 2: detector score and threshold.
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=idx[t0:], y=det.score[t0:], name="detector score", line=dict(color="#4a4a4a", width=1.2)))
limit = settings.k if method == "Robust Z-score" else settings.h
fig2.add_hline(y=limit, line=dict(color="#d62728", dash="dash"), annotation_text="threshold")
fig2.add_vrect(x0=idx[window[0]], x1=idx[min(window[1], len(idx)) - 1], fillcolor="#f5a623", opacity=0.18, line_width=0)
fig2.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10),
                   yaxis_title="|z|" if method == "Robust Z-score" else "CUSUM statistic", showlegend=False)
st.plotly_chart(fig2, width='stretch')

clean_rate = pipe.clean_false_alarms(settings)
st.caption(
    f"On the untouched test block this detector raises {clean_rate:.1f} alarm events per week "
    f"(all of them counted as false alarms). The event ends at {idx[min(window[1], len(idx)) - 1]:%a %d %b %H:%M}."
)

with st.expander("Repeat with 20 random placements of this event"):
    st.write("Same event type, amplitude and duration, placed at 20 random times in the test block. "
             "The random seed is fixed, so the numbers repeat.")
    if st.button("Run 20 placements"):
        out = pipe.batch(kind, amplitude, duration_hours, settings, runs=20)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Detection rate", f"{out['detection_rate'] * 100:.0f} %")
        b2.metric("Median latency", "-" if out["median_latency_hours"] is None else f"{out['median_latency_hours']:.2f} h")
        b3.metric("90th percentile latency", "-" if out["p90_latency_hours"] is None else f"{out['p90_latency_hours']:.2f} h")
        b4.metric("False alarms per week", f"{out['false_alarms_per_week']:.2f}")

with st.expander("What the demo does and does not do"):
    st.markdown(
        "- The load is the sum of the origin-destination demands of the Abilene traffic matrices. "
        "It approximates link or router load and is not a link measurement.\n"
        "- Forecasts are one step ahead. The last hour comes from the observed series, including the event. "
        "Values a day or a week back come from the cleaned history, as if past incidents had already been removed.\n"
        "- Training uses weeks 1 to 3, the residual scale is calibrated on week 4, and weeks 5 and 6 are the test block.\n"
        "- Recurrent networks and the autoencoder detector are planned for the second semester."
    )

