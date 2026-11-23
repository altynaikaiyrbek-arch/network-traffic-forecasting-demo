# Network traffic forecasting and early anomaly warning: demo

Demonstration version of the BSc dissertation project *Network Traffic Forecasting for Capacity Planning and Early Anomaly Warning* (Heriot-Watt University, Computational Sciences and Software Engineering).

The demo loads one aggregate traffic series, forecasts it one step ahead, scores the forecast error with a robust Z-score or CUSUM, lets you inject a synthetic event of chosen type, size and length, and plots the series, the forecast and the alarms. It prints the detection latency for the injected event.

## What is included

| Part | Content |
|---|---|
| Data | Abilene traffic matrices (Internet2 NetFlow, 5-minute resolution), six consecutive weeks, 29 May to 9 July 2004 |
| Forecasters | Weekly seasonal naive; LightGBM on lag, rolling-mean and calendar features |
| Detectors | Robust Z-score (median and MAD from a validation week); two-sided CUSUM |
| Events | Spike (two intervals), level shift, ramp, outage; amplitude 10 to 100 per cent; duration of level shift, ramp and outage from 15 minutes to 48 hours |
| Metrics | RMSE, MAE, MASE; detection, detection latency, false alarms per week |

Recurrent networks (LSTM, GRU), the autoencoder detector, the other two datasets and the full experimental grid belong to the second semester and are not part of this demo.

## Run it

Python 3.10 or newer. In PowerShell, from the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
streamlit run app.py
```

`download_data.py` fetches about 45 MB from the public Abilene release at https://www.cs.utexas.edu/~yzhang/research/AbileneTM/ into `data/raw/`. The raw files are not stored in this repository. The app opens at http://localhost:8501. On the first start Streamlit may ask for an email address; press Enter to skip it. The first load takes about a minute while the forecaster is trained.

Run the tests with `pytest`.

## How to use the demo

1. Pick the series (the traffic that starts at one router, or the network total), the resolution and the forecaster in the sidebar.
2. Pick an event type, amplitude, duration and start time. The orange band on the plot is the injected event.
3. Pick a detector and its threshold. Red crosses are alarms.
4. Read the result at the top: whether the event was detected, the latency in hours, the false alarms outside the event, and the forecast error on the clean test block.
5. Open "Repeat with 20 random placements" to get detection rate, latency and false alarms per week averaged over 20 random placements of the same event.

## How it works

- Load is the sum of the origin-destination demands of the traffic matrix, converted to Mbit/s. It approximates link or router load; it is not a link measurement.
- Time order is respected. Weeks 1 to 3 train the forecaster, week 4 calibrates the residual scale (median and MAD), weeks 5 and 6 are the test block. Alarms are evaluated only on the test block.
- Forecasts are one step ahead. The last hour comes from the observed series, including the injected event. Values a day or a week back come from the cleaned history, as if past incidents had already been removed. Without this, an event would reappear in the forecast a week later and create false alarms.
- An alarm is merged with the alarms that follow within the refractory period, so one long anomaly counts as one alarm event.
- An event is detected if an alarm starts between its start and a tolerance window after its end. Latency is the time from the event start to the first such alarm. All other alarm events on the test block count as false alarms. The Abilene series may contain real anomalies, which are not marked yet, so some alarms counted as false may be real.

## Repository layout

```
app.py                  Streamlit interface
trafficdemo/data.py     loading and aggregation of the Abilene matrices
trafficdemo/forecasters.py   seasonal naive and LightGBM
trafficdemo/injection.py     the four event types
trafficdemo/detectors.py     Z-score, CUSUM, alarm merging
trafficdemo/pipeline.py      forecast, detection, metrics, random placements
scripts/download_data.py     data download
tests/                  unit tests
```

## Data source and licence note

Y. Zhang, Abilene traffic matrices, derived from Internet2 NetFlow data with the last 11 bits of the IP addresses masked. The matrices contain no traffic destined for AS11537 (Abilene itself). The data are anonymised quantitative time series without personal information.
