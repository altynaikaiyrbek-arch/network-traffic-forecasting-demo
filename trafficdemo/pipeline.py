"""The demo pipeline: forecast, residuals, detector, event injection, metrics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import split_blocks, steps_per
from .detectors import cusum_alarms, merge_alarms, robust_scale, zscore_alarms, zscores
from .forecasters import BoostedForecaster, seasonal_naive
from .injection import inject

FORECASTERS = ("Seasonal naive (weekly)", "LightGBM")
DETECTORS = ("Robust Z-score", "CUSUM")


@dataclass
class DetectorSettings:
    method: str = "Robust Z-score"
    k: float = 4.0               # Z-score threshold
    min_consecutive: int = 1     # steps above the threshold in a row
    drift: float = 0.5           # CUSUM allowance, in units of the residual scale
    h: float = 8.0               # CUSUM threshold
    refractory_hours: float = 6.0
    tolerance_hours: float = 2.0  # time after the event end in which an alarm still counts


@dataclass
class Detection:
    forecast: pd.Series
    residual: pd.Series
    z: np.ndarray
    score: np.ndarray
    alarm_mask: np.ndarray
    alarm_starts: list


class Pipeline:
    def __init__(self, series, freq, forecaster=FORECASTERS[1], seed=0):
        self.series = series
        self.freq = freq
        self.steps = steps_per(freq)
        self.step_hours = pd.Timedelta(freq) / pd.Timedelta("1h")
        self.train, self.val, self.test = split_blocks(len(series), freq)
        self.forecaster_name = forecaster
        self.seed = seed
        if forecaster == FORECASTERS[1]:
            self.model = BoostedForecaster(self.steps, seed).fit(series.iloc[self.train])
        else:
            self.model = None
        clean = self.forecast(series)
        self.center, self.scale = robust_scale((series - clean).iloc[self.val])

    @property
    def test_start(self):
        return self.test.start

    def forecast(self, live):
        """One-step-ahead forecast. History older than a day comes from the clean series."""
        if self.model is None:
            return seasonal_naive(self.series, self.steps["week"]).rename("forecast")
        return self.model.predict(live, self.series)

    def accuracy(self):
        """RMSE, MAE and MASE on the clean test block (MASE against weekly seasonal naive)."""
        y = self.series.iloc[self.test]
        err = y - self.forecast(self.series).iloc[self.test]
        base = y - seasonal_naive(self.series, self.steps["week"]).iloc[self.test]
        return {
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAE": float(err.abs().mean()),
            "MASE": float(err.abs().mean() / base.abs().mean()),
        }

    def detect(self, live, s=DetectorSettings()):
        forecast = self.forecast(live)
        residual = live - forecast
        z = zscores(residual.to_numpy(), self.center, self.scale)
        if s.method == "CUSUM":
            mask, score = cusum_alarms(z, s.drift, s.h)
        else:
            mask, score = zscore_alarms(z, s.k, s.min_consecutive)
        mask[: self.test_start] = False  # alarms are only evaluated on the test block
        refractory = max(1, round(s.refractory_hours / self.step_hours))
        return Detection(forecast, residual, z, score, mask, merge_alarms(mask, refractory))

    def score_event(self, det, window, s=DetectorSettings()):
        """Detection and latency for one event, and the alarms outside it."""
        start, end = window
        tol = round(s.tolerance_hours / self.step_hours)
        inside = [a for a in det.alarm_starts if start <= a < end + tol]
        false = [a for a in det.alarm_starts if a not in inside and a >= self.test_start]
        weeks = (len(self.series) - self.test_start) / self.steps["week"]
        return {
            "detected": bool(inside),
            "latency_hours": (inside[0] - start) * self.step_hours if inside else None,
            "false_alarms": len(false),
            "false_alarms_per_week": len(false) / weeks,
            "false_alarm_steps": false,
        }

    def run_event(self, kind, start, amplitude, duration_hours, s=DetectorSettings()):
        duration = max(1, round(duration_hours / self.step_hours))
        live, window = inject(self.series, kind, start, amplitude, duration)
        det = self.detect(live, s)
        return live, window, det, self.score_event(det, window, s)

    def clean_false_alarms(self, s=DetectorSettings()):
        """Alarm events on the untouched test block, per week."""
        det = self.detect(self.series, s)
        weeks = (len(self.series) - self.test_start) / self.steps["week"]
        return len(det.alarm_starts) / weeks

    def batch(self, kind, amplitude, duration_hours, s=DetectorSettings(), runs=20, seed=1):
        """Random placements of one event in the test block, as in the planned experiments."""
        rng = np.random.default_rng(seed)
        duration = max(1, round(duration_hours / self.step_hours))
        tol = round(s.tolerance_hours / self.step_hours)
        latest = len(self.series) - duration - tol - 1
        starts = rng.integers(self.test_start + self.steps["day"], latest, size=runs)
        rows = [self.run_event(kind, int(t), amplitude, duration_hours, s)[3] for t in starts]
        lat = [r["latency_hours"] for r in rows if r["detected"]]
        return {
            "runs": runs,
            "detection_rate": float(np.mean([r["detected"] for r in rows])),
            "median_latency_hours": float(np.median(lat)) if lat else None,
            "p90_latency_hours": float(np.percentile(lat, 90)) if lat else None,
            "false_alarms_per_week": float(np.mean([r["false_alarms_per_week"] for r in rows])),
        }
