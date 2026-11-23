import numpy as np
import pandas as pd
import pytest

from trafficdemo.detectors import cusum_alarms, merge_alarms, robust_scale, zscore_alarms, zscores
from trafficdemo.injection import inject
from trafficdemo.pipeline import DetectorSettings, Pipeline


def make_series(weeks=6, noise=0.02, seed=0):
    """Synthetic hourly series with daily and weekly cycles."""
    idx = pd.date_range("2004-05-29", periods=168 * weeks, freq="1h")
    rng = np.random.default_rng(seed)
    hour = idx.hour.to_numpy()
    base = 1000 + 400 * np.sin(2 * np.pi * (hour - 6) / 24) - 150 * (idx.day_of_week.to_numpy() >= 5)
    return pd.Series(base * (1 + noise * rng.standard_normal(len(idx))), index=idx)


def test_robust_scale_ignores_outlier():
    center, scale = robust_scale([0, 1, -1, 0, 1, -1, 1000])
    assert abs(center) < 1 and scale < 3


def test_zscore_needs_consecutive_steps():
    z = np.array([0, 5, 0, 5, 5, 5, 0])
    mask, _ = zscore_alarms(z, k=4, min_consecutive=3)
    assert mask.tolist() == [False, False, False, False, False, True, False]


def test_cusum_detects_small_persistent_shift():
    z = np.full(30, 1.5)
    mask, _ = cusum_alarms(z, drift=0.5, threshold=5)
    assert mask.any() and np.flatnonzero(mask)[0] == 5  # 1.0 per step, crosses 5 at the sixth step


def test_cusum_ignores_noise_inside_drift():
    z = np.tile([0.4, -0.4], 50)
    mask, _ = cusum_alarms(z, drift=0.5, threshold=5)
    assert not mask.any()


def test_merge_alarms_counts_long_event_once():
    mask = np.zeros(50, dtype=bool)
    mask[10:20] = True
    mask[30] = True
    assert merge_alarms(mask, refractory=3) == [10, 30]


@pytest.mark.parametrize("kind,expected_end", [("spike", 13), ("ramp", 13), ("outage", 13), ("level shift", 100)])
def test_injection_window(kind, expected_end):
    s = pd.Series(np.ones(100), index=pd.date_range("2004-01-01", periods=100, freq="1h"))
    out, (a, b) = inject(s, kind, start=10, amplitude=0.5, duration=3)
    assert (a, b) == (10, expected_end)
    assert (out.iloc[:10] == 1).all()
    if kind != "level shift":
        assert (out.iloc[b:] == 1).all()
    assert (out.iloc[a:b] != 1).all()


def test_ramp_rises_to_amplitude():
    s = pd.Series(np.ones(20), index=pd.date_range("2004-01-01", periods=20, freq="1h"))
    out, (a, b) = inject(s, "ramp", start=5, amplitude=0.6, duration=3)
    assert np.allclose(out.iloc[5:8], [1.2, 1.4, 1.6])


@pytest.mark.parametrize("forecaster", ["Seasonal naive (weekly)", "LightGBM"])
def test_big_outage_is_detected_quickly(forecaster):
    pipe = Pipeline(make_series(), "1h", forecaster)
    start = pipe.test_start + 24 * 3 + 10
    _, window, det, res = pipe.run_event("outage", start, 0, 4, DetectorSettings())
    assert res["detected"] and res["latency_hours"] <= 1


def test_clean_series_gives_few_alarms():
    pipe = Pipeline(make_series(), "1h", "LightGBM")
    assert pipe.clean_false_alarms(DetectorSettings()) < 3


def test_batch_is_repeatable():
    pipe = Pipeline(make_series(), "1h", "Seasonal naive (weekly)")
    a = pipe.batch("level shift", 0.5, 6, runs=5)
    b = pipe.batch("level shift", 0.5, 6, runs=5)
    assert a == b

