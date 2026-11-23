"""Synthetic anomaly injection: four event types of chosen size and length."""
import numpy as np
import pandas as pd

EVENT_TYPES = ("spike", "level shift", "ramp", "outage")
OUTAGE_LEVEL = 0.05  # an outage leaves 5 per cent of the expected load


def event_multiplier(kind, amplitude, duration, length):
    """Multiplier applied to the series from the event start, one value per step.

    spike        load is raised by `amplitude` for `duration` steps.
    level shift  load is raised by `amplitude` from the start to the end of the series.
    ramp         load rises linearly from 0 to `amplitude` over `duration` steps, then returns.
    outage       load drops to 5 per cent for `duration` steps (amplitude is not used).
    """
    if kind not in EVENT_TYPES:
        raise ValueError(f"unknown event type {kind!r}")
    duration = max(1, int(duration))
    if kind == "level shift":
        return np.full(length, 1 + amplitude)
    n = min(duration, length)
    if kind == "spike":
        return np.full(n, 1 + amplitude)
    if kind == "ramp":
        return 1 + amplitude * np.arange(1, n + 1) / duration
    return np.full(n, OUTAGE_LEVEL)


def inject(series, kind, start, amplitude, duration):
    """Return (series with the event, (first_step, last_step_exclusive))."""
    out = series.copy()
    mult = event_multiplier(kind, amplitude, duration, len(series) - start)
    end = start + len(mult)
    out.iloc[start:end] = out.iloc[start:end].to_numpy() * mult
    return out, (start, end)
