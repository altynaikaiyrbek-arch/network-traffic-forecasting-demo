"""Synthetic anomaly injection: four event types of chosen size and length."""
import numpy as np
import pandas as pd

EVENT_TYPES = ("spike", "level shift", "ramp", "outage")
OUTAGE_LEVEL = 0.05  # an outage leaves 5 per cent of the expected load


SPIKE_STEPS = 2  # a spike is a level shift that lasts one or two intervals


def effective_duration(kind, duration):
    """Number of steps an event lasts. A spike always lasts SPIKE_STEPS."""
    return SPIKE_STEPS if kind == "spike" else max(1, int(duration))


def event_multiplier(kind, amplitude, duration, length):
    """Multiplier applied to the series from the event start, one value per step.

    Follows the injection model of the report: the event adds a * mu(t) * g(u) to the load,
    with mu the expected load and u the position inside the event.

    spike        g = 1 for two intervals.
    level shift  g = 1 for `duration` steps.
    ramp         g rises linearly from 0 to 1 over `duration` steps, then the load returns.
    outage       the load drops to 5 per cent for `duration` steps (amplitude is not used).
    """
    if kind not in EVENT_TYPES:
        raise ValueError(f"unknown event type {kind!r}")
    steps = effective_duration(kind, duration)
    n = min(steps, length)
    if kind in ("spike", "level shift"):
        return np.full(n, 1 + amplitude)
    if kind == "ramp":
        return 1 + amplitude * np.arange(1, n + 1) / steps
    return np.full(n, OUTAGE_LEVEL)


def inject(series, kind, start, amplitude, duration):
    """Return (series with the event, (first_step, last_step_exclusive))."""
    out = series.copy()
    mult = event_multiplier(kind, amplitude, duration, len(series) - start)
    end = start + len(mult)
    out.iloc[start:end] = out.iloc[start:end].to_numpy() * mult
    return out, (start, end)
