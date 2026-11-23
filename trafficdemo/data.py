"""Loading the Abilene data and turning it into one regular load series."""
from pathlib import Path

import numpy as np
import pandas as pd

# First day of each weekly file, from the readme of the release.
WEEK_START = {
    "X10": "2004-05-29", "X11": "2004-06-05", "X12": "2004-06-12",
    "X13": "2004-06-19", "X14": "2004-06-26", "X15": "2004-07-03",
}
DEFAULT_WEEKS = tuple(WEEK_START)
STEP_5MIN = pd.Timedelta(minutes=5)
# Raw unit is 100 bytes per 5 minutes; this factor converts it to Mbit/s.
TO_MBPS = 100 * 8 / 300 / 1e6


def read_demands(raw_dir):
    """Return a list of (source, destination) in column order of the matrices."""
    pairs = []
    for line in Path(raw_dir, "demands").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        od, _index = line.split()
        src, dst = od.split(",")
        pairs.append((src, dst))
    return pairs


def list_nodes(raw_dir):
    return sorted({src for src, _ in read_demands(raw_dir)})


def load_abilene(raw_dir="data/raw", node=None, freq="1h", weeks=DEFAULT_WEEKS):
    """Aggregate load in Mbit/s as a pandas Series on a regular time grid.

    node=None sums all 144 origin-destination demands (load entering the network).
    With node="ATLAng" only the demands that start at that router are summed.
    This approximates link or node load and is an assumption, not a measurement.
    """
    pairs = read_demands(raw_dir)
    if len(pairs) != 144:
        raise ValueError(f"expected 144 demands, found {len(pairs)}")
    cols = [i for i, (src, _) in enumerate(pairs) if node is None or src == node]
    if not cols:
        raise ValueError(f"unknown node {node!r}")

    parts = []
    for week in weeks:
        raw = np.loadtxt(Path(raw_dir, f"{week}.gz"))
        # Each row has 144 blocks of 5 values; the first value of a block is the measured demand.
        measured = raw.reshape(raw.shape[0], 144, 5)[:, :, 0]
        start = pd.Timestamp(WEEK_START[week])
        index = start + STEP_5MIN * np.arange(raw.shape[0])
        parts.append(pd.Series(measured[:, cols].sum(axis=1) * TO_MBPS, index=index))
    series = pd.concat(parts)

    expected = pd.date_range(series.index[0], periods=len(series), freq="5min")
    if not series.index.equals(expected):
        raise ValueError("the chosen weeks are not consecutive; pick consecutive weeks only")
    series = series.resample(freq).mean()
    series.name = "load_mbps"
    return series


def steps_per(freq):
    """Number of time steps in an hour, a day and a week for a given resolution."""
    step = pd.Timedelta(freq)
    return {
        "hour": max(1, int(pd.Timedelta("1h") / step)),
        "day": int(pd.Timedelta("1D") / step),
        "week": int(pd.Timedelta("7D") / step),
    }


def split_blocks(length, freq, train_weeks=3, val_weeks=1):
    """Time-ordered split: train, validation (scale and thresholds), test."""
    week = steps_per(freq)["week"]
    a, b = train_weeks * week, (train_weeks + val_weeks) * week
    return slice(0, a), slice(a, b), slice(b, length)
