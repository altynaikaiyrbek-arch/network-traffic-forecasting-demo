"""Residual scoring and alarm rules: robust Z-score and CUSUM."""
import numpy as np


def robust_scale(residuals):
    """Median and scaled median absolute deviation of validation residuals."""
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    center = float(np.median(r))
    scale = float(1.4826 * np.median(np.abs(r - center)))
    return center, max(scale, 1e-9)


def zscores(residuals, center, scale):
    return (np.asarray(residuals, dtype=float) - center) / scale


def zscore_alarms(z, k, min_consecutive=1):
    """Alarm when |z| > k for at least `min_consecutive` steps in a row.

    Returns the alarm mask (marked from the step where the run is long enough)
    and the score that is plotted, which is |z|.
    """
    over = np.abs(np.nan_to_num(z)) > k
    mask = np.zeros(len(z), dtype=bool)
    run = 0
    for i, flag in enumerate(over):
        run = run + 1 if flag else 0
        mask[i] = run >= min_consecutive
    return mask, np.abs(z)


def cusum_alarms(z, drift, threshold):
    """Two-sided CUSUM on the standardised residual.

    S+ = max(0, S+ + z - drift), S- = max(0, S- - z - drift). Alarm when either
    exceeds `threshold`; both sums restart from zero after an alarm.
    """
    z = np.nan_to_num(np.asarray(z, dtype=float))
    hi = lo = 0.0
    score = np.zeros(len(z))
    mask = np.zeros(len(z), dtype=bool)
    for i, value in enumerate(z):
        hi = max(0.0, hi + value - drift)
        lo = max(0.0, lo - value - drift)
        score[i] = max(hi, lo)
        if score[i] > threshold:
            mask[i] = True
            hi = lo = 0.0
    return mask, score


def merge_alarms(mask, refractory):
    """Merge alarm steps closer than `refractory` steps into one event.

    A long anomaly then counts as one alarm event and not as many false alarms.
    Returns the start step of every merged event.
    """
    starts, last = [], None
    for i in np.flatnonzero(mask):
        if last is None or i - last > refractory:
            starts.append(int(i))
        last = i
    return starts
