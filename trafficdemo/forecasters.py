"""Forecasters: seasonal naive and a boosted-tree model on lag features."""
import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = ["lag1", "mean_3h", "lag_day", "lag_week", "hour", "dow"]


def seasonal_naive(reference, lag):
    """Forecast for time t is the value at t - lag (one week by default in the demo).

    The reference series is the operator's cleaned history. A past incident does
    not leak into the baseline a day or a week later.
    """
    return reference.shift(lag)


def make_features(live, reference, steps):
    """Feature table for one-step-ahead forecasting.

    lag1 and mean_3h use the live series, which is what the monitor sees now,
    including any anomaly. lag_day and lag_week use the cleaned reference history.
    """
    f = pd.DataFrame(index=live.index)
    f["lag1"] = live.shift(1)
    f["mean_3h"] = live.shift(1).rolling(3 * steps["hour"]).mean()
    f["lag_day"] = reference.shift(steps["day"])
    f["lag_week"] = reference.shift(steps["week"])
    f["hour"] = live.index.hour + live.index.minute / 60
    f["dow"] = live.index.day_of_week
    return f[FEATURES]


class BoostedForecaster:
    """LightGBM on lagged values, a rolling mean and calendar variables."""

    def __init__(self, steps, seed=0):
        self.steps = steps
        self.params = dict(
            objective="regression", learning_rate=0.05, num_leaves=15,
            min_data_in_leaf=20, feature_fraction=0.9, seed=seed,
            deterministic=True, num_threads=1, verbose=-1,
        )
        self.rounds = 300
        self.model = None

    def fit(self, clean_train):
        X = make_features(clean_train, clean_train, self.steps)
        ok = X.notna().all(axis=1)
        data = lgb.Dataset(X[ok], label=clean_train[ok])
        self.model = lgb.train(self.params, data, num_boost_round=self.rounds)
        return self

    def predict(self, live, reference):
        X = make_features(live, reference, self.steps)
        out = pd.Series(np.nan, index=live.index, name="forecast")
        ok = X.notna().all(axis=1)
        out[ok] = self.model.predict(X[ok])
        return out

