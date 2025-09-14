"""
Fair comparison of the forecasting approaches on one target: total revenue over the 30 days after a
forecast date. Every model is scored on the same forecast dates (weekly origins in the held-out final
20% of the history), against a naive baseline that predicts the previous 30 days' total.

Training windows for the learned models end before the first test origin, so no test revenue leaks
into training. ARIMA and Holt-Winters are refit at each origin on the history up to that date.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZON = 30    # days ahead whose revenue is summed
LOOKBACK = 30   # daily values used as features by the learned models
STEP = 7        # days between forecast origins


def daily_revenue(df: pd.DataFrame, country: str = None) -> pd.Series:
    """Daily revenue with missing days filled with 0 (the shop has quiet days, not missing data)."""
    # Every series spans the whole data range, so a late-starting country still lines up with last year's dates
    d = df if country is None else df[df["country"] == country]
    s = d.groupby("date")["price"].sum()
    return s.reindex(pd.date_range(df["date"].min(), df["date"].max(), freq="D"), fill_value=0.0)


def windows(series: pd.Series, origins: List[int]):
    """Features: the LOOKBACK days before each origin. Target: the HORIZON-day sum from the origin on."""
    v = series.values
    X = np.array([v[o - LOOKBACK:o] for o in origins])
    y = np.array([v[o:o + HORIZON].sum() for o in origins])
    return X, y


def learned(make_model: Callable) -> Callable:
    def fit_predict(series: pd.Series, train_origins: List[int], test_origins: List[int]) -> np.ndarray:
        X, y = windows(series, train_origins)
        model = make_model().fit(X, y)
        return model.predict(windows(series, test_origins)[0])
    return fit_predict


def naive(series, train_origins, test_origins):
    v = series.values
    return np.array([v[o - HORIZON:o].sum() for o in test_origins])


YEAR = 364  # 52 weeks, so last year's window starts on the same weekday


def seasonal_naive(series, train_origins, test_origins):
    """The same 30 days one year earlier."""
    v = series.values
    return np.array([v[o - YEAR:o - YEAR + HORIZON].sum() for o in test_origins])


def seasonal_growth(series, train_origins, test_origins):
    """Last year's same 30 days, scaled by how the last 30 days compare with the same 30 days a year earlier."""
    v = series.values
    out = []
    for o in test_origins:
        last_year_next = v[o - YEAR:o - YEAR + HORIZON].sum()
        recent, recent_last_year = v[o - HORIZON:o].sum(), v[o - YEAR - HORIZON:o - YEAR].sum()
        ratio = recent / recent_last_year if recent_last_year > 0 else 1.0
        out.append(last_year_next * float(np.clip(ratio, 0.5, 2.0)))
    return np.array(out)


def blend(series, train_origins, test_origins):
    """Fixed 50/50 average of the last 30 days and the same 30 days last year: nothing to tune, so nothing to overfit."""
    return 0.5 * naive(series, train_origins, test_origins) + 0.5 * seasonal_naive(series, train_origins, test_origins)


def seasonal_features(series: pd.Series, origins: List[int]):
    """Last 4 weekly totals, the last 30 days, and last year's same and preceding 30 days."""
    v = series.values
    X = np.array([[v[o - 7:o].sum(), v[o - 14:o - 7].sum(), v[o - 21:o - 14].sum(), v[o - 28:o - 21].sum(),
                   v[o - HORIZON:o].sum(), v[o - YEAR:o - YEAR + HORIZON].sum(), v[o - YEAR - HORIZON:o - YEAR].sum()]
                  for o in origins])
    y = np.array([v[o:o + HORIZON].sum() for o in origins])
    return X, y


def seasonal_ridge(series, train_origins, test_origins):
    """Ridge regression on recent weekly totals plus last year's windows (needs a year of history per window)."""
    train = [o for o in train_origins if o - YEAR - HORIZON >= 0]
    if len(train) < 20:   # not enough year-long history to fit; fall back to the seasonal heuristic
        return seasonal_growth(series, train_origins, test_origins)
    X, y = seasonal_features(series, train)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    return np.maximum(model.predict(seasonal_features(series, test_origins)[0]), 0.0)


def arima(series, train_origins, test_origins):
    from statsmodels.tsa.arima.model import ARIMA
    out = []
    for o in test_origins:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(series.values[:o], order=(2, 1, 2)).fit()
        out.append(max(fit.forecast(steps=HORIZON).sum(), 0.0))
    return np.array(out)


def holt_winters(series, train_origins, test_origins):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    out = []
    for o in test_origins:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(series.values[:o], trend="add", seasonal="add", seasonal_periods=7).fit()
        out.append(max(fit.forecast(HORIZON).sum(), 0.0))
    return np.array(out)


MODELS: Dict[str, Callable] = {
    "naive (last 30 days)": naive,
    "ridge regression": learned(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
    "random forest": learned(lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    "gradient boosting": learned(lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    "seasonal naive (last year)": seasonal_naive,
    "seasonal naive x recent growth": seasonal_growth,
    "blend: last 30 days + last year": blend,
    "seasonal ridge": seasonal_ridge,
    "ARIMA(2,1,2)": arima,
    "Holt-Winters (weekly)": holt_winters,
}


@dataclass
class Split:
    train_origins: List[int]
    test_origins: List[int]


def split(series: pd.Series, test_share: float = 0.2) -> Split:
    """Weekly test origins over the final share of the history; training windows end before the first one."""
    last = len(series) - HORIZON
    first_test = LOOKBACK + int((last - LOOKBACK) * (1 - test_share))
    test = list(range(first_test, last + 1, STEP))
    train = list(range(LOOKBACK, first_test - HORIZON + 1))
    return Split(train, test)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    err = predicted - actual
    # WAPE (total absolute error / total revenue) instead of MAPE: small countries have 30-day windows with no revenue
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "wape_pct": float(np.abs(err).sum() / actual.sum() * 100), "bias": float(np.mean(err))}


def compare(series: pd.Series, models: Dict[str, Callable] = None) -> pd.DataFrame:
    models = models or MODELS
    sp = split(series)
    actual = windows(series, sp.test_origins)[1]
    rows = []
    for name, fn in models.items():
        m = metrics(actual, fn(series, sp.train_origins, sp.test_origins))
        rows.append({"model": name, **m})
    table = pd.DataFrame(rows)
    naive_mae = table.loc[table["model"] == "naive (last 30 days)", "mae"].iloc[0]
    table["mae_vs_naive_pct"] = (table["mae"] / naive_mae - 1) * 100
    table["test_origins"] = len(sp.test_origins)
    table["test_from"] = series.index[sp.test_origins[0]].date()
    return table.sort_values("mae").reset_index(drop=True)
