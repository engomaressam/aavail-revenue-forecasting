"""Tests for the forecasting methods and the backtest split (src/evaluation.py)."""

import numpy as np
import pandas as pd

import evaluation as ev


def series(values):
    return pd.Series(values, index=pd.date_range("2018-01-01", periods=len(values), freq="D"), dtype=float)


def test_naive_is_the_last_30_days():
    s = series(np.arange(500))
    assert ev.naive(s, [], [400])[0] == sum(range(370, 400))


def test_seasonal_naive_is_the_same_30_days_last_year():
    s = series(np.arange(500))
    origin = 420
    assert ev.seasonal_naive(s, [], [origin])[0] == sum(range(origin - ev.YEAR, origin - ev.YEAR + 30))


def test_blend_is_the_average_of_the_two():
    s = series(np.random.default_rng(1).uniform(0, 100, 500))
    o = [450]
    assert np.isclose(ev.blend(s, [], o)[0], (ev.naive(s, [], o)[0] + ev.seasonal_naive(s, [], o)[0]) / 2)


def test_split_has_no_leakage():
    s = series(np.ones(600))
    sp = ev.split(s)
    # every training target window ends before the first test forecast date
    assert max(sp.train_origins) + ev.HORIZON <= min(sp.test_origins)
    assert all(b - a == ev.STEP for a, b in zip(sp.test_origins, sp.test_origins[1:]))
    assert max(sp.test_origins) + ev.HORIZON <= len(s)


def test_daily_revenue_spans_the_whole_data_range():
    df = pd.DataFrame({"date": pd.to_datetime(["2018-01-01", "2018-01-10", "2018-01-05"]),
                       "country": ["A", "B", "B"], "price": [5.0, 7.0, 3.0]})
    b = ev.daily_revenue(df, "B")
    assert b.index[0] == pd.Timestamp("2018-01-01") and b.index[-1] == pd.Timestamp("2018-01-10")
    assert b.sum() == 10.0 and b.loc["2018-01-01"] == 0.0


def test_wape_handles_zero_revenue_windows():
    m = ev.metrics(np.array([0.0, 100.0]), np.array([10.0, 90.0]))
    assert m["wape_pct"] == 20.0 and m["mae"] == 10.0 and m["bias"] == 0.0
