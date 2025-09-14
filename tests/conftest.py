import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def invoice_dir(tmp_path):
    """Synthetic invoices in AAVAIL's format: two countries, 15 months, a weekly pattern and a December peak."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2018-01-01", "2019-03-31", freq="D")
    by_month = {}
    for i, d in enumerate(days):
        level = 1000 * (1.6 if d.month == 12 else 1.0) * (0.6 if d.dayofweek >= 5 else 1.0)
        for country, share in (("United Kingdom", 0.9), ("France", 0.1)):
            by_month.setdefault(f"{d:%Y-%m}", []).append({
                "country": country, "customer_id": int(rng.integers(1, 500)), "invoice": f"{i:06d}",
                "price": round(float(level * share * rng.uniform(0.9, 1.1)), 2), "stream_id": "1",
                "times_viewed": int(rng.integers(1, 20)), "year": d.year, "month": d.month, "day": d.day})
    for month, rows in by_month.items():
        (tmp_path / f"invoices-{month}.json").write_text(json.dumps(rows))
    return tmp_path
