"""
Reproduce the numbers in the README.

1. Backtest: every approach on the same weekly forecast dates in the last 20% of the training months
   (cs-train, Nov 2017 - Jul 2019), for all countries together and for each of the ten largest countries.
2. Production check: the approach chosen on the backtest is retrained on all training months and scored on
   IBM's separate production months (cs-production, Aug - Dec 2019), next to the naive baseline.

Writes results/*.csv and figures/*.png.  Usage: python scripts/evaluate.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.disable(logging.WARNING)
from data_ingestion import DataIngestion          # noqa: E402
import evaluation as ev                          # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

train = DataIngestion(str(ROOT / "data" / "cs-train")).load_all_data()
prod = DataIngestion(str(ROOT / "data" / "cs-production")).load_all_data()
top10 = train.groupby("country")["price"].sum().sort_values(ascending=False).head(10).index.tolist()

# 1. backtest
tables = []
for name in [None] + top10:
    t = ev.compare(ev.daily_revenue(train, name))
    t.insert(0, "series", name or "All countries")
    tables.append(t)
    print(f"{name or 'All countries':16} best: {t.iloc[0]['model']:22} MAE {t.iloc[0]['mae']:>9,.0f}  "
          f"({t.iloc[0]['mae_vs_naive_pct']:+.0f}% vs naive)")
backtest = pd.concat(tables, ignore_index=True)
backtest.round(2).to_csv(RESULTS / "backtest_by_series.csv", index=False)

# which approach wins most often, and by how much on aggregate
rank = backtest.groupby("model").agg(mean_mae_vs_naive_pct=("mae_vs_naive_pct", "mean"),
                                     wins=("mae", lambda s: 0)).reset_index()
winners = backtest.loc[backtest.groupby("series")["mae"].idxmin(), "model"].value_counts()
rank["wins"] = rank["model"].map(winners).fillna(0).astype(int)
rank = rank.sort_values("mean_mae_vs_naive_pct").reset_index(drop=True)
rank.round(1).to_csv(RESULTS / "backtest_summary.csv", index=False)
print(rank.round(1).to_string(index=False))

# 2. production check, all countries together
chosen = rank.iloc[0]["model"] if rank.iloc[0]["model"] != "naive (last 30 days)" else rank.iloc[1]["model"]
print("chosen for production:", chosen)
full = pd.concat([train, prod], ignore_index=True)
series = ev.daily_revenue(full)
first_prod = series.index.get_loc(pd.Timestamp(prod["date"].min()))
last = len(series) - ev.HORIZON
test_origins = list(range(first_prod, last + 1, ev.STEP))
train_origins = list(range(ev.LOOKBACK, first_prod - ev.HORIZON + 1))
actual = ev.windows(series, test_origins)[1]
rows, preds = [], {}
for name in dict.fromkeys(["naive (last 30 days)", "seasonal naive (last year)", "blend: last 30 days + last year", chosen]):
    preds[name] = ev.MODELS[name](series, train_origins, test_origins)
    rows.append({"model": name, **ev.metrics(actual, preds[name])})
production = pd.DataFrame(rows)
production["test_origins"] = len(test_origins)
production.round(2).to_csv(RESULTS / "production_check.csv", index=False)
print("\nproduction months:\n", production.round(1).to_string(index=False))

# figure: the production check
dates = series.index[test_origins]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(dates, actual / 1000, "o-", color="#15120f", lw=2, label="actual")
styles = {"naive (last 30 days)": ("^:", "#8a8278"), "seasonal naive (last year)": ("v:", "#4a6fa5"),
          "blend: last 30 days + last year": ("s-", "#c45a12")}
for name, pred in preds.items():
    mk, col = styles.get(name, ("D--", "#2d7a4f"))
    ax.plot(dates, pred / 1000, mk, color=col, label=name, alpha=.9)
ax.set_ylabel("revenue in the next 30 days (thousands)")
ax.set_title("Production months: 30-day revenue forecasts from each forecast date")
ax.legend(frameon=False)
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(FIGURES / "production_check.png", dpi=150)

# figure: backtest error vs naive by series
piv = backtest.pivot(index="series", columns="model", values="mae_vs_naive_pct").reindex(["All countries"] + top10)
fig, ax = plt.subplots(figsize=(9, 4.6))
for col, mk in zip([c for c in piv.columns if not c.startswith("naive")], "osD^v"):
    ax.plot(piv[col].values, piv.index, mk, label=col, alpha=.85)
ax.axvline(0, color="#15120f", lw=1)
ax.set_xlabel("MAE relative to the naive baseline (%), lower is better")
ax.invert_yaxis()
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(FIGURES / "backtest_vs_naive.png", dpi=150)
print("\nwrote results/ and figures/")
