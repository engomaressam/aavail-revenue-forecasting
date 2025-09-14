"""
Exploratory look at the AAVAIL invoices (cs-train): how revenue moves by month, which countries carry it,
and how it varies by weekday. Writes figures/monthly_revenue.png and results/country_revenue.csv.
Usage: python scripts/eda.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
logging.disable(logging.WARNING)
from data_ingestion import DataIngestion  # noqa: E402

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
df = DataIngestion(str(ROOT / "data" / "cs-train")).load_all_data()

print(f"{len(df):,} invoice lines, {df['invoice'].nunique():,} invoices, {df['customer_id'].nunique():,} customers, "
      f"{df['country'].nunique()} countries, {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}")

by_country = df.groupby("country").agg(revenue=("price", "sum"), invoices=("invoice", "nunique"),
                                       customers=("customer_id", "nunique")).sort_values("revenue", ascending=False)
by_country["share_pct"] = by_country["revenue"] / by_country["revenue"].sum() * 100
by_country.round(1).to_csv(ROOT / "results" / "country_revenue.csv")
print(by_country.head(10).round(1).to_string())

daily = df.groupby("date")["price"].sum()
weekday = daily.groupby(daily.index.dayofweek).mean()
print("mean daily revenue by weekday (Mon..Sun):", weekday.round(0).tolist())
print("days without any sale:", int((daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).fillna(0) == 0).sum()))

monthly = df.groupby(df["date"].dt.to_period("M"))["price"].sum()
monthly = monthly.iloc[1:]   # the first month (Nov 2017) starts on the 28th
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(monthly.index.to_timestamp(), monthly.values / 1000, width=20, color="#c45a12")
ax.set_ylabel("revenue (thousands)")
ax.set_title("AAVAIL monthly revenue, Dec 2017 to Jul 2019")
ax.grid(axis="y", alpha=.25)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "monthly_revenue.png", dpi=150)
print("wrote figures/monthly_revenue.png and results/country_revenue.csv")
