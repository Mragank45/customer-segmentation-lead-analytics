"""Generate a small synthetic dataset so the project runs without any private data.

Creates two files in /data:
  - transactions.csv : one row per purchase (customer_id, order_date, amount)
  - leads.csv        : one row per lead (lead_id, source, region, created_date, converted, deal_value)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_transactions(rng: np.random.Generator, n_customers: int = 500) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-12-31")
    span_days = (end - start).days

    # Three hidden customer "types" so segmentation has real structure to find
    for cid in range(1, n_customers + 1):
        kind = rng.choice(["loyal", "occasional", "lapsed"], p=[0.2, 0.5, 0.3])
        if kind == "loyal":
            n_orders = rng.integers(10, 25)
            amount_mu, last_gap = 3000, rng.integers(0, 30)
        elif kind == "occasional":
            n_orders = rng.integers(2, 8)
            amount_mu, last_gap = 1200, rng.integers(20, 150)
        else:
            n_orders = rng.integers(1, 4)
            amount_mu, last_gap = 800, rng.integers(180, 330)

        last_day = span_days - last_gap
        days = np.sort(rng.integers(0, max(last_day, 1), size=n_orders))
        for d in days:
            amount = max(100, rng.normal(amount_mu, amount_mu * 0.3))
            rows.append((f"C{cid:04d}", start + pd.Timedelta(days=int(d)), round(float(amount), 2)))

    return pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])


def make_leads(rng: np.random.Generator, n_leads: int = 1500) -> pd.DataFrame:
    sources = ["Website", "Referral", "LinkedIn", "Cold Call", "Email Campaign", "Trade Show"]
    # Different sources convert at different (hidden) rates
    conv_rate = {
        "Website": 0.18, "Referral": 0.35, "LinkedIn": 0.15,
        "Cold Call": 0.06, "Email Campaign": 0.10, "Trade Show": 0.22,
    }
    regions = ["North", "South", "East", "West"]
    src = rng.choice(sources, size=n_leads, p=[0.25, 0.10, 0.20, 0.20, 0.15, 0.10])
    reg = rng.choice(regions, size=n_leads)
    created = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, size=n_leads), unit="D")
    converted = np.array([rng.random() < conv_rate[s] for s in src])
    deal_value = np.where(converted, rng.normal(2500, 600, size=n_leads).clip(500), 0).round(2)

    return pd.DataFrame({
        "lead_id": [f"L{i:05d}" for i in range(1, n_leads + 1)],
        "source": src,
        "region": reg,
        "created_date": created,
        "converted": converted.astype(int),
        "deal_value": deal_value,
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    make_transactions(rng).to_csv(DATA_DIR / "transactions.csv", index=False)
    make_leads(rng).to_csv(DATA_DIR / "leads.csv", index=False)
    print(f"Sample data written to {DATA_DIR}")


if __name__ == "__main__":
    main()
