"""Lead generation analytics: which sources and regions bring in leads that convert?"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def conversion_by(leads: pd.DataFrame, column: str) -> pd.DataFrame:
    out = leads.groupby(column).agg(
        leads=("lead_id", "count"),
        converted=("converted", "sum"),
        revenue=("deal_value", "sum"),
    )
    out["conversion_rate_pct"] = (100 * out["converted"] / out["leads"]).round(1)
    out["revenue_per_lead"] = (out["revenue"] / out["leads"]).round(1)
    return out.sort_values("conversion_rate_pct", ascending=False)


def monthly_trend(leads: pd.DataFrame) -> pd.DataFrame:
    m = leads.assign(month=leads["created_date"].dt.to_period("M").astype(str))
    out = m.groupby("month").agg(leads=("lead_id", "count"), converted=("converted", "sum"))
    out["conversion_rate_pct"] = (100 * out["converted"] / out["leads"]).round(1)
    return out


def plot_sources(by_source: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    by_source["conversion_rate_pct"].plot.bar(ax=ax, color="#54A24B")
    ax.set_title("Lead conversion rate by source (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "lead_conversion_by_source.png", dpi=150)
    plt.close(fig)


def run() -> pd.DataFrame:
    OUT_DIR.mkdir(exist_ok=True)
    leads = pd.read_csv(DATA_DIR / "leads.csv", parse_dates=["created_date"])
    by_source = conversion_by(leads, "source")
    by_region = conversion_by(leads, "region")
    trend = monthly_trend(leads)
    by_source.to_csv(OUT_DIR / "lead_by_source.csv")
    by_region.to_csv(OUT_DIR / "lead_by_region.csv")
    trend.to_csv(OUT_DIR / "lead_monthly_trend.csv")
    plot_sources(by_source)
    return by_source


if __name__ == "__main__":
    print(run())
