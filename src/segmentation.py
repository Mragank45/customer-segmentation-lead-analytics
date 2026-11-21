"""Customer segmentation using RFM (Recency, Frequency, Monetary) + K-Means."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def build_rfm(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: days since last order, order count, total spend."""
    snapshot = transactions["order_date"].max() + pd.Timedelta(days=1)
    rfm = transactions.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("amount", "sum"),
    )
    return rfm.round(2)


def cluster_customers(rfm: pd.DataFrame, k: int = 3, seed: int = 42) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    rfm = rfm.copy()
    rfm["cluster"] = model.fit_predict(scaled)
    return rfm


def label_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """Give each cluster a business-friendly name based on its average behaviour."""
    profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    names = {}
    names[profile["monetary"].idxmax()] = "Champions (high value)"
    names[profile["recency"].idxmax()] = "At Risk (inactive)"
    for c in profile.index:
        names.setdefault(c, "Potential (occasional)")
    rfm = rfm.copy()
    rfm["segment"] = rfm["cluster"].map(names)
    return rfm


def summarize(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("segment").agg(
        customers=("monetary", "size"),
        avg_recency_days=("recency", "mean"),
        avg_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    ).round(1)
    summary["revenue_share_pct"] = (100 * summary["total_revenue"] / summary["total_revenue"].sum()).round(1)
    return summary.sort_values("total_revenue", ascending=False)


def plot_segments(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    summary["customers"].plot.bar(ax=axes[0], color="#4C78A8")
    axes[0].set_title("Customers per segment")
    axes[0].set_xlabel("")
    summary["revenue_share_pct"].plot.bar(ax=axes[1], color="#F58518")
    axes[1].set_title("Revenue share (%) per segment")
    axes[1].set_xlabel("")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "segments.png", dpi=150)
    plt.close(fig)


def run() -> pd.DataFrame:
    OUT_DIR.mkdir(exist_ok=True)
    tx = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["order_date"])
    rfm = label_clusters(cluster_customers(build_rfm(tx)))
    summary = summarize(rfm)
    rfm.to_csv(OUT_DIR / "customer_segments.csv")
    summary.to_csv(OUT_DIR / "segment_summary.csv")
    plot_segments(summary)
    return summary


if __name__ == "__main__":
    print(run())
