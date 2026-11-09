# Auto-generated from 02_news_features.ipynb
# Intended to run in Google Colab with the original Google Drive paths preserved.
#
# RavenPack DJPR (firm-level, US, COMP entities) daily news features.

import os
from datetime import datetime

try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass


def display(obj):
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


PROJECT_DIR = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline"
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_env_file(os.path.join(PROJECT_DIR, ".env"))

DATA_DIR = os.path.join(PROJECT_DIR, "_data")
IMAGES_DIR = os.path.join(PROJECT_DIR, "_images")
PARQUET_PATH = os.path.join(PROJECT_DIR, "data", "ravenpack_djpr.parquet")
OUTPUT_V1 = os.path.join(DATA_DIR, "news_features_daily.parquet")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

COLS = [
    "timestamp_utc", "relevance", "css", "event_sentiment_score",
    "rp_entity_id", "entity_type", "country_code",
]
EVENTS = [
    ("Dot-com peak", "2000-03-10"),
    ("9/11", "2001-09-11"),
    ("GFC", "2008-09-15"),
    ("COVID", "2020-03-16"),
    ("Rate hikes", "2022-03-16"),
]
CHECK_DATES = ["2008-09-15", "2008-09-16", "2020-03-16", "2020-03-17"]


def _aggregate_ravenpack(entity_types, label: str, output_path: str, sanity_image: str) -> None:
    if os.path.exists(output_path):
        print(f"Skipping RavenPack {label}; output already exists: {output_path}")
        return

    use_in_filter = len(entity_types) > 1
    print(f"\n=== RavenPack {label} ===")

    def agg_year(year: int) -> pd.DataFrame:
        filters = [
            ("timestamp_utc", ">=", datetime(year, 1, 1)),
            ("timestamp_utc", "<", datetime(year + 1, 1, 1)),
            ("country_code", "=", "US"),
        ]
        if use_in_filter:
            filters.insert(2, ("entity_type", "in", list(entity_types)))
        else:
            filters.insert(2, ("entity_type", "=", entity_types[0]))
        batch = pd.read_parquet(PARQUET_PATH, filters=filters, columns=COLS)
        batch = batch[batch["relevance"] >= 75].copy()
        batch["date"] = pd.to_datetime(batch["timestamp_utc"]).dt.normalize()
        batch["css"] = pd.to_numeric(batch["css"], errors="coerce")
        batch["ess"] = pd.to_numeric(batch["event_sentiment_score"], errors="coerce")
        return batch.groupby("date").agg(
            css_mean=("css", "mean"),
            css_std=("css", "std"),
            css_count=("css", "count"),
            ess_mean=("ess", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
            n_articles=("rp_entity_id", "count"),
            sent_pos_frac=("css", lambda x: (x > 0.1).mean()),
            sent_neg_frac=("css", lambda x: (x < -0.1).mean()),
        )

    frames = []
    for yr in range(2000, 2026):
        df_yr = agg_year(yr)
        frames.append(df_yr)
        print(f"  {yr}: {len(df_yr)} days")

    news = pd.concat(frames).sort_index()
    news.index = pd.to_datetime(news.index)
    news.index.name = "date"
    news["sent_net"] = news["sent_pos_frac"] - news["sent_neg_frac"]
    news["css_mean_21"] = news["css_mean"].rolling(21, min_periods=10).mean()
    print(f"Total: {len(news)} days  |  {news.index.min()} — {news.index.max()}")
    print(news.describe())

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(news.index, news["css_mean_21"], linewidth=0.8, color="steelblue", label="CSS (21d MA)")
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Mean CSS")
    axes[0].legend(fontsize=8)
    axes[1].plot(news.index, news["n_articles"].rolling(21).mean(), linewidth=0.8,
                 color="darkorange", label="Article count (21d MA)")
    axes[1].set_ylabel("n_articles")
    axes[1].legend(fontsize=8)
    for ax in axes:
        for _, dt in EVENTS:
            ax.axvline(pd.Timestamp(dt), color="red", alpha=0.35, linewidth=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    os.makedirs(IMAGES_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(sanity_image, dpi=100)
    plt.close("all")

    print(f"\nSpot-check crisis dates ({label}):")
    print(news.loc[news.index.isin(pd.to_datetime(CHECK_DATES)),
                   ["css_mean", "css_count", "n_articles"]])
    os.makedirs(DATA_DIR, exist_ok=True)
    news.to_parquet(output_path)
    print(f"Saved: {output_path}")


# RavenPack DJPR firm-level features (US, COMP entities, relevance >= 75)
_aggregate_ravenpack(
    entity_types=["COMP"],
    label="v1",
    output_path=OUTPUT_V1,
    sanity_image=os.path.join(IMAGES_DIR, "news_css_sanity_check.png"),
)
