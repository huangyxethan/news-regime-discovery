# Pull macro-entity headlines from WRDS RavenPack DJ Global Macro feed and
# build daily features with the SAME schema as 02_news_features.py output.
#
# Output: _data/news_features_macro_daily.parquet
# Cache:  data/ravenpack_macro_cache_YYYY.parquet  (one per year)
#
# CLI:
#   python 02d_news_macro.py                # full pull 2000-2026, all years
#   python 02d_news_macro.py --year 2020    # single-year sanity check
#   python 02d_news_macro.py --years 2019 2020 2021   # subset
#
# Schema parity with 02_news_features.py:
#   css_mean, css_std, css_count, ess_mean, n_articles,
#   sent_pos_frac, sent_neg_frac, sent_net, css_mean_21
# Reading downstream by 06/07/08 requires zero changes.

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass


PREFERRED_PROJECT_DIR = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline"
PROJECT_DIR = (PREFERRED_PROJECT_DIR if os.path.exists(PREFERRED_PROJECT_DIR)
               else os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "_data")
IMAGES_DIR = os.path.join(PROJECT_DIR, "_images")
CACHE_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "news_features_macro_daily.parquet")

# RavenPack DJ Global Macro tables on WRDS — one per calendar year.
WRDS_SCHEMA = "ravenpack_dj"
TABLE_PREFIX = "rpa_djpr_global_macro_"   # → rpa_djpr_global_macro_2020 etc.

# Same column set as 02_news_features.py:74-77 (kept identical for parity).
SELECT_COLS = [
    "timestamp_utc", "relevance", "css", "event_sentiment_score",
    "rp_entity_id", "entity_type", "country_code", "news_type", "headline",
]

# Macro entity types (mirrors what was reviewed in the sanity sample):
# MKT = market, ORGA = organisation (central banks, IMF, OPEC…),
# PLCE = place, CMDT = commodity, CURR = currency, NATL = national economy.
MACRO_ENTITY_TYPES = ("MKT", "ORGA", "PLCE", "CMDT", "CURR", "NATL")

# News-type blacklist — mirrors 04_topic_models.py:96-100.
NEWS_TYPE_BLACKLIST = ("TABULAR-MATERIAL", "LEGAL", "SHARE-TRANSACTIONS",
                       "STATS", "STOCK-PRICES")

RELEVANCE_MIN = 75
COUNTRY_DEFAULT = "US"   # match 02_news_features.py geographic scope


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


def open_wrds():
    load_env_file(os.path.join(PROJECT_DIR, ".env"))
    user = os.environ.get("WRDS_USERNAME", "").strip()
    if not user or user == "your_wrds_username_here":
        raise SystemExit("WRDS_USERNAME missing in .env")
    os.environ.setdefault("PGUSER", user)
    pw = os.environ.get("WRDS_PASSWORD", "").strip()
    if pw and pw != "your_wrds_password_here":
        os.environ.setdefault("PGPASSWORD", pw)
    import wrds
    return wrds.Connection(wrds_username=user)


def pull_year(conn, year: int, country: str = COUNTRY_DEFAULT) -> pd.DataFrame:
    """Pull one yearly partition, filtered to macro entities + relevance >= 75."""
    table = f"{WRDS_SCHEMA}.{TABLE_PREFIX}{year}"
    entity_in = "(" + ", ".join(f"'{e}'" for e in MACRO_ENTITY_TYPES) + ")"
    news_not_in = "(" + ", ".join(f"'{n}'" for n in NEWS_TYPE_BLACKLIST) + ")"
    select_list = ", ".join(SELECT_COLS)
    country_clause = f"AND country_code = '{country}'" if country else ""
    query = f"""
        SELECT {select_list}
        FROM {table}
        WHERE relevance >= {RELEVANCE_MIN}
          AND entity_type IN {entity_in}
          AND news_type NOT IN {news_not_in}
          {country_clause}
    """
    print(f"[macro] querying {table} ...", flush=True)
    df = conn.raw_sql(query, date_cols=["timestamp_utc"])
    print(f"[macro]   {year}: {len(df):,} rows")
    return df


def pull_or_load_year(conn, year: int) -> pd.DataFrame:
    """Use yearly local cache when present; otherwise pull from WRDS."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"ravenpack_macro_cache_{year}.parquet")
    if os.path.exists(cache):
        print(f"[macro] cache hit {cache}")
        return pd.read_parquet(cache)
    if conn is None:
        print(f"[macro] cache miss and no WRDS connection — year {year} skipped")
        return pd.DataFrame()
    df = pull_year(conn, year)
    if not df.empty:
        df.to_parquet(cache, index=False)
        print(f"[macro]   wrote {cache}")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily aggregation — same column names as 02_news_features.py:111-119."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.normalize()
    df["css"] = pd.to_numeric(df["css"], errors="coerce")
    df["ess"] = pd.to_numeric(df["event_sentiment_score"], errors="coerce")
    return df.groupby("date").agg(
        css_mean=("css", "mean"),
        css_std=("css", "std"),
        css_count=("css", "count"),
        ess_mean=("ess", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
        n_articles=("rp_entity_id", "count"),
        sent_pos_frac=("css", lambda x: (x > 0.1).mean()),
        sent_neg_frac=("css", lambda x: (x < -0.1).mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="WRDS macro news daily features.")
    parser.add_argument("--year", type=int, default=None,
                        help="Pull a single year (sanity test).")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Pull specific years.")
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2026,
                        help="Inclusive end year.")
    parser.add_argument("--no-output", action="store_true",
                        help="Skip writing the aggregated parquet (cache only).")
    args, _ = parser.parse_known_args()

    if os.path.exists(OUTPUT_PATH) and args.year is None and args.years is None:
        print(f"Skipping macro daily features; output exists: {OUTPUT_PATH}")
        return

    if args.year is not None:
        year_range = [args.year]
    elif args.years is not None:
        year_range = sorted(set(args.years))
    else:
        year_range = list(range(args.start, args.end + 1))

    # Check if we even need WRDS (all caches present?)
    cache_paths = [os.path.join(CACHE_DIR, f"ravenpack_macro_cache_{y}.parquet")
                   for y in year_range]
    all_cached = all(os.path.exists(p) for p in cache_paths)
    conn = None if all_cached else open_wrds()

    frames = []
    try:
        for yr in year_range:
            df_yr = pull_or_load_year(conn, yr)
            if not df_yr.empty:
                frames.append(df_yr)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    if not frames:
        raise SystemExit("[macro] no data pulled — abort.")

    raw = pd.concat(frames, ignore_index=True)
    print(f"[macro] raw concatenated: {len(raw):,} rows  "
          f"({raw['timestamp_utc'].min()} → {raw['timestamp_utc'].max()})")

    if args.no_output:
        print("[macro] --no-output set; skipping aggregate parquet write.")
        return

    daily = aggregate_daily(raw)
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily["sent_net"] = daily["sent_pos_frac"] - daily["sent_neg_frac"]
    daily["css_mean_21"] = daily["css_mean"].rolling(21, min_periods=10).mean()
    print(f"[macro] daily aggregate: {daily.shape}  "
          f"({daily.index.min()} → {daily.index.max()})")
    print(daily.describe().round(4))

    os.makedirs(DATA_DIR, exist_ok=True)
    daily.to_parquet(OUTPUT_PATH)
    print(f"[macro] wrote {OUTPUT_PATH}")

    _write_sanity_image(daily)


def _write_sanity_image(daily: pd.DataFrame) -> None:
    """Mirror 02_news_features.py:135-151 sanity plot for visual parity."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        return
    events = [("Dot-com peak", "2000-03-10"), ("9/11", "2001-09-11"),
              ("GFC", "2008-09-15"), ("COVID", "2020-03-16"),
              ("Rate hikes", "2022-03-16")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(daily.index, daily["css_mean_21"], linewidth=0.8,
                 color="steelblue", label="macro CSS (21d MA)")
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Mean CSS"); axes[0].legend(fontsize=8)
    axes[1].plot(daily.index, daily["n_articles"].rolling(21).mean(),
                 linewidth=0.8, color="darkorange",
                 label="macro article count (21d MA)")
    axes[1].set_ylabel("n_articles"); axes[1].legend(fontsize=8)
    for ax in axes:
        for _, dt in events:
            ax.axvline(pd.Timestamp(dt), color="red", alpha=0.35, linewidth=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = os.path.join(IMAGES_DIR, "news_css_sanity_check_macro.png")
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close("all")
    print(f"[macro] wrote {out}")


if __name__ == "__main__":
    main()
