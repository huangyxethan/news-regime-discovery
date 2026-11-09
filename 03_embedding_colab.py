# Auto-generated from 03_embedding_colab.ipynb
# Intended to run in Google Colab with the original Google Drive paths preserved.
#
# Sections:
#   1. FinBERT embeddings on RavenPack DJPR (firm-level, US, COMP, always)
#   2. FinBERT embeddings on WRDS macro corpus (--enable-macro)

import argparse
import os
import random
import sys
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


parser = argparse.ArgumentParser(description="Daily FinBERT embeddings.")
parser.add_argument("--enable-macro", action="store_true",
                    help="Also build macro-corpus FinBERT embeddings from "
                         "data/ravenpack_macro_cache_YYYY.parquet (requires "
                         "02d_news_macro.py to have run).")
args, _ = parser.parse_known_args()

PROJECT_DIR = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline"
if not os.path.exists(PROJECT_DIR):
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "_data")
PARQUET_PATH = os.path.join(PROJECT_DIR, "data", "ravenpack_djpr.parquet")
MACRO_CACHE_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_V1 = os.path.join(DATA_DIR, "daily_finbert_embeddings.npz")
OUTPUT_MACRO = os.path.join(DATA_DIR, "daily_finbert_embeddings_macro.npz")
MAX_HEADLINES_PER_DAY_MACRO = 30   # macro has fewer per-day; smaller cap is cheaper and still representative

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

MAX_HEADLINES_PER_DAY = 50
BATCH_SIZE = 64
MODEL_NAME = "ProsusAI/finbert"


def _run_embeddings(entity_types: list[str], label: str, output_path: str) -> None:
    if os.path.exists(output_path):
        print(f"Skipping FinBERT {label}; output already exists: {output_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== FinBERT {label} ===  device={device}")

    use_in_filter = len(entity_types) > 1

    def load_headlines_year(year: int) -> pd.DataFrame:
        filters = [
            ("timestamp_utc", ">=", datetime(year, 1, 1)),
            ("timestamp_utc", "<", datetime(year + 1, 1, 1)),
            ("country_code", "=", "US"),
        ]
        if use_in_filter:
            filters.insert(2, ("entity_type", "in", list(entity_types)))
        else:
            filters.insert(2, ("entity_type", "=", entity_types[0]))
        df = pd.read_parquet(
            PARQUET_PATH,
            filters=filters,
            columns=["timestamp_utc", "relevance", "headline"],
        )
        df = df[df["relevance"] >= 75].copy()
        df["date"] = pd.to_datetime(df["timestamp_utc"]).dt.normalize()
        df = df[df["headline"].notna() & (df["headline"].str.strip() != "")]
        return df[["date", "headline"]]

    day_to_headlines: dict[pd.Timestamp, list[str]] = {}
    for yr in range(2000, 2026):
        df_yr = load_headlines_year(yr)
        for date, grp in df_yr.groupby("date"):
            texts = grp["headline"].dropna().tolist()
            if date in day_to_headlines:
                day_to_headlines[date].extend(texts)
            else:
                day_to_headlines[date] = texts
        print(f"  {yr}: {df_yr['date'].nunique()} days loaded")

    all_dates = sorted(day_to_headlines.keys())
    print(f"Total trading days with headlines: {len(all_dates)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    def embed_texts(texts: list[str]) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model(**enc)
            all_embeds.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(all_embeds)

    daily_embeddings = np.zeros((len(all_dates), 768), dtype=np.float32)
    for i, dt in enumerate(all_dates):
        texts = day_to_headlines[dt]
        if len(texts) > MAX_HEADLINES_PER_DAY:
            texts = random.sample(texts, MAX_HEADLINES_PER_DAY)
        daily_embeddings[i] = embed_texts(texts).mean(axis=0)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(all_dates)} days done")

    date_strings = np.array([str(d.date()) for d in all_dates])
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(output_path, embeddings=daily_embeddings, dates=date_strings)
    print(f"Saved: {output_path}")
    print(f"  embeddings: {daily_embeddings.shape}")


def _run_embeddings_macro(output_path: str) -> None:
    """Macro variant: reads pre-filtered yearly caches written by 02d_news_macro.py.

    Mirrors _run_embeddings() body; only the year-loader differs (yearly parquet
    cache instead of single big DJPR parquet with predicate pushdown).
    Per-day cap is MAX_HEADLINES_PER_DAY_MACRO (30) instead of 50 — macro feed
    has fewer headlines per day on average, smaller cap is faster and still
    representative.
    """
    if os.path.exists(output_path):
        print(f"Skipping FinBERT macro; output already exists: {output_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== FinBERT macro ===  device={device}")

    day_to_headlines: dict[pd.Timestamp, list[str]] = {}
    for yr in range(2000, 2027):
        cache_yr = os.path.join(MACRO_CACHE_DIR, f"ravenpack_macro_cache_{yr}.parquet")
        if not os.path.exists(cache_yr):
            continue
        df_yr = pd.read_parquet(cache_yr, columns=["timestamp_utc", "headline"])
        df_yr = df_yr[df_yr["headline"].notna() & (df_yr["headline"].str.strip() != "")]
        df_yr["date"] = pd.to_datetime(df_yr["timestamp_utc"]).dt.normalize()
        for date, grp in df_yr.groupby("date"):
            texts = grp["headline"].dropna().tolist()
            if date in day_to_headlines:
                day_to_headlines[date].extend(texts)
            else:
                day_to_headlines[date] = texts
        print(f"  {yr}: {df_yr['date'].nunique()} days loaded")

    all_dates = sorted(day_to_headlines.keys())
    print(f"Total days with macro headlines: {len(all_dates)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    def embed_texts(texts: list[str]) -> np.ndarray:
        all_embeds = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                chunk, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model(**enc)
            all_embeds.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(all_embeds)

    daily_embeddings = np.zeros((len(all_dates), 768), dtype=np.float32)
    for i, dt in enumerate(all_dates):
        texts = day_to_headlines[dt]
        if len(texts) > MAX_HEADLINES_PER_DAY_MACRO:
            texts = random.sample(texts, MAX_HEADLINES_PER_DAY_MACRO)
        daily_embeddings[i] = embed_texts(texts).mean(axis=0)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(all_dates)} days done")

    date_strings = np.array([str(d.date()) for d in all_dates])
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(output_path, embeddings=daily_embeddings, dates=date_strings)
    print(f"Saved: {output_path}")
    print(f"  embeddings: {daily_embeddings.shape}")


_run_embeddings(["COMP"], "v1", OUTPUT_V1)
if args.enable_macro:
    _run_embeddings_macro(OUTPUT_MACRO)
