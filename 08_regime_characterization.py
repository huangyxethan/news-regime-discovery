# Auto-generated from 08_regime_characterization.ipynb
# Intended to run in Google Colab with the original Google Drive paths preserved.
#
# Phase 4 refactor: loops over PIPELINE_LIST (4 canonical pipelines) and
# produces per-pipeline timeline + share figures, plus one master 4-panel
# comparison figure for the report (regime_tour_4panel.png), and one master
# regime-stats parquet (regime_characterization_master.parquet).

import os

try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    # Allows syntax checks outside Colab; data paths still point to Google Drive.
    pass


def display(obj):
    """Notebook display fallback for script runs."""
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


# %% [markdown]
# # 08 — Regime Characterization Tour (4 canonical pipelines)


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import random
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

DATA_DIR        = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data"
IMAGES_DIR      = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_images"
PARQUET_PATH    = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/data/ravenpack_djpr.parquet"
MACRO_CACHE_DIR = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/data"

os.makedirs(IMAGES_DIR, exist_ok=True)

# ── Pipelines to characterize ────────────────────────────────────────────────
# 3 method pathways from proposal §4: price-only / news-only / hybrid.
# Plus DJPR-vs-Macro contrast (firm-level news vs aggregate news).
PIPELINE_LIST = [
        ("price-HMM",              "regimes_price.parquet",            "hmm_k5_full"),
        ("text_scalar_smooth-HMM", "regimes_text_scalar.parquet",      "hmm"),
        ("macro_only-HMM",         "regimes_macro_supplement.parquet", "macro_only"),
        ("djpr_plus_macro-HMM",    "regimes_macro_supplement.parquet", "djpr_plus_macro"),
        ("ft_news_fusion_best",    "regimes_ft_news_fusion_best.parquet", "regime"),
        ("ft_full_stack_best",     "regimes_ft_full_stack_best.parquet",  "regime"),
    ]

TOP_N = 10   # headlines per regime
EPISODES = [
    ('Dot-com burst',  '2000-03-10', '2002-10-09'),
    ('9/11',           '2001-09-11', '2001-09-21'),
    ('GFC',            '2007-10-09', '2009-03-09'),
    ('COVID crash',    '2020-02-19', '2020-03-23'),
    ('Rate hike cycle','2022-01-03', '2022-12-31'),
]


# %% [markdown]
# ## Shared data loads (price + news + LDA + macro headline cache)


# %%
mkt  = pd.read_parquet(f"{DATA_DIR}/market_features.parquet")
news = pd.read_parquet(f"{DATA_DIR}/news_features_daily.parquet")
print(f"market: {mkt.shape}  news: {news.shape}")

# DJPR + Macro LDA topic files (used to pick "top days" per regime)
def _find_lda(macro_first: bool):
    cands = (
        [f"{DATA_DIR}/lda_topics_macro_k{k}_daily.parquet" for k in [15, 20]]
        if macro_first else []
    ) + [f"{DATA_DIR}/lda_topics_k{k}_daily.parquet" for k in [15, 20, 10]]
    for p in cands:
        if os.path.exists(p):
            return pd.read_parquet(p), p
    return None, None

_djpr_lda, _djpr_lda_p = _find_lda(False)
_macro_lda, _macro_lda_p = _find_lda(True)
if _djpr_lda is not None:
    print(f"djpr_lda: {os.path.basename(_djpr_lda_p)}  {_djpr_lda.shape}")
if _macro_lda is not None and _macro_lda_p != _djpr_lda_p:
    print(f"macro_lda: {os.path.basename(_macro_lda_p)}  {_macro_lda.shape}")

# Macro FinBERT embeddings (semantic retrieval for macro-mode pipelines)
MACRO_EMBED_PATH = f"{DATA_DIR}/daily_finbert_embeddings_macro.npz"
if os.path.exists(MACRO_EMBED_PATH):
    _npz = np.load(MACRO_EMBED_PATH, allow_pickle=True)
    _macro_embed = pd.DataFrame(
        _npz["embeddings"], index=pd.to_datetime(_npz["dates"])
    )
    print(f"macro FinBERT: {_macro_embed.shape}")
else:
    _macro_embed = None


# %% [markdown]
# ## Helpers


# %%
def load_regime_labels(file: str, col: str) -> pd.DataFrame:
    """Load regime labels, drop NaN, cast to int."""
    df = pd.read_parquet(f"{DATA_DIR}/{file}")[[col]].rename(columns={col: "regime"})
    df = df.dropna(subset=["regime"])
    df["regime"] = df["regime"].astype(int)
    return df


def per_regime_stats(regimes: pd.DataFrame, mkt: pd.DataFrame,
                     news: pd.DataFrame) -> pd.DataFrame:
    joined = (mkt.join(news[["css_mean", "n_articles"]], how="left")
                 .join(regimes, how="inner")
                 .dropna(subset=["regime"]))
    rows = []
    for r, grp in joined.groupby("regime"):
        ann_ret  = grp["ret"].mean() * 252
        ann_vol  = grp["ret"].std() * np.sqrt(252)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum      = np.exp(grp["ret"].cumsum())
        drawdown = ((cum - cum.cummax()) / cum.cummax()).min()
        rows.append({
            "regime":       int(r),
            "n_days":       len(grp),
            "date_from":    str(grp.index.min().date()),
            "date_to":      str(grp.index.max().date()),
            "ann_ret":      round(ann_ret, 4),
            "ann_vol":      round(ann_vol, 4),
            "sharpe":       round(sharpe, 3),
            "max_drawdown": round(drawdown, 4),
            "avg_vix":      round(grp["vix"].mean(), 2),
            "avg_css":      round(grp["css_mean"].mean(), 4),
        })
    return pd.DataFrame(rows).set_index("regime")


def plot_regime_timeline(name: str, regimes: pd.DataFrame, mkt: pd.DataFrame,
                         out_path: str, ax=None):
    """Cum return shaded by regime + episode vlines. Returns the matplotlib axes."""
    joined = mkt[["ret"]].join(regimes, how="inner").dropna(subset=["regime"])
    cum = np.exp(joined["ret"].cumsum())
    labels = joined["regime"].values.astype(int)
    dates = joined.index.to_numpy()
    K = labels.max() + 1
    colors = plt.cm.Set1.colors

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3.5))
        standalone = True
    else:
        standalone = False

    ax.plot(dates, cum, linewidth=0.6, color="black", zorder=3)
    for r in range(K):
        mask = labels == r
        ax.fill_between(dates, 0, cum, where=mask, alpha=0.3,
                        color=colors[r % len(colors)], label=f"R{r}")
    for ep_name, start, end in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.08, color="gray", zorder=0)
    ax.set_ylabel("S&P 500 ($1→)")
    ax.legend(fontsize=7, loc="upper left", ncol=K)
    ax.set_title(name, fontsize=10, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    if standalone:
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close("all")


def plot_regime_shares(name: str, regimes: pd.DataFrame, out_path: str):
    annual = regimes.copy()
    annual["year"] = annual.index.year
    shares = annual.groupby(["year", "regime"]).size().unstack(fill_value=0)
    shares = shares.div(shares.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(14, 4))
    shares.plot.area(ax=ax, colormap="Set1", alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Regime share")
    ax.set_title(f"Annual regime composition — {name}")
    ax.legend([f"R{r}" for r in shares.columns], fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close("all")


def top_dates_by_lda_dominance(regimes: pd.DataFrame, lda_day: pd.DataFrame,
                                K: int, per_regime: int = 50):
    """For each regime: find dominant topic, then top N days by that topic weight."""
    if lda_day is None:
        return {}
    joined = lda_day.join(regimes, how="inner").dropna(subset=["regime"])
    joined["regime"] = joined["regime"].astype(int)
    topic_cols = list(lda_day.columns)
    out = {}
    for r in range(K):
        grp = joined[joined["regime"] == r]
        if grp.empty:
            continue
        dom = grp[topic_cols].mean().idxmax()
        out[r] = (dom, grp[dom].nlargest(per_regime).index.tolist())
    return out


def load_headlines_for_dates(dates_needed: pd.DatetimeIndex, macro_mode: bool) -> pd.DataFrame:
    """Fetch headlines for given dates from DJPR parquet or macro yearly caches."""
    if len(dates_needed) == 0:
        return pd.DataFrame(columns=["date", "css", "headline"])
    years = sorted(set(dates_needed.year))
    frames = []
    if macro_mode:
        for yr in tqdm(years, desc="Loading macro headline years"):
            cache_yr = os.path.join(MACRO_CACHE_DIR, f"ravenpack_macro_cache_{yr}.parquet")
            if not os.path.exists(cache_yr):
                continue
            batch = pd.read_parquet(cache_yr, columns=["timestamp_utc", "css", "headline"])
            batch["date"] = pd.to_datetime(batch["timestamp_utc"]).dt.normalize()
            batch = batch[batch["date"].isin(dates_needed)]
            batch["css"] = pd.to_numeric(batch["css"], errors="coerce")
            frames.append(batch[["date", "css", "headline"]])
    else:
        for yr in tqdm(years, desc="Loading DJPR headline years"):
            batch = pd.read_parquet(
                PARQUET_PATH,
                filters=[
                    ("timestamp_utc", ">=", datetime(yr, 1, 1)),
                    ("timestamp_utc", "<",  datetime(yr + 1, 1, 1)),
                    ("entity_type",   "=",  "COMP"),
                    ("country_code",  "=",  "US"),
                ],
                columns=["timestamp_utc", "relevance", "css", "headline"],
            )
            batch = batch[batch["relevance"] >= 75].copy()
            batch["date"] = pd.to_datetime(batch["timestamp_utc"]).dt.normalize()
            batch = batch[batch["date"].isin(dates_needed)]
            batch["css"] = pd.to_numeric(batch["css"], errors="coerce")
            frames.append(batch[["date", "css", "headline"]])
    if not frames:
        return pd.DataFrame(columns=["date", "css", "headline"])
    return pd.concat(frames, ignore_index=True)


def semantic_top_dates_via_finbert(regimes: pd.DataFrame, K: int, top_k: int = 30):
    """Use macro FinBERT day-mean embeddings to find top-K days closest to per-regime centroid."""
    if _macro_embed is None:
        return {}
    joined = _macro_embed.join(regimes, how="inner").dropna(subset=["regime"])
    joined["regime"] = joined["regime"].astype(int)
    out = {}
    for r in range(K):
        grp = joined[joined["regime"] == r]
        if grp.empty:
            continue
        vecs = grp.drop(columns="regime").values.astype(np.float32)
        centroid = vecs.mean(axis=0)
        norms = np.linalg.norm(vecs, axis=1) + 1e-9
        cnorm = np.linalg.norm(centroid) + 1e-9
        sims = (vecs @ centroid) / (norms * cnorm)
        order = np.argsort(-sims)[:top_k]
        out[r] = pd.DatetimeIndex(grp.index[order])
    return out


# %% [markdown]
# ## Per-pipeline characterization function


# %%
def characterize_pipeline(name: str, file: str, col: str) -> pd.DataFrame:
    """Full characterization for one pipeline. Returns per-regime stats DataFrame."""
    print(f"\n{'='*72}\nCharacterizing: {name}  ({file} [{col}])\n{'='*72}")
    macro_mode = "macro" in file.lower()
    print(f"  Mode: {'MACRO' if macro_mode else 'DJPR'} corpus")

    # 1. Labels + per-regime market/sentiment stats
    regimes = load_regime_labels(file, col)
    K = int(regimes["regime"].max()) + 1
    print(f"  {len(regimes)} days, K={K}")
    stats = per_regime_stats(regimes, mkt, news)
    display(stats)

    # 2. Timeline figure (per-pipeline)
    safe = name.replace("-", "_").replace(" ", "_")
    plot_regime_timeline(name, regimes, mkt, f"{IMAGES_DIR}/regime_tl_{safe}.png")
    print(f"  Saved: regime_tl_{safe}.png")

    # 3. Annual regime share figure
    plot_regime_shares(name, regimes, f"{IMAGES_DIR}/regime_shares_{safe}.png")
    print(f"  Saved: regime_shares_{safe}.png")

    # 4. Top headlines per regime — via LDA-dominant-topic days
    lda_day = _macro_lda if macro_mode and _macro_lda is not None else _djpr_lda
    if lda_day is None:
        print("  WARNING: no LDA file found, skipping top-headlines step")
    else:
        dom_topic_dates = top_dates_by_lda_dominance(regimes, lda_day, K)
        all_dates = pd.DatetimeIndex(sorted({d for _, ds in dom_topic_dates.values() for d in ds}))
        hl_df = load_headlines_for_dates(all_dates, macro_mode)
        hl_df = hl_df.join(regimes, on="date", how="inner").dropna(subset=["regime"])
        hl_df["regime"] = hl_df["regime"].astype(int)
        for r in range(K):
            if r not in dom_topic_dates:
                continue
            dom, _ = dom_topic_dates[r]
            regime_hl = hl_df[hl_df["regime"] == r]
            regime_hl = regime_hl[regime_hl["headline"].str.split().str.len() >= 6]
            if regime_hl.empty:
                continue
            sample = regime_hl.sample(min(TOP_N, len(regime_hl)), random_state=r+1)
            print(f"\n  Regime {r} (dominant LDA topic: {dom}) — {len(sample)} headlines:")
            for _, row in sample.iterrows():
                print(f"    [{row['date'].date()}] {row['headline']}")

    # 5. Macro FinBERT semantic retrieval (only for macro pipelines + if embeddings exist)
    if macro_mode and _macro_embed is not None:
        sem_dates = semantic_top_dates_via_finbert(regimes, K)
        # Reuse hl_df if available; else load fresh
        all_sem_dates = pd.DatetimeIndex(sorted({d for ds in sem_dates.values() for d in ds}))
        sem_hl = load_headlines_for_dates(all_sem_dates, macro_mode=True)
        sem_hl = sem_hl.join(regimes, on="date", how="inner").dropna(subset=["regime"])
        sem_hl["regime"] = sem_hl["regime"].astype(int)
        print(f"\n  ── FinBERT-centroid semantic retrieval ──")
        for r in range(K):
            if r not in sem_dates:
                continue
            regime_dates = sem_dates[r]
            regime_hl = sem_hl[(sem_hl["regime"] == r) & (sem_hl["date"].isin(regime_dates))]
            regime_hl = regime_hl[regime_hl["headline"].str.split().str.len() >= 6]
            sample = regime_hl.head(TOP_N)
            if sample.empty:
                continue
            print(f"\n  Regime {r} — top {len(sample)} headlines (semantic centroid):")
            for _, row in sample.iterrows():
                print(f"    [{row['date'].date()}] {row['headline']}")

    # Attach pipeline name + return for master concat
    stats = stats.reset_index()
    stats["pipeline"] = name
    return stats


# %% [markdown]
# ## Run tour over all 4 pipelines


# %%
all_stats_frames = []
for _name, _file, _col in PIPELINE_LIST:
    try:
        _df = characterize_pipeline(_name, _file, _col)
        all_stats_frames.append(_df)
    except Exception as exc:
        print(f"  ERROR on {_name}: {exc}")

master = pd.concat(all_stats_frames, ignore_index=True)
print(f"\n{'='*72}")
print(f"Master regime stats: {master.shape}")
print(f"Pipelines: {master['pipeline'].unique().tolist()}")
print(f"{'='*72}")
master.to_parquet(f"{DATA_DIR}/regime_characterization_master.parquet")
print(f"Saved: {DATA_DIR}/regime_characterization_master.parquet")


# %% [markdown]
# ## Master 4-panel comparison figure (the report headline)


# %%
fig, axes = plt.subplots(len(PIPELINE_LIST), 1,
                         figsize=(16, 3.2 * len(PIPELINE_LIST)),
                         sharex=True)
if len(PIPELINE_LIST) == 1:
    axes = [axes]
for ax, (name, file, col) in zip(axes, PIPELINE_LIST):
    try:
        regimes = load_regime_labels(file, col)
        plot_regime_timeline(name, regimes, mkt, out_path=None, ax=ax)
    except Exception as exc:
        ax.text(0.5, 0.5, f"{name}: {exc}", ha="center", va="center", transform=ax.transAxes)
# Add episode labels only on top panel
for ep_name, start, _ in EPISODES:
    axes[0].text(pd.Timestamp(start), axes[0].get_ylim()[1] * 0.95, ep_name,
                 fontsize=7, rotation=90, va="top", color="gray")
plt.tight_layout()
plt.savefig(f"{IMAGES_DIR}/regime_tour_4panel.png", dpi=120)
plt.close("all")
print(f"Saved: {IMAGES_DIR}/regime_tour_4panel.png")


# %% [markdown]
# ## Pivot view: regime stats by pipeline


# %%
pivot_sharpe = master.pivot_table(index="pipeline", columns="regime",
                                   values="sharpe", aggfunc="first")
pivot_ndays  = master.pivot_table(index="pipeline", columns="regime",
                                   values="n_days", aggfunc="first")
print("\n=== Per-regime Sharpe by pipeline ===")
display(pivot_sharpe.round(2))
print("\n=== Per-regime n_days by pipeline ===")
display(pivot_ndays.fillna(0).astype(int))
