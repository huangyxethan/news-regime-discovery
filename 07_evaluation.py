# Auto-generated from 07_evaluation.ipynb
# Intended to run in Google Colab with the original Google Drive paths preserved.

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
# # 07 — Evaluation: Four Dimensions


# %%
import hashlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

DATA_DIR   = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data"
IMAGES_DIR = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_images"
# Per-feature-type rolling windows — must match 06_text_hybrid_models.py.
# See the comment block at the top of 06 for tuning history.
ROLL_SCALAR  = 35   # DJPR scalar (news) — must match 06
ROLL_LDA     = 30   # LDA topic distributions
ROLL_BTOPIC  = 21   # BERTopic — must match 06
ROLL_MACRO   = 30   # macro scalar (§11)
ROLL_TEXT    = ROLL_LDA   # backward-compat alias for GDELT pickup
MIN_REGIME_DAYS = 250   # regimes smaller than this are flagged as micro-clusters

# ── Per-pipeline incremental cache ──────────────────────────────────────────
# Slow metrics (silhouette, rolling_ari, OOS) take ~5-30s each × 22 pipelines.
# We hash (regime labels + feat_cols) per pipeline; if unchanged from a previous
# run, reuse the cached metric.  Cache wiped automatically when labels change.
EVAL_CACHE_PATH = f"{DATA_DIR}/eval_cache.json"


def _pipeline_signature(labels, feat_cols) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(labels, dtype=np.int32).tobytes())
    h.update("|".join(str(c) for c in feat_cols).encode())
    return h.hexdigest()[:16]


def _load_eval_cache() -> dict:
    if not os.path.exists(EVAL_CACHE_PATH):
        return {}
    try:
        with open(EVAL_CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[07-cache] failed to load {EVAL_CACHE_PATH}: {exc}; starting fresh")
        return {}


def _save_eval_cache(cache: dict) -> None:
    with open(EVAL_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def _bucket(cache: dict, name: str, sig: str) -> dict:
    """Return the per-pipeline cache bucket; wipe + restart if sig changed."""
    if cache.get(name, {}).get("sig") != sig:
        cache[name] = {"sig": sig}
    return cache[name]


_eval_cache = _load_eval_cache()
print(f"[07-cache] loaded {len(_eval_cache)} pipeline entries from {EVAL_CACHE_PATH}")


# %% [markdown]
# ## 1. Load All Regime Files + Feature Sources


# %%
# ── Market features ─────────────────────────────────────────────────────────
mkt = pd.read_parquet(f"{DATA_DIR}/market_features.parquet")
PRICE_FEATS       = ["ret", "vol_21", "vix", "term_spread", "credit_spread"]
TEXT_SCALAR_FEATS = ["css_mean", "css_std", "n_articles", "ess_mean", "sent_net"]
trading_days = mkt[PRICE_FEATS].dropna().index

# ── Smoothed text-scalar (21-day rolling, matches nb06) ──────────────────────
news_raw = pd.read_parquet(f"{DATA_DIR}/news_features_daily.parquet")
news = news_raw[TEXT_SCALAR_FEATS].reindex(trading_days).rolling(
    window=ROLL_SCALAR, min_periods=15
).mean()

# ── LDA (headline-level, smoothed) ───────────────────────────────────────────
import os
for _k in [15, 20, 10]:
    _p = f"{DATA_DIR}/lda_topics_k{_k}_daily.parquet"
    if os.path.exists(_p):
        lda_raw = pd.read_parquet(_p).reindex(trading_days)
        print(f"LDA k={_k}: {lda_raw.shape}")
        break
lda = lda_raw.rolling(window=ROLL_TEXT, min_periods=10).mean()

# ── Hybrid aligned features (price + smoothed text) ───────────────────────────
hybrid_feats = mkt[PRICE_FEATS].join(news, how="inner").dropna()
HYBRID_FEATS = PRICE_FEATS + TEXT_SCALAR_FEATS

# ── Regime specifications ─────────────────────────────────────────────────────
# (filepath, column, pipeline_name, feature_source_df, feature_cols)
# Knob 4: 4 low-quality variants dropped from main eval (commented below).
# Kept: price-GMM (methodological baseline per proposal §1).
regime_specs = [
    # Price-only baselines
    (f"{DATA_DIR}/regimes_price.parquet",       "gmm_k3",      "price-GMM",             mkt,         PRICE_FEATS),
    (f"{DATA_DIR}/regimes_price.parquet",       "hmm_k5_full", "price-HMM",             mkt,         PRICE_FEATS),
    # Text-scalar (smoothed)
    (f"{DATA_DIR}/regimes_text_scalar.parquet", "gmm",         "text_scalar_smooth-GMM",news,        TEXT_SCALAR_FEATS),
    (f"{DATA_DIR}/regimes_text_scalar.parquet", "hmm",         "text_scalar_smooth-HMM",news,        TEXT_SCALAR_FEATS),
    # Text-LDA (headline-level, smoothed) — KMeans variant dropped (Knob 4: dur 51 < 60, no time structure)
    (f"{DATA_DIR}/regimes_text_lda.parquet",    "gmm",         "text_lda_smooth-GMM",   lda,         lda.columns.tolist()),
    (f"{DATA_DIR}/regimes_text_lda.parquet",    "hmm",         "text_lda_smooth-HMM",   lda,         lda.columns.tolist()),
    # (f"{DATA_DIR}/regimes_text_lda.parquet",    "kmeans",      "text_lda_smooth-KMeans",lda,         lda.columns.tolist()),
    # Hybrid PCA-weighted — GMM variant dropped (Knob 4: sil 0.16 bottom of table)
    # (f"{DATA_DIR}/regimes_hybrid.parquet",      "gmm",         "hybrid_pca-GMM",        hybrid_feats,HYBRID_FEATS),
    (f"{DATA_DIR}/regimes_hybrid.parquet",      "hmm",         "hybrid_pca-HMM",        hybrid_feats,HYBRID_FEATS),
]

# ── Alternative feature sources (path-A and path-B add-ons) ──────────────────
LEAN_PRICE = ["ret", "vol_21", "vix"]
LEAN_TEXT  = ["css_mean", "sent_net", "n_articles"]
LEAN_FEATS = LEAN_PRICE + LEAN_TEXT

# Lean hybrid pipelines (built in 06_text_hybrid_models.py §9)
# Knob 4: lean_hybrid-GMM dropped from main eval (dur=13d, GMM has no temporal
# structure on small dim).  Underlying parquet stays on disk for ablation use.
lean_feats_df = mkt[LEAN_PRICE].join(news[LEAN_TEXT], how="inner")
for col, name in [("hmm", "lean_hybrid-HMM")]:
    regime_specs.append(
        (f"{DATA_DIR}/regimes_lean_hybrid.parquet", col, name, lean_feats_df, LEAN_FEATS)
    )

# BERTopic-based text-embed (built in 06_text_hybrid_models.py §9)
# Knob 4: text_btopic_smooth-GMM dropped (unstable across re-runs, K=3 mega
# cluster pathology). HMM variant retained.
if os.path.exists(f"{DATA_DIR}/bertopic_daily.parquet"):
    btopic_raw = pd.read_parquet(f"{DATA_DIR}/bertopic_daily.parquet").reindex(trading_days)
    btopic = btopic_raw.rolling(window=ROLL_BTOPIC, min_periods=5).mean()
    for col, name in [("hmm", "text_btopic_smooth-HMM")]:
        regime_specs.append(
            (f"{DATA_DIR}/regimes_text_btopic.parquet", col, name, btopic, btopic.columns.tolist())
        )
else:
    btopic = None

# GDELT supplement (post-2015 sub-window, built in 06_text_hybrid_models.py §10)
# Same-window ablation: silhouette can only be compared WITHIN this trio,
# not against the 2000-onward pipelines above, because the trading-day
# windows differ.
GDELT_REGIMES_PATH = f"{DATA_DIR}/regimes_gdelt_supplement.parquet"
gdelt_feat_path = f"{DATA_DIR}/gdelt_features_daily.parquet"
if os.path.exists(GDELT_REGIMES_PATH) and os.path.exists(gdelt_feat_path):
    gdelt_raw = pd.read_parquet(gdelt_feat_path)
    gdelt_raw.index = pd.to_datetime(gdelt_raw.index)
    GDELT_FEATS = [c for c in gdelt_raw.columns if c.startswith("gdelt_")]
    gdelt_post = gdelt_raw[GDELT_FEATS].reindex(trading_days).rolling(
        window=ROLL_TEXT, min_periods=10
    ).mean()
    for c in GDELT_FEATS:
        if c in gdelt_post.columns:
            gdelt_post[c] = gdelt_post[c].fillna(0.0)

    # Each model in regimes_gdelt_supplement.parquet is evaluated on its
    # OWN feature universe (mirror of how it was fit in 06c).
    price_post = mkt[PRICE_FEATS]
    rp_post    = mkt[PRICE_FEATS].join(news[TEXT_SCALAR_FEATS], how="inner")
    all_post   = mkt[PRICE_FEATS].join(news[TEXT_SCALAR_FEATS], how="inner").join(gdelt_post, how="left")
    for c in GDELT_FEATS:
        if c in all_post.columns:
            all_post[c] = all_post[c].fillna(0.0)

    regime_specs.extend([
        (GDELT_REGIMES_PATH, "price_only",            "price_only_post2015-HMM",            price_post, PRICE_FEATS),
        (GDELT_REGIMES_PATH, "ravenpack_text",        "ravenpack_text_post2015-HMM",        rp_post,    PRICE_FEATS + TEXT_SCALAR_FEATS),
        (GDELT_REGIMES_PATH, "ravenpack_plus_gdelt", "ravenpack_plus_gdelt_post2015-HMM", all_post,   PRICE_FEATS + TEXT_SCALAR_FEATS + GDELT_FEATS),
    ])
else:
    GDELT_FEATS = []

# Macro WRDS-RavenPack ablation (built in 06_text_hybrid_models.py §11).
# Only the days the macro feed covers are scored; same-window constraint applies
# (silhouettes only comparable within the macro sub-window).
MACRO_REGIMES_PATH = f"{DATA_DIR}/regimes_macro_supplement.parquet"
macro_feat_path = f"{DATA_DIR}/news_features_macro_daily.parquet"
if os.path.exists(MACRO_REGIMES_PATH) and os.path.exists(macro_feat_path):
    macro_raw = pd.read_parquet(macro_feat_path).reindex(trading_days)
    MACRO_SCALAR_FEATS = ["css_mean", "css_std", "n_articles", "ess_mean", "sent_net"]
    macro_smooth = macro_raw[MACRO_SCALAR_FEATS].rolling(
        window=ROLL_MACRO, min_periods=10
    ).mean().add_prefix("macro_")
    MACRO_FEATS_PREFIXED = [f"macro_{c}" for c in MACRO_SCALAR_FEATS]

    # Feature universes mirror how each model was fit in 06 §11.
    macro_only_feat   = macro_smooth
    djpr_plus_macro_f = news[TEXT_SCALAR_FEATS].join(macro_smooth, how="inner")
    price_plus_macro_f = mkt[PRICE_FEATS].join(macro_smooth, how="inner")
    hybrid_full_f     = (mkt[PRICE_FEATS]
                         .join(news[TEXT_SCALAR_FEATS], how="inner")
                         .join(macro_smooth, how="inner"))

    regime_specs.extend([
        (MACRO_REGIMES_PATH, "macro_only",       "macro_only-HMM",
         macro_only_feat,   MACRO_FEATS_PREFIXED),
        (MACRO_REGIMES_PATH, "djpr_plus_macro",  "djpr_plus_macro-HMM",
         djpr_plus_macro_f, TEXT_SCALAR_FEATS + MACRO_FEATS_PREFIXED),
        (MACRO_REGIMES_PATH, "price_plus_macro", "price_plus_macro-HMM",
         price_plus_macro_f, PRICE_FEATS + MACRO_FEATS_PREFIXED),
        (MACRO_REGIMES_PATH, "hybrid_full",      "hybrid_full-HMM",
         hybrid_full_f,     PRICE_FEATS + TEXT_SCALAR_FEATS + MACRO_FEATS_PREFIXED),
    ])
else:
    MACRO_FEATS_PREFIXED = []

# Macro TEXT ablation (built in 06_text_hybrid_models.py §12).
# Same-window constraint applies — silhouettes only comparable within the
# macro sub-window (~2000-2024).
MACRO_TEXT_REGIMES_PATH = f"{DATA_DIR}/regimes_macro_text_supplement.parquet"
MACRO_LDA_FEAT_PATH = f"{DATA_DIR}/lda_topics_macro_k15_daily.parquet"
MACRO_BTOPIC_FEAT_PATH = f"{DATA_DIR}/bertopic_macro_daily.parquet"
DJPR_BTOPIC_FEAT_PATH = f"{DATA_DIR}/bertopic_daily.parquet"

if os.path.exists(MACRO_TEXT_REGIMES_PATH):
    _macro_text_cols = pd.read_parquet(MACRO_TEXT_REGIMES_PATH).columns.tolist()
    # IMPORTANT: 06 §12 uses 7-day rolling on topic distributions (see "Tuning
    # history" comment in that section).  We mirror that here so silhouette /
    # ARI are computed on the same X the HMM saw.
    ROLL_MACRO_EVAL = 7
    macro_lda_feat_eval = None
    macro_btopic_feat_eval = None
    djpr_btopic_feat_eval = None
    djpr_lda_feat_eval = None
    MACRO_LDA_COLS_EVAL: list[str] = []
    MACRO_BTOPIC_COLS_EVAL: list[str] = []
    DJPR_BTOPIC_COLS_EVAL: list[str] = []
    DJPR_LDA_COLS_EVAL: list[str] = []

    if os.path.exists(MACRO_LDA_FEAT_PATH):
        macro_lda_feat_eval = pd.read_parquet(MACRO_LDA_FEAT_PATH).reindex(
            trading_days).rolling(
            window=ROLL_MACRO_EVAL, min_periods=3
        ).mean().add_prefix("macro_lda_")
        MACRO_LDA_COLS_EVAL = list(macro_lda_feat_eval.columns)
    if os.path.exists(MACRO_BTOPIC_FEAT_PATH):
        macro_btopic_feat_eval = pd.read_parquet(MACRO_BTOPIC_FEAT_PATH).reindex(
            trading_days).rolling(
            window=ROLL_MACRO_EVAL, min_periods=3
        ).mean().add_prefix("macro_btopic_")
        MACRO_BTOPIC_COLS_EVAL = list(macro_btopic_feat_eval.columns)
    if os.path.exists(DJPR_BTOPIC_FEAT_PATH):
        djpr_btopic_feat_eval = pd.read_parquet(DJPR_BTOPIC_FEAT_PATH).reindex(
            trading_days).rolling(
            window=ROLL_MACRO_EVAL, min_periods=3
        ).mean()
        DJPR_BTOPIC_COLS_EVAL = list(djpr_btopic_feat_eval.columns)
    # DJPR LDA: separate 7d-rolled copy for §12 fusion eval (the 21d-rolled
    # `lda` is for the text_lda_smooth baseline, not §12).
    for _k in [15, 20, 10]:
        _p = f"{DATA_DIR}/lda_topics_k{_k}_daily.parquet"
        if os.path.exists(_p):
            djpr_lda_feat_eval = pd.read_parquet(_p).reindex(
                trading_days).rolling(
                window=ROLL_MACRO_EVAL, min_periods=3
            ).mean()
            DJPR_LDA_COLS_EVAL = list(djpr_lda_feat_eval.columns)
            break

    # Feature universes mirror how each model was fit in 06 §12.
    _candidates = [
        # (col_in_parquet, pipeline_name, feature_df, feature_cols)
        ("macro_lda_only", "macro_lda_only-HMM",
         macro_lda_feat_eval, MACRO_LDA_COLS_EVAL),
        ("macro_btopic_only", "macro_btopic_only-HMM",
         macro_btopic_feat_eval, MACRO_BTOPIC_COLS_EVAL),
        ("djpr_lda_plus_macro_lda", "djpr_lda_plus_macro_lda-HMM",
         (djpr_lda_feat_eval.join(macro_lda_feat_eval, how="inner")
          if (djpr_lda_feat_eval is not None and macro_lda_feat_eval is not None) else None),
         DJPR_LDA_COLS_EVAL + MACRO_LDA_COLS_EVAL),
        ("djpr_btopic_plus_macro_btopic", "djpr_btopic_plus_macro_btopic-HMM",
         (djpr_btopic_feat_eval.join(macro_btopic_feat_eval, how="inner")
          if (djpr_btopic_feat_eval is not None and macro_btopic_feat_eval is not None) else None),
         DJPR_BTOPIC_COLS_EVAL + MACRO_BTOPIC_COLS_EVAL),
        ("text_full_lda", "text_full_lda-HMM",
         ((mkt[PRICE_FEATS].join(djpr_lda_feat_eval, how="inner").join(macro_lda_feat_eval, how="inner"))
          if (djpr_lda_feat_eval is not None and macro_lda_feat_eval is not None) else None),
         PRICE_FEATS + DJPR_LDA_COLS_EVAL + MACRO_LDA_COLS_EVAL),
    ]
    for col, name, feat_df, feat_cols in _candidates:
        if col in _macro_text_cols and feat_df is not None and feat_cols:
            regime_specs.append(
                (MACRO_TEXT_REGIMES_PATH, col, name, feat_df, feat_cols)
            )


# Fine-tuning winners (from finetuning/promote_best.py).
# These pipelines were chosen by the 90-config grid search and need to be
# evaluated on the SAME post-PCA feature space they were clustered on
# (otherwise silhouette differs from the sweep's reported value).
FT_NEWS_FUSION_PATH = f"{DATA_DIR}/regimes_ft_news_fusion_best.parquet"
FT_FULL_STACK_PATH  = f"{DATA_DIR}/regimes_ft_full_stack_best.parquet"

if os.path.exists(FT_NEWS_FUSION_PATH) or os.path.exists(FT_FULL_STACK_PATH):
    # Import from finetuning module to reconstruct exactly the X the sweep used.
    import sys as _sys
    _ft_dir = f"{PROJECT_DIR}/finetuning" if 'PROJECT_DIR' in dir() else (
        "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/finetuning")
    if not os.path.exists(_ft_dir):
        _ft_dir = os.path.join(os.path.dirname(DATA_DIR), "finetuning")
    if _ft_dir not in _sys.path:
        _sys.path.insert(0, _ft_dir)
    try:
        from _common import (load_djpr_scalar as _ft_djpr,
                              load_macro_scalar as _ft_macro,
                              load_price as _ft_price,
                              load_text_repr as _ft_text,
                              reduce_dim_post as _ft_reduce)
        for _name, _path, _include_price in [
            ("ft_news_fusion_best", FT_NEWS_FUSION_PATH, False),
            ("ft_full_stack_best",  FT_FULL_STACK_PATH,  True),
        ]:
            if not os.path.exists(_path):
                continue
            _parts = [_ft_djpr(), _ft_macro(),
                      _ft_text("BERTopic", "djpr"),
                      _ft_text("BERTopic", "macro")]
            if _include_price:
                _parts = [_ft_price()] + _parts
            _feat = _parts[0]
            for _p in _parts[1:]:
                _feat = _feat.join(_p, how="inner")
            _feat = _feat.dropna()
            # Apply PCA-95% (matches finetuning sweep dim_red="pca95")
            _X = StandardScaler().fit_transform(_feat.values)
            _X_red, _ = _ft_reduce(_X, "pca95")
            _feat_pca = pd.DataFrame(
                _X_red, index=_feat.index,
                columns=[f"pc{i}" for i in range(_X_red.shape[1])],
            )
            regime_specs.append(
                (_path, "regime", _name, _feat_pca, list(_feat_pca.columns))
            )
            print(f"[ft pickup] {_name}: dim_in={_feat.shape[1]} → PCA={_X_red.shape[1]}")
    except Exception as _exc:
        print(f"[ft pickup] skipped: {type(_exc).__name__}: {_exc}")


def load_regimes(filepath, col):
    import os
    if not os.path.exists(filepath):
        return None
    df = pd.read_parquet(filepath)
    if col in df.columns:
        return df[[col]].rename(columns={col: "regime"})
    return None

print("Feature sources and regime files loaded.")
print(f"Total pipelines in regime_specs: {len(regime_specs)}")


# %% [markdown]
# ## 2. Clustering Quality (Silhouette + Davies-Bouldin)


# %%
def clustering_quality(labels, X_scaled):
    """Return (silhouette, dbi). Auto PCA-compress high-dim inputs."""
    if len(set(labels)) < 2:
        return np.nan, np.nan
    if X_scaled.shape[1] > 50:
        X_scaled = PCA(n_components=20, random_state=42).fit_transform(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    dbi = davies_bouldin_score(X_scaled, labels)
    return sil, dbi


quality_rows = []

for filepath, col, name, feat_df, feat_cols in regime_specs:
    regimes = load_regimes(filepath, col)
    if regimes is None:
        continue
    if feat_df is None or feat_cols is None:
        quality_rows.append({"pipeline": name, "silhouette": np.nan, "dbi": np.nan})
        continue
    aligned = feat_df[feat_cols].join(regimes, how="inner").dropna()
    labels_arr = aligned["regime"].values.astype(int)
    sig = _pipeline_signature(labels_arr, feat_cols)
    bucket = _bucket(_eval_cache, name, sig)
    if "silhouette" in bucket and "dbi" in bucket:
        sil, dbi = bucket["silhouette"], bucket["dbi"]
        print(f"  {name:35s}  sil={sil:.4f}  DBI={dbi:.4f}  [cached]")
    else:
        X = StandardScaler().fit_transform(aligned[feat_cols].values)
        sil, dbi = clustering_quality(labels_arr, X)
        bucket["silhouette"] = float(sil)
        bucket["dbi"] = float(dbi)
        print(f"  {name:35s}  sil={sil:.4f}  DBI={dbi:.4f}")
    quality_rows.append({"pipeline": name, "silhouette": round(sil, 4), "dbi": round(dbi, 4)})

df_quality = pd.DataFrame(quality_rows).set_index("pipeline")
df_quality


# %% [markdown]
# ## 3. Regime Stability


# %%
def mean_duration(labels: np.ndarray) -> float:
    """Average number of consecutive days in the same regime."""
    durations = []
    count = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            count += 1
        else:
            durations.append(count)
            count = 1
    durations.append(count)
    return float(np.mean(durations))


stability_rows = []

for filepath, col, name, _, _ in regime_specs:
    regimes = load_regimes(filepath, col)
    if regimes is None:
        continue
    labels = regimes['regime'].values.astype(int)
    md = mean_duration(labels)
    stability_rows.append({'pipeline': name, 'mean_duration_days': round(md, 1)})
    print(f"  {name:30s}  mean duration={md:.1f} days")

df_stability = pd.DataFrame(stability_rows).set_index('pipeline')
df_stability


# %% [markdown]
# ### 3b. Rolling-Window ARI (Stability Under Re-fitting)


# %%
def rolling_ari(X, full_labels, window=1250, step=63):
    k = len(set(full_labels))
    aris = []
    for start in range(0, len(X) - window, step):
        X_win = X[start : start + window]
        labels_full_win = full_labels[start : start + window]
        try:
            gmm = GaussianMixture(n_components=k, n_init=3, random_state=42)
            labels_win = gmm.fit_predict(X_win)
            aris.append(adjusted_rand_score(labels_full_win, labels_win))
        except Exception:
            pass
    return float(np.mean(aris)) if aris else np.nan


ari_rows = []
for filepath, col, name, feat_df, feat_cols in regime_specs:
    regimes = load_regimes(filepath, col)
    if regimes is None or feat_df is None:
        ari_rows.append({"pipeline": name, "rolling_ari": np.nan})
        continue
    aligned = feat_df[feat_cols].join(regimes, how="inner").dropna()
    labels_arr = aligned["regime"].values.astype(int)
    sig = _pipeline_signature(labels_arr, feat_cols)
    bucket = _bucket(_eval_cache, name, sig)
    if "rolling_ari" in bucket:
        ari = bucket["rolling_ari"]
        print(f"  {name:35s}  rolling ARI={ari:.4f}  [cached]")
    else:
        X = StandardScaler().fit_transform(aligned[feat_cols].values)
        ari = rolling_ari(X, labels_arr)
        bucket["rolling_ari"] = float(ari) if not np.isnan(ari) else None
        print(f"  {name:35s}  rolling ARI={ari:.4f}")
    ari_rows.append({"pipeline": name, "rolling_ari": round(ari, 4) if ari is not None and not (isinstance(ari, float) and np.isnan(ari)) else np.nan})

df_ari = pd.DataFrame(ari_rows).set_index("pipeline")
df_ari


# %% [markdown]
# ## 4. Financial Interpretability


# %%
def financial_stats(regimes_df, mkt_df, min_days=MIN_REGIME_DAYS):
    """Per-regime annualized return, vol, Sharpe, max drawdown.
    Regimes with fewer than min_days are flagged as micro-clusters.
    """
    joined = mkt_df[["ret"]].join(regimes_df[["regime"]], how="inner").dropna()
    rows = []
    for r, grp in joined.groupby("regime"):
        ann_ret  = grp["ret"].mean() * 252
        ann_vol  = grp["ret"].std()  * np.sqrt(252)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum      = np.exp(grp["ret"].cumsum())
        drawdown = ((cum - cum.cummax()) / cum.cummax()).min()
        n_days   = len(grp)
        is_micro = n_days < min_days
        rows.append({
            "regime":   r,
            "n_days":   n_days,
            "micro":    is_micro,
            "ann_ret":  round(ann_ret, 4),
            "ann_vol":  round(ann_vol, 4),
            "sharpe":   round(sharpe, 3) if not is_micro else np.nan,
            "max_dd":   round(drawdown, 4),
        })
    return pd.DataFrame(rows)


fin_rows = []
for filepath, col, name, feat_df, feat_cols in regime_specs:
    regimes = load_regimes(filepath, col)
    if regimes is None:
        continue
    # Use same sig as quality/ari blocks so cache stays consistent per pipeline.
    if feat_df is not None and feat_cols is not None:
        aligned = feat_df[feat_cols].join(regimes, how="inner").dropna()
        sig = _pipeline_signature(aligned["regime"].values.astype(int), feat_cols)
    else:
        sig = _pipeline_signature(regimes["regime"].values.astype(int), [name])
    bucket = _bucket(_eval_cache, name, sig)
    if "financial_stats" in bucket:
        stats = pd.DataFrame(bucket["financial_stats"])
        cached_tag = "  [cached]"
    else:
        stats = financial_stats(regimes, mkt)
        bucket["financial_stats"] = stats.to_dict(orient="records")
        cached_tag = ""
    stats["pipeline"] = name
    fin_rows.append(stats)
    micro_flag = " [MICRO CLUSTERS PRESENT]" if stats["micro"].any() else ""
    print(f"\n{name}{cached_tag}{micro_flag}:")
    display(stats)

df_fin = pd.concat(fin_rows, ignore_index=True)


# %% [markdown]
# ## 5. Master Summary Table


# %%
# best-regime Sharpe excludes micro-clusters (n_days < MIN_REGIME_DAYS)
fin_valid = df_fin[~df_fin["micro"]]
fin_best  = fin_valid.groupby("pipeline")["sharpe"].max().rename("best_regime_sharpe")

summary = df_quality.join(df_stability).join(df_ari).join(fin_best)

# ── Gap 4: join statistical-fit metrics from 06 ──────────────────────────────
_STATS_PATH = f"{DATA_DIR}/pipeline_stats.json"
if os.path.exists(_STATS_PATH):
    with open(_STATS_PATH) as _f:
        _stats_dict = json.load(_f)
    df_stats = pd.DataFrame.from_dict(_stats_dict, orient="index")
    summary = summary.join(df_stats, how="left")
    print(f"[stats] joined {df_stats.shape[0]} pipeline entries from {_STATS_PATH}")
else:
    print(f"[stats] {_STATS_PATH} not found; statistical fit columns omitted")

summary = summary.sort_values("silhouette", ascending=False)

print("\n=== Master Evaluation Summary (v3 — with statistical fit) ===")
display(summary)

summary.to_parquet(f"{DATA_DIR}/eval_summary.parquet")
print("Saved: eval_summary.parquet")


# %% [markdown]
# ## 6. Out-of-Sample Stability Check


# %%
SPLIT = "2013-01-01"


def oos_check(filepath, col, name, feat_df, feat_cols):
    """Fit GMM on train, predict on test, compare financial stats."""
    regimes = load_regimes(filepath, col)
    if regimes is None or feat_df is None:
        return None
    aligned = feat_df[feat_cols].join(regimes, how="inner").dropna()
    dates = aligned.index
    train_mask = dates < SPLIT
    test_mask  = ~train_mask
    if train_mask.sum() < 100 or test_mask.sum() < 100:
        return None

    # Cache uses same per-pipeline bucket sig as quality/ari/financial blocks.
    labels_arr = aligned["regime"].values.astype(int)
    sig = _pipeline_signature(labels_arr, feat_cols)
    bucket = _bucket(_eval_cache, name, sig)
    if "oos_sharpe" in bucket and "is_sharpe" in bucket:
        is_s = pd.Series({int(k): v for k, v in bucket["is_sharpe"].items()},
                          name="IS")
        oos_s = pd.Series({int(k): v for k, v in bucket["oos_sharpe"].items()},
                          name="OOS")
        print(f"\n{name}: in-sample vs out-of-sample Sharpe  [cached]")
        display(pd.concat([is_s, oos_s], axis=1).round(3))
        return None

    X = StandardScaler().fit_transform(aligned[feat_cols].values)
    k = int(aligned["regime"].max()) + 1
    try:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=42)
        gmm.fit(X[train_mask])
        labels_oos = gmm.predict(X[test_mask])
    except Exception:
        return None
    oos_df = pd.DataFrame({"regime": labels_oos}, index=dates[test_mask])
    stats_is  = financial_stats(regimes.loc[dates[train_mask]], mkt)
    stats_oos = financial_stats(oos_df, mkt)
    is_s  = stats_is.set_index("regime")["sharpe"].rename("IS")
    oos_s = stats_oos.set_index("regime")["sharpe"].rename("OOS")
    bucket["is_sharpe"] = {int(k_): (None if pd.isna(v) else float(v))
                            for k_, v in is_s.items()}
    bucket["oos_sharpe"] = {int(k_): (None if pd.isna(v) else float(v))
                             for k_, v in oos_s.items()}
    print(f"\n{name}: in-sample vs out-of-sample Sharpe")
    display(pd.concat([is_s, oos_s], axis=1).round(3))
    return stats_oos


# Run OOS check for key pipelines only (expensive)
oos_pipelines = [
    (f"{DATA_DIR}/regimes_price.parquet",       "gmm_k3",  "price-GMM",            mkt,    PRICE_FEATS),
    (f"{DATA_DIR}/regimes_price.parquet",       "hmm_k5_full","price-HMM",          mkt,    PRICE_FEATS),
    (f"{DATA_DIR}/regimes_text_scalar.parquet", "hmm",     "text_scalar_smooth-HMM",news,   TEXT_SCALAR_FEATS),
    (f"{DATA_DIR}/regimes_text_lda.parquet",    "hmm",     "text_lda_smooth-HMM",   lda,    lda.columns.tolist()),
    (f"{DATA_DIR}/regimes_hybrid.parquet",      "hmm",     "hybrid_pca-HMM",        hybrid_feats, HYBRID_FEATS),
]
for args in oos_pipelines:
    oos_check(*args)

# Persist the cache so the next run can short-circuit unchanged pipelines.
_save_eval_cache(_eval_cache)
print(f"\n[07-cache] saved {len(_eval_cache)} pipeline entries to {EVAL_CACHE_PATH}")


# %% [markdown]
# ## 6. Visualisation — Regime Comparison Heatmap


# %%
fig, ax = plt.subplots(figsize=(10, 5))

metrics = ['silhouette', 'mean_duration_days', 'rolling_ari', 'best_regime_sharpe']
plot_data = summary[metrics].dropna(how='all')

# Normalise each column 0–1 for heatmap
normed = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min() + 1e-9)

im = ax.imshow(normed.values, aspect='auto', cmap='RdYlGn')
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data.index, fontsize=9)
plt.colorbar(im, ax=ax, label='Normalised score (higher = better)')
ax.set_title('Pipeline comparison heatmap')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_images/eval_heatmap.png', dpi=120)
plt.close("all")
