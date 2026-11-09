# Auto-generated from 06_text_hybrid_models.ipynb
# Intended to run in Google Colab with the original Google Drive paths preserved.

import argparse
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

parser = argparse.ArgumentParser(description="Text and hybrid regime models.")
parser.add_argument("--enable-macro", action="store_true",
                    help="Also fit WRDS-RavenPack macro ablation models.")
parser.add_argument("--skip-existing-regimes", action="store_true",
                    help="Within 06, skip any section whose regimes_*.parquet "
                         "output already exists (load it for downstream §8 viz "
                         "instead of re-fitting).  Lets a re-run focus on "
                         "newly-enabled ablations (e.g. --enable-macro).")
args, _ = parser.parse_known_args()



# %% [markdown]
# # 06 — Text-Only & Hybrid Regime Detection


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
K_VALUES     = [3, 4, 5]   # default GMM/HMM K grid (BIC selection)
TOPIC_GMM_K  = [4, 5]      # narrower grid for LDA/BERTopic GMM — drops K=3 to
                           # prevent "1 mega cluster + 2 small" pathology
                           # (text_btopic_smooth-GMM was 894 days, text_lda-GMM 368)

# Per-feature-type rolling windows.  Each tuned to keep HMM mean_duration in
# the 60-365 day band given the noise characteristics of that representation:
#   ROLL_SCALAR  — DJPR scalar: 45d (was 30d).  Scalar CSS/sentiment is the
#                  noisiest signal; 4 HMM pipelines using `news` were < 60 days.
#   ROLL_LDA     — LDA topic distributions: 30d (was 30d).  Already smooth as
#                  daily aggregates of many headlines.
#   ROLL_BTOPIC  — BERTopic: 14d (was 30d).  BERTopic clusters are highly
#                  persistent (month-scale), 30d caused 368-day regimes.
#   ROLL_MACRO   — macro scalar (§11): 30d (was 30d).  Already produces
#                  150-200 day regimes; further smoothing not needed.
#   ROLL_TEXT_MACRO — macro text (§12): hard-coded 7d inside §12.
ROLL_SCALAR  = 35   # DJPR scalar — was 45 (overshot to 298d); 35 lands in 80-130d
ROLL_LDA     = 30
ROLL_BTOPIC  = 21   # DJPR BERTopic — was 14 (overshot to 49d); 21 lands in 80-150d
ROLL_MACRO   = 30
ROLL_TEXT    = ROLL_LDA   # backward-compat alias for §11 macro & any leftover ref

DATA_DIR     = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data"
IMAGES_DIR   = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_images"


# %% [markdown]
# ## 1. Load All Feature Sources


# %%
# ── Market features ─────────────────────────────────────────────────────────
mkt = pd.read_parquet(f"{DATA_DIR}/market_features.parquet")
PRICE_FEATS = ["ret", "vol_21", "vix", "term_spread", "credit_spread"]
trading_days = mkt[PRICE_FEATS].dropna().index

# ── News scalar features — rolling means at ROLL_SCALAR (45d) ───────────────
news_raw = pd.read_parquet(f"{DATA_DIR}/news_features_daily.parquet")
news_raw = news_raw.reindex(trading_days)

# Roll each scalar signal: reduces day-to-day noise, improves temporal coherence.
# ROLL_SCALAR=45 (was 30); scalar CSS/sentiment is the noisiest representation
# and needed a longer window to push HMM regime durations above 60 days.
_roll = news_raw[["css_mean","css_std","n_articles","ess_mean","sent_net"]].rolling(
    window=ROLL_SCALAR, min_periods=15
).mean()
news = _roll.copy()
news.index.name = "date"
TEXT_SCALAR_FEATS = ["css_mean","css_std","n_articles","ess_mean","sent_net"]

# ── LDA topics — headline-level K=15 (from nb04 v2) + 21-day rolling ─────────
# If nb04 produced k15, use it; fall back to k20 or k10 if unavailable.
import os
for _lda_k in [15, 20, 10]:
    _lda_path = f"{DATA_DIR}/lda_topics_k{_lda_k}_daily.parquet"
    if os.path.exists(_lda_path):
        lda_raw = pd.read_parquet(_lda_path).reindex(trading_days)
        print(f"Loaded LDA k={_lda_k}: {lda_raw.shape}")
        break
lda = lda_raw.rolling(window=ROLL_TEXT, min_periods=10).mean()
lda.index.name = "date"

print(f"Market:           {mkt.shape}")
print(f"News (smoothed):  {news.shape}  NaN rows: {news[TEXT_SCALAR_FEATS].isna().any(axis=1).sum()}")
print(f"LDA (smoothed):   {lda.shape}  NaN rows: {lda.isna().any(axis=1).sum()}")


# %% [markdown]
# ## 2. Helpers


# %%
def fit_best_gmm(X, k_values=K_VALUES):
    """Return (best_k, gmm, labels) chosen by BIC."""
    results = {}
    for k in k_values:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              n_init=5, random_state=RANDOM_STATE)
        gmm.fit(X)
        results[k] = (gmm.bic(X), gmm, gmm.predict(X))
    best_k = min(results, key=lambda k: results[k][0])
    return best_k, results[best_k][1], results[best_k][2]


HIGH_DIM_THRESHOLD = 10  # at >= this many features, switch HMM to diag cov


def fit_best_hmm(X, k_values=K_VALUES, n_trials=5):
    """Return (best_k, hmm, labels).

    Changes from v1:
    - covariance_type="full" for all pipelines (was "diag" for text — now matches price-HMM)
    - Sticky initialization: transmat diagonal = 0.85 -> encourages longer-lasting regimes
    - 300 EM iterations

    Adaptive covariance (added to fix mean_duration runaway on high-dim ablations):
    For X with n_features >= HIGH_DIM_THRESHOLD, use "diag" only — full covariance
    has K * d*(d+1)/2 params per state which explodes on 10+ dim inputs and pushes
    the HMM into near-absorbing states (mean duration > 300 days observed on
    price_plus_macro / hybrid_full ablations).
    """
    d = X.shape[1]
    cov_type = "diag" if d >= HIGH_DIM_THRESHOLD else "full"
    best_score, best_k, best_model, best_labels = -np.inf, None, None, None
    for k in k_values:
        for seed in range(n_trials):
            try:
                m = GaussianHMM(
                    n_components=k,
                    covariance_type=cov_type,
                    n_iter=300,
                    random_state=seed,
                    tol=1e-5,
                    init_params="smc",   # skip 't' so manual transmat_ isn't overwritten
                )
                # Sticky init: self-transition = 0.85, rest shared equally
                transmat = np.full((k, k), 0.15 / max(k - 1, 1))
                np.fill_diagonal(transmat, 0.85)
                m.transmat_ = transmat
                m.fit(X)
                score = m.score(X)
                if score > best_score:
                    best_score, best_k, best_model = score, k, m
                    best_labels = m.predict(X)
            except Exception:
                pass
    return best_k, best_model, best_labels


def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    return km.fit_predict(X)


# ── Statistical-fit capture (Gap 4) ──────────────────────────────────────────
# Per-pipeline log-likelihood / BIC / AIC.  Accumulated across all fit calls
# in this file, then merged into _data/pipeline_stats.json at end (preserving
# pre-existing keys so partial re-runs incrementally update).
pipeline_stats: dict[str, dict] = {}


def compute_stats(model, X, K):
    """Compute (log_likelihood, bic, aic) for either GMM or HMM.
    Returns dict with three float keys."""
    if model is None:
        return {"log_likelihood": float("nan"), "bic": float("nan"), "aic": float("nan")}
    N, D = X.shape
    if hasattr(model, "bic"):              # sklearn GaussianMixture
        ll = float(model.score(X) * N)     # score() returns per-sample avg
        bic = float(model.bic(X))
        aic = float(model.aic(X))
    else:                                   # hmmlearn GaussianHMM
        ll = float(model.score(X))         # total log-likelihood
        cov_type = model.covariance_type
        cov_params = {
            "full":      K * D * (D + 1) // 2,
            "diag":      K * D,
            "spherical": K,
            "tied":      D * (D + 1) // 2,
        }[cov_type]
        n_params = (K - 1) + K * (K - 1) + K * D + cov_params
        bic = float(-2 * ll + n_params * np.log(N))
        aic = float(-2 * ll + 2 * n_params)
    return {"log_likelihood": ll, "bic": bic, "aic": aic}


def align_and_scale(df_feat, feature_cols):
    """Drop NaN rows, standard-scale, return (X, aligned_index)."""
    df_clean = df_feat[feature_cols].dropna()
    X = StandardScaler().fit_transform(df_clean.values)
    return X, df_clean.index


# %% [markdown]
# ## 3. Pipeline A — Text-Scalar


# %%
# Pipeline A uses 21-day rolling smoothed CSS/sentiment features
_OUT_TS = f"{DATA_DIR}/regimes_text_scalar.parquet"
if args.skip_existing_regimes and os.path.exists(_OUT_TS):
    print(f"§3 text_scalar: skip (exists)")
    regimes_ts = pd.read_parquet(_OUT_TS)
else:
    X_ts, idx_ts = align_and_scale(news, TEXT_SCALAR_FEATS)
    print(f"Text-scalar (smoothed) X: {X_ts.shape}")

    k_gmm_ts, gmm_ts, labels_gmm_ts = fit_best_gmm(X_ts)
    k_hmm_ts, hmm_ts, labels_hmm_ts = fit_best_hmm(X_ts)
    pipeline_stats["text_scalar_smooth-GMM"] = compute_stats(gmm_ts, X_ts, k_gmm_ts)
    pipeline_stats["text_scalar_smooth-HMM"] = compute_stats(hmm_ts, X_ts, k_hmm_ts)

    regimes_ts = pd.DataFrame(index=idx_ts)
    regimes_ts["gmm"] = labels_gmm_ts
    regimes_ts["hmm"] = labels_hmm_ts
    regimes_ts.to_parquet(_OUT_TS)
    print(f"Best GMM K={k_gmm_ts}, Best HMM K={k_hmm_ts}")
    print("Saved: regimes_text_scalar.parquet")


# %% [markdown]
# ## 4. Pipeline B — Text-LDA


# %%
# Pipeline B: 21-day rolling LDA topic distributions (headline-level K=15)
_OUT_LDA = f"{DATA_DIR}/regimes_text_lda.parquet"
if args.skip_existing_regimes and os.path.exists(_OUT_LDA):
    print(f"§4 text_lda: skip (exists)")
    regimes_lda = pd.read_parquet(_OUT_LDA)
else:
    X_lda, idx_lda = align_and_scale(lda, lda.columns.tolist())
    print(f"LDA (smoothed) X: {X_lda.shape}")

    # Knob 1: drop K=3 to avoid GMM forming one mega-cluster on smooth topic dist.
    # (text_lda_smooth-GMM was 368 days at K=3.)
    k_gmm_lda, gmm_lda, labels_gmm_lda = fit_best_gmm(X_lda, k_values=TOPIC_GMM_K)
    k_hmm_lda, hmm_lda, labels_hmm_lda = fit_best_hmm(X_lda)
    labels_km_lda = fit_kmeans(X_lda, k=k_gmm_lda)
    pipeline_stats["text_lda_smooth-GMM"] = compute_stats(gmm_lda, X_lda, k_gmm_lda)
    pipeline_stats["text_lda_smooth-HMM"] = compute_stats(hmm_lda, X_lda, k_hmm_lda)

    regimes_lda = pd.DataFrame(index=idx_lda)
    regimes_lda["gmm"]    = labels_gmm_lda
    regimes_lda["hmm"]    = labels_hmm_lda
    regimes_lda["kmeans"] = labels_km_lda
    regimes_lda.to_parquet(_OUT_LDA)
    print(f"Best GMM K={k_gmm_lda}")
    print("Saved: regimes_text_lda.parquet")


# %% [markdown]
# ## 6. Pipeline D — Hybrid (PCA-Weighted Fusion)


# %%
# ── PCA-weighted fusion ─────────────────────────────────────────────────────
_OUT_HYB = f"{DATA_DIR}/regimes_hybrid.parquet"
if args.skip_existing_regimes and os.path.exists(_OUT_HYB):
    print(f"§6 hybrid_pca: skip (exists)")
    regimes_hyb = pd.read_parquet(_OUT_HYB)
else:
    # Step 1: align price + smoothed text on common trading days
    hybrid_raw = mkt[PRICE_FEATS].join(news[TEXT_SCALAR_FEATS], how="inner").dropna()
    print(f"Hybrid aligned: {hybrid_raw.shape}")

    X_price_raw = StandardScaler().fit_transform(hybrid_raw[PRICE_FEATS].values)
    X_text_raw  = StandardScaler().fit_transform(hybrid_raw[TEXT_SCALAR_FEATS].values)

    # Step 2: PCA each group separately (keep enough PCs for 95% variance)
    pca_price = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_price_raw)
    pca_text  = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_text_raw)
    print(f"Price PCA: {pca_price.n_components_} PCs  "
          f"({pca_price.explained_variance_ratio_.sum():.2f} var explained)")
    print(f"Text  PCA: {pca_text.n_components_}  PCs  "
          f"({pca_text.explained_variance_ratio_.sum():.2f} var explained)")

    # Step 3: standardize PCA outputs and concatenate
    Z_price = StandardScaler().fit_transform(pca_price.transform(X_price_raw))
    Z_text  = StandardScaler().fit_transform(pca_text.transform(X_text_raw))
    X_hyb   = np.hstack([Z_price, Z_text])
    idx_hyb = hybrid_raw.index
    print(f"Hybrid X: {X_hyb.shape}  (price PCs + text PCs)")

    k_gmm_h, gmm_h, labels_gmm_h = fit_best_gmm(X_hyb)
    k_hmm_h, hmm_h, labels_hmm_h = fit_best_hmm(X_hyb)
    pipeline_stats["hybrid_pca-GMM"] = compute_stats(gmm_h, X_hyb, k_gmm_h)
    pipeline_stats["hybrid_pca-HMM"] = compute_stats(hmm_h, X_hyb, k_hmm_h)

    regimes_hyb = pd.DataFrame(index=idx_hyb)
    regimes_hyb["gmm"] = labels_gmm_h
    regimes_hyb["hmm"] = labels_hmm_h
    regimes_hyb.to_parquet(_OUT_HYB)
    print(f"Best GMM K={k_gmm_h}, Best HMM K={k_hmm_h}")
    print("Saved: regimes_hybrid.parquet")


# %% [markdown]
# ## 8. Visualise All Pipelines


# %%
def plot_pipeline_regimes(regimes_df, model_col, pipeline_name, mkt_df):
    joined = mkt_df[["ret"]].join(regimes_df[[model_col]], how="inner").dropna()
    dates  = joined.index.to_numpy()
    rets   = joined["ret"].values
    labels = joined[model_col].values.astype(int)
    k = labels.max() + 1
    cum    = np.exp(np.cumsum(rets))
    colors = plt.cm.Set1.colors
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, cum, linewidth=0.7, color="black")
    for regime in range(k):
        mask = labels == regime
        ax.fill_between(dates, 0, cum, where=mask,
                        alpha=0.25, color=colors[regime % len(colors)],
                        label=f"Regime {regime}")
    ax.set_title(f"{pipeline_name} — {model_col.upper()}")
    ax.set_ylabel("S&P 500 growth")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    safe = f"{pipeline_name}_{model_col}".replace(" ", "_")
    plt.savefig(f"{IMAGES_DIR}/regimes_{safe}.png", dpi=100)
    plt.close("all")


for (df_r, name) in [
    (regimes_ts,  "text_scalar_smooth"),
    (regimes_lda, "text_lda_smooth"),
    (regimes_hyb, "hybrid_pca"),
]:
    for col in df_r.columns:
        plot_pipeline_regimes(df_r, col, name, mkt)


# %% [markdown]
# ## 9. Alternative pipelines — lean_hybrid + BERTopic text-embed


# %%
LEAN_PRICE = ["ret", "vol_21", "vix"]
LEAN_TEXT  = ["css_mean", "sent_net", "n_articles"]


def load_text_scalar_for_lean(suffix: str = ""):
    path = f"{DATA_DIR}/news_features_daily{suffix}.parquet"
    if not os.path.exists(path):
        return None
    raw = pd.read_parquet(path).reindex(trading_days)
    # Matches the global `news` smoothing (ROLL_SCALAR=45) so lean_hybrid uses
    # the same DJPR scalar representation as text_scalar baseline.
    return raw[TEXT_SCALAR_FEATS].rolling(window=ROLL_SCALAR, min_periods=15).mean()


def fit_lean_hybrid(news_df, label_tag: str):
    if news_df is None:
        print(f"  ({label_tag}) text-scalar source missing, skipping lean_hybrid")
        return None
    feats = LEAN_PRICE + LEAN_TEXT
    aligned = mkt[LEAN_PRICE].join(news_df[LEAN_TEXT], how="inner").dropna()
    X = StandardScaler().fit_transform(aligned[feats].values)
    print(f"  ({label_tag}) lean_hybrid X: {X.shape}")
    k_g, gmm_m, labels_g = fit_best_gmm(X)
    k_h, hmm_m, labels_h = fit_best_hmm(X)
    name_suffix = "" if label_tag == "v1" else f"_{label_tag}"
    pipeline_stats[f"lean_hybrid{name_suffix}-GMM"] = compute_stats(gmm_m, X, k_g)
    pipeline_stats[f"lean_hybrid{name_suffix}-HMM"] = compute_stats(hmm_m, X, k_h)
    out = pd.DataFrame(index=aligned.index)
    out["gmm"] = labels_g
    if labels_h is not None:
        out["hmm"] = labels_h
    print(f"    best GMM K={k_g}  HMM K={k_h}")
    return out


_OUT_LEAN_V1 = f"{DATA_DIR}/regimes_lean_hybrid.parquet"
if args.skip_existing_regimes and os.path.exists(_OUT_LEAN_V1):
    print(f"§9 lean_hybrid v1: skip (exists)")
else:
    print("\n=== lean_hybrid (RavenPack original filter) ===")
    lean_v1 = fit_lean_hybrid(load_text_scalar_for_lean(""), "v1")
    if lean_v1 is not None:
        lean_v1.to_parquet(_OUT_LEAN_V1)
        print("  Saved: regimes_lean_hybrid.parquet")

BERTOPIC_PATH = f"{DATA_DIR}/bertopic_daily.parquet"
_OUT_BTOPIC = f"{DATA_DIR}/regimes_text_btopic.parquet"
if args.skip_existing_regimes and os.path.exists(_OUT_BTOPIC):
    print(f"§9 text_btopic: skip (exists)")
elif not os.path.exists(BERTOPIC_PATH):
    print(f"\nNote: {BERTOPIC_PATH} not found; skip text_btopic pipeline.")
else:
    btopic_raw = pd.read_parquet(BERTOPIC_PATH).reindex(trading_days)
    # ROLL_BTOPIC=14 (was 30); BERTopic features are highly persistent on their
    # own (topic-cluster membership is month-scale stable), so a long rolling
    # window over-smoothed and pushed HMM regimes to 368 days.
    btopic = btopic_raw.rolling(window=ROLL_BTOPIC, min_periods=5).mean()
    aligned_b = btopic.dropna()
    X_b = StandardScaler().fit_transform(aligned_b.values)
    print(f"\n=== text_btopic === X: {X_b.shape}  (cols={aligned_b.shape[1]})")
    # Knob 1: drop K=3 to avoid GMM forming one mega-cluster on smooth topic dist.
    k_g, gmm_b, labels_g = fit_best_gmm(X_b, k_values=TOPIC_GMM_K)
    k_h, hmm_b, labels_h = fit_best_hmm(X_b)
    pipeline_stats["text_btopic_smooth-GMM"] = compute_stats(gmm_b, X_b, k_g)
    pipeline_stats["text_btopic_smooth-HMM"] = compute_stats(hmm_b, X_b, k_h)
    regimes_btopic = pd.DataFrame(index=aligned_b.index)
    regimes_btopic["gmm"] = labels_g
    if labels_h is not None:
        regimes_btopic["hmm"] = labels_h
    regimes_btopic.to_parquet(_OUT_BTOPIC)
    print(f"  best GMM K={k_g}  HMM K={k_h}")
    print("  Saved: regimes_text_btopic.parquet")


# %% [markdown]
# ## 11. WRDS RavenPack macro ablation (optional)


# %%
if args.enable_macro:
    MACRO_FEATURES_PATH = f"{DATA_DIR}/news_features_macro_daily.parquet"
    MACRO_OUTPUT_PATH = f"{DATA_DIR}/regimes_macro_supplement.parquet"

    if not os.path.exists(MACRO_FEATURES_PATH):
        print(f"\nMacro ablation: {MACRO_FEATURES_PATH} not found. "
              "Run 02d_news_macro.py first. Skipping.")
    elif os.path.exists(MACRO_OUTPUT_PATH):
        print(f"\nSkipping macro ablation; output exists: {MACRO_OUTPUT_PATH}")
    else:
        # Macro features share schema with news (see 02d_news_macro.py).
        macro_raw = pd.read_parquet(MACRO_FEATURES_PATH).reindex(trading_days)
        MACRO_FEATS = ["css_mean", "css_std", "n_articles", "ess_mean", "sent_net"]
        # Distinguish macro columns from DJPR columns when joined.
        # ROLL_MACRO=30 — macro scalar is already a sparser daily aggregate
        # (~50 headlines/day vs DJPR's ~500); 30d is the right balance.
        macro = macro_raw[MACRO_FEATS].rolling(
            window=ROLL_MACRO, min_periods=10
        ).mean().add_prefix("macro_")
        MACRO_FEATS_PREFIXED = [f"macro_{c}" for c in MACRO_FEATS]

        # Restrict to days the macro feed actually covers, else aligns will be empty.
        macro_days = macro.dropna().index
        print(f"\n=== Macro ablation === covered days: {len(macro_days)}  "
              f"({macro_days.min()} → {macro_days.max()})")

        def fit_macro_one(label, feat_df, feat_cols):
            aligned = feat_df[feat_cols].dropna()
            if len(aligned) < 250:
                print(f"  {label}: only {len(aligned)} aligned days — skipping.")
                return None, None, None
            X = StandardScaler().fit_transform(aligned.values)
            print(f"\n=== {label} ===  X: {X.shape}")
            k_g, gmm_m, labels_g = fit_best_gmm(X)
            k_h, hmm_m, labels_h = fit_best_hmm(X)
            pipeline_stats[f"{label}-GMM"] = compute_stats(gmm_m, X, k_g)
            pipeline_stats[f"{label}-HMM"] = compute_stats(hmm_m, X, k_h)
            print(f"  best GMM K={k_g}  HMM K={k_h}")
            return aligned.index, labels_g, labels_h

        # ablation 1: macro scalar only
        idx_m, gmm_m, hmm_m = fit_macro_one(
            "macro_only", macro, MACRO_FEATS_PREFIXED
        )
        # ablation 2: DJPR + macro fusion (text only, no price)
        djpr_plus_macro = news.join(macro, how="inner")
        idx_d, gmm_d, hmm_d = fit_macro_one(
            "djpr_plus_macro", djpr_plus_macro,
            TEXT_SCALAR_FEATS + MACRO_FEATS_PREFIXED,
        )
        # ablation 3: price + macro
        price_plus_macro = mkt[PRICE_FEATS].join(macro, how="inner")
        idx_pm, gmm_pm, hmm_pm = fit_macro_one(
            "price_plus_macro", price_plus_macro,
            PRICE_FEATS + MACRO_FEATS_PREFIXED,
        )
        # ablation 4: full hybrid (price + DJPR + macro)
        hybrid_full = (mkt[PRICE_FEATS]
                       .join(news, how="inner")
                       .join(macro, how="inner"))
        idx_h, gmm_h, hmm_h = fit_macro_one(
            "hybrid_full", hybrid_full,
            PRICE_FEATS + TEXT_SCALAR_FEATS + MACRO_FEATS_PREFIXED,
        )

        def build_macro_frame(idx, name, gmm_labels, hmm_labels):
            if idx is None:
                return pd.DataFrame()
            df = pd.DataFrame(index=idx)
            df[f"{name}_gmm"] = gmm_labels
            if hmm_labels is not None:
                df[name] = hmm_labels
            return df

        macro_out = (
            build_macro_frame(idx_m, "macro_only", gmm_m, hmm_m)
            .join(build_macro_frame(idx_d, "djpr_plus_macro", gmm_d, hmm_d), how="outer")
            .join(build_macro_frame(idx_pm, "price_plus_macro", gmm_pm, hmm_pm), how="outer")
            .join(build_macro_frame(idx_h, "hybrid_full", gmm_h, hmm_h), how="outer")
        )
        if macro_out.empty:
            print("\nMacro ablation produced no usable regimes (likely insufficient "
                  "macro coverage).  Skipping output.")
        else:
            macro_out.index.name = "date"
            macro_out.to_parquet(MACRO_OUTPUT_PATH)
            print(f"\nSaved: {MACRO_OUTPUT_PATH}  shape={macro_out.shape}")


# %% [markdown]
# ## 12. WRDS RavenPack macro TEXT ablation (LDA + BERTopic, optional)


# %%
if args.enable_macro:
    MACRO_LDA_PATH = f"{DATA_DIR}/lda_topics_macro_k15_daily.parquet"
    MACRO_BTOPIC_PATH = f"{DATA_DIR}/bertopic_macro_daily.parquet"
    MACRO_TEXT_OUTPUT_PATH = f"{DATA_DIR}/regimes_macro_text_supplement.parquet"

    have_macro_lda = os.path.exists(MACRO_LDA_PATH)
    have_macro_btopic = os.path.exists(MACRO_BTOPIC_PATH)

    if not (have_macro_lda or have_macro_btopic):
        print(f"\nMacro text ablation: neither {MACRO_LDA_PATH} nor "
              f"{MACRO_BTOPIC_PATH} found.  Run 04_topic_models.py --enable-macro "
              "first.  Skipping §12.")
    elif os.path.exists(MACRO_TEXT_OUTPUT_PATH):
        print(f"\nSkipping macro text ablation; output exists: {MACRO_TEXT_OUTPUT_PATH}")
    else:
        print("\n=== §12 Macro text ablation (LDA + BERTopic) ===")

        # ── Load macro text features (7-day rolling smoothing) ──────────────
        # Tuning history:
        #   v1 (21d rolling): macro_lda fusion mean_duration = 894 days
        #       — too smooth, HMM saw 2 regimes over 25 years.
        #   v2 (no rolling):  mean_duration too short (< 60 days)
        #       — too noisy, HMM oscillated.
        #   v3 (7d rolling, current): target mean_duration 60-120 days,
        #       matching DJPR baselines (text_lda_smooth-HMM = 62 days).
        # DJPR baselines stay at 21d (which works for them because DJPR has
        # ~500 headlines/day vs macro's ~50, so single-day topic dists are
        # already cleaner on DJPR side).
        ROLL_TEXT_MACRO = 7

        macro_lda_feat = None
        macro_btopic_feat = None
        MACRO_LDA_COLS: list[str] = []
        MACRO_BTOPIC_COLS: list[str] = []

        if have_macro_lda:
            macro_lda_feat = pd.read_parquet(MACRO_LDA_PATH).reindex(
                trading_days).rolling(
                window=ROLL_TEXT_MACRO, min_periods=3
            ).mean().add_prefix("macro_lda_")
            MACRO_LDA_COLS = list(macro_lda_feat.columns)
            print(f"Loaded macro LDA ({ROLL_TEXT_MACRO}d roll): {macro_lda_feat.shape}  "
                  f"NaN rows: {macro_lda_feat.isna().any(axis=1).sum()}")

        if have_macro_btopic:
            macro_btopic_feat = pd.read_parquet(MACRO_BTOPIC_PATH).reindex(
                trading_days).rolling(
                window=ROLL_TEXT_MACRO, min_periods=3
            ).mean().add_prefix("macro_btopic_")
            MACRO_BTOPIC_COLS = list(macro_btopic_feat.columns)
            print(f"Loaded macro BERTopic ({ROLL_TEXT_MACRO}d roll): {macro_btopic_feat.shape}  "
                  f"NaN rows: {macro_btopic_feat.isna().any(axis=1).sum()}")

        # Re-load DJPR LDA / BERTopic WITH SAME 7d rolling for §12 fusion ablations
        # (so DJPR + macro pairs are on matched smoothing).  The global `lda` (used
        # by §4 text_lda_smooth baseline) keeps its 21d rolling — only §12 uses 7d.
        djpr_lda_feat = None
        DJPR_LDA_COLS: list[str] = []
        for _k in [15, 20, 10]:
            _p = f"{DATA_DIR}/lda_topics_k{_k}_daily.parquet"
            if os.path.exists(_p):
                djpr_lda_feat = pd.read_parquet(_p).reindex(trading_days).rolling(
                    window=ROLL_TEXT_MACRO, min_periods=3
                ).mean()
                DJPR_LDA_COLS = list(djpr_lda_feat.columns)
                break

        djpr_btopic_feat = None
        DJPR_BTOPIC_COLS: list[str] = []
        if os.path.exists(BERTOPIC_PATH):
            djpr_btopic_feat = pd.read_parquet(BERTOPIC_PATH).reindex(
                trading_days).rolling(
                window=ROLL_TEXT_MACRO, min_periods=3
            ).mean()
            DJPR_BTOPIC_COLS = list(djpr_btopic_feat.columns)

        def fit_text_one(label, feat_df, feat_cols):
            aligned = feat_df[feat_cols].dropna()
            if len(aligned) < 250:
                print(f"  {label}: only {len(aligned)} aligned days — skipping.")
                return None, None, None
            X = StandardScaler().fit_transform(aligned.values)
            print(f"\n=== {label} ===  X: {X.shape}")
            k_g, gmm_m, labels_g = fit_best_gmm(X)
            k_h, hmm_m, labels_h = fit_best_hmm(X)
            pipeline_stats[f"{label}-GMM"] = compute_stats(gmm_m, X, k_g)
            pipeline_stats[f"{label}-HMM"] = compute_stats(hmm_m, X, k_h)
            print(f"  best GMM K={k_g}  HMM K={k_h}")
            return aligned.index, labels_g, labels_h

        # ── 5 ablations (using RAW topic distributions, no rolling) ──────────
        results: dict = {}

        # 1. macro_lda_only (15 dim)
        if macro_lda_feat is not None:
            results["macro_lda_only"] = fit_text_one(
                "macro_lda_only", macro_lda_feat, MACRO_LDA_COLS
            )

        # 2. macro_btopic_only (variable dim, ~15)
        if macro_btopic_feat is not None:
            results["macro_btopic_only"] = fit_text_one(
                "macro_btopic_only", macro_btopic_feat, MACRO_BTOPIC_COLS
            )

        # 3. djpr_lda + macro_lda fusion (30 dim)
        if macro_lda_feat is not None and djpr_lda_feat is not None:
            fused_lda = djpr_lda_feat.join(macro_lda_feat, how="inner")
            results["djpr_lda_plus_macro_lda"] = fit_text_one(
                "djpr_lda_plus_macro_lda", fused_lda,
                DJPR_LDA_COLS + MACRO_LDA_COLS,
            )

        # 4. djpr_btopic + macro_btopic fusion (variable, ~30 dim)
        if macro_btopic_feat is not None and djpr_btopic_feat is not None:
            fused_btopic = djpr_btopic_feat.join(macro_btopic_feat, how="inner")
            results["djpr_btopic_plus_macro_btopic"] = fit_text_one(
                "djpr_btopic_plus_macro_btopic", fused_btopic,
                DJPR_BTOPIC_COLS + MACRO_BTOPIC_COLS,
            )

        # 5. text_full_lda: price + djpr_lda + macro_lda (35 dim)
        if macro_lda_feat is not None and djpr_lda_feat is not None:
            text_full = (mkt[PRICE_FEATS]
                         .join(djpr_lda_feat, how="inner")
                         .join(macro_lda_feat, how="inner"))
            results["text_full_lda"] = fit_text_one(
                "text_full_lda", text_full,
                PRICE_FEATS + DJPR_LDA_COLS + MACRO_LDA_COLS,
            )

        # ── Build output frame ───────────────────────────────────────────────
        def build_text_frame(idx, name, gmm_labels, hmm_labels):
            if idx is None:
                return pd.DataFrame()
            df = pd.DataFrame(index=idx)
            df[f"{name}_gmm"] = gmm_labels
            if hmm_labels is not None:
                df[name] = hmm_labels
            return df

        frames = []
        for name, (idx, gmm_labels, hmm_labels) in results.items():
            frames.append(build_text_frame(idx, name, gmm_labels, hmm_labels))

        if not frames or all(f.empty for f in frames):
            print("\nMacro text ablation produced no usable regimes.  "
                  "Skipping output.")
        else:
            text_out = frames[0]
            for f in frames[1:]:
                if not f.empty:
                    text_out = text_out.join(f, how="outer")
            text_out.index.name = "date"
            text_out.to_parquet(MACRO_TEXT_OUTPUT_PATH)
            print(f"\nSaved: {MACRO_TEXT_OUTPUT_PATH}  shape={text_out.shape}")


# ── Persist pipeline_stats (Gap 4) ───────────────────────────────────────────
# Merge with existing JSON so partial re-runs incrementally update.
# Skipped sections didn't add to pipeline_stats, so their old entries survive.
import json as _json
_STATS_PATH = f"{DATA_DIR}/pipeline_stats.json"
if pipeline_stats:
    existing = {}
    if os.path.exists(_STATS_PATH):
        try:
            with open(_STATS_PATH) as _f:
                existing = _json.load(_f)
        except Exception as _e:
            print(f"[stats] failed to load {_STATS_PATH}: {_e}; starting fresh")
    existing.update(pipeline_stats)
    with open(_STATS_PATH, "w") as _f:
        _json.dump(existing, _f, indent=2)
    print(f"\n[stats] wrote {len(pipeline_stats)} new + {len(existing) - len(pipeline_stats)} cached → "
          f"{_STATS_PATH} ({len(existing)} total)")
else:
    print("\n[stats] no new pipelines fit this run; pipeline_stats.json unchanged")
