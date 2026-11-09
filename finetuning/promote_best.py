# Promote the best finetuning configs into regime label parquets.
#
# Reproduces:
#   Pipeline A (news-only):  FinBERT_UMAP10 | umap10 | GMM | K=5
#       → _data/regimes_ft_news_fusion_best.parquet
#   Pipeline B (price+news): FinBERT_UMAP10 | umap10 | GMM | K=5
#       → _data/regimes_ft_full_stack_best.parquet
#
# Also produces a 2-panel side-by-side timeline figure for visualization:
#   _images/ft_best_comparison.png
#
# Run after both ft_news_fusion.py and ft_full_stack.py have completed.

# %%
import os
try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass

# Robust _common.py resolution (same idiom as ft_news_fusion.py)
import sys
_FT_PATH_CANDIDATES = []
try:
    _FT_PATH_CANDIDATES.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
_FT_PATH_CANDIDATES.extend([
    "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/finetuning",
    "G:/我的云端硬盘/TTIC31220-Spring26/newsheadline/finetuning",
    os.path.join(os.getcwd(), "finetuning"),
    os.getcwd(),
])
for _cand in _FT_PATH_CANDIDATES:
    if os.path.exists(os.path.join(_cand, "_common.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        print(f"[promote_best] _common.py resolved at: {_cand}")
        break
else:
    raise ImportError(f"Cannot find _common.py.  Tried: {_FT_PATH_CANDIDATES}.")
import _common as C

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# ## Best configs (from ft_*_results.parquet inspection)


# %%
# Config choice rationale:
# The "highest silhouette" grid winners were FinBERT_UMAP10|umap10|GMM|K=5
# (A: sil=0.56/dur=272; B: sil=0.66/dur=418).  But UMAP-on-UMAP geometric
# collapse turned those into "market era detection" (3-5 regimes each covering
# multiple years), not classical regime detection.
#
# We picked the *second-tier* sweet spot: BERTopic|pca95|KMeans|K=5.
#   - sil clearly beats price-HMM (0.27 vs 0.20)
#   - mean_duration ~200d shows actual regime switching dynamics
#   - K=5 matches price-HMM convention
#   - Same config on both A and B → isolates the marginal effect of adding price
#
# Trade-off: KMeans has no likelihood, so log_likelihood/BIC/AIC columns are NaN.
BEST_CONFIGS = {
    "ft_news_fusion_best": {
        "label":        "news_fusion (DJPR_S+Macro_S+DJPR_BTopic+Macro_BTopic)",
        "text_method":  "BERTopic",
        "dim_red":      "pca95",
        "cluster":      "KMeans",
        "K":            5,
        "include_price": False,
        # Reference metrics from sweep (for sanity verification):
        "expected_sil": 0.279,
        "expected_dur": 202,
    },
    "ft_full_stack_best": {
        "label":        "full_stack (price + news_fusion)",
        "text_method":  "BERTopic",
        "dim_red":      "pca95",
        "cluster":      "KMeans",
        "K":            5,
        "include_price": True,
        "expected_sil": 0.272,
        "expected_dur": 224,
    },
}

EPISODES = [
    ("Dot-com burst",   "2000-03-10", "2002-10-09"),
    ("9/11",            "2001-09-11", "2001-09-21"),
    ("GFC",             "2007-10-09", "2009-03-09"),
    ("COVID crash",     "2020-02-19", "2020-03-23"),
    ("Rate hike cycle", "2022-01-03", "2022-12-31"),
]


# %% [markdown]
# ## Helpers


# %%
def assemble_features(text_method: str, include_price: bool) -> pd.DataFrame:
    """Reproduce the exact feature stack used in the sweep."""
    djpr_scalar  = C.load_djpr_scalar()
    macro_scalar = C.load_macro_scalar()
    djpr_text    = C.load_text_repr(text_method, "djpr")
    macro_text   = C.load_text_repr(text_method, "macro")
    parts = [djpr_scalar, macro_scalar, djpr_text, macro_text]
    if include_price:
        parts = [C.load_price()] + parts
    X = parts[0]
    for p in parts[1:]:
        X = X.join(p, how="inner")
    return X.dropna()


def fit_and_save(name: str, cfg: dict) -> pd.DataFrame:
    """Fit one config, save labels parquet, print per-regime market stats."""
    print(f"\n{'='*72}\n{name}: {cfg['label']}\n{'='*72}")
    print(f"  text_method={cfg['text_method']}  dim_red={cfg['dim_red']}  "
          f"cluster={cfg['cluster']}  K={cfg['K']}  price={cfg['include_price']}")

    feat_df = assemble_features(cfg["text_method"], cfg["include_price"])
    print(f"  Features: {feat_df.shape}  (dropna applied)")

    X = StandardScaler().fit_transform(feat_df.values)
    X_red, dim_out = C.reduce_dim_post(X, cfg["dim_red"])
    print(f"  After {cfg['dim_red']}: dim {X.shape[1]} → {dim_out}")

    if cfg["cluster"] == "GMM":
        K, model, labels = C.fit_best_gmm(X_red, k_values=[cfg["K"]])
    elif cfg["cluster"] == "HMM":
        K, model, labels = C.fit_best_hmm(X_red, k_values=[cfg["K"]])
    elif cfg["cluster"] == "KMeans":
        K, model, labels = C.fit_kmeans(X_red, cfg["K"])
    else:
        raise ValueError(f"unknown cluster {cfg['cluster']!r}")

    # Verify sanity vs expected metrics
    sil, dbi = C.clustering_quality(X_red, labels)
    dur = C.mean_duration(labels)
    print(f"  Achieved: sil={sil:.4f}  dur={dur:.1f}  dbi={dbi:.4f}  K={K}")
    print(f"  Expected: sil≈{cfg['expected_sil']}  dur≈{cfg['expected_dur']}")

    # Save labels
    out = pd.DataFrame({"regime": labels.astype(int)}, index=feat_df.index)
    out.index.name = "date"
    out_path = os.path.join(C.DATA_DIR, f"regimes_{name}.parquet")
    out.to_parquet(out_path)
    print(f"  Saved: {out_path}  shape={out.shape}")

    # ── Per-regime market + news summary ────────────────────────────────────
    mkt = pd.read_parquet(C.MARKET_PATH)
    news = pd.read_parquet(C.DJPR_SCALAR_PATH)[["css_mean", "n_articles"]]
    joined = (mkt.join(news, how="left")
                 .join(out, how="inner")
                 .dropna(subset=["regime"]))
    print(f"\n  Per-regime stats:")
    print(f"  {'reg':>3} {'n_days':>6} {'date_from':>10} {'date_to':>10} "
          f"{'ann_ret':>8} {'ann_vol':>8} {'sharpe':>7} {'vix':>6} {'css':>7}")
    for r in sorted(out["regime"].unique()):
        grp = joined[joined["regime"] == r]
        if grp.empty:
            continue
        ann_ret = grp["ret"].mean() * 252
        ann_vol = grp["ret"].std() * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        print(f"  {int(r):>3} {len(grp):>6d} {str(grp.index.min().date()):>10} "
              f"{str(grp.index.max().date()):>10} {ann_ret:>+8.2%} "
              f"{ann_vol:>8.2%} {sharpe:>+7.2f} "
              f"{grp['vix'].mean():>6.2f} {grp['css_mean'].mean():>+7.4f}")
    return out


# %% [markdown]
# ## Fit and save both best configs


# %%
all_labels = {}
for name, cfg in BEST_CONFIGS.items():
    all_labels[name] = fit_and_save(name, cfg)


# %% [markdown]
# ## 2-panel side-by-side timeline comparison figure


# %%
def plot_timeline(ax, labels_df: pd.DataFrame, mkt: pd.DataFrame, title: str):
    joined = mkt[["ret"]].join(labels_df, how="inner").dropna(subset=["regime"])
    dates = joined.index.to_numpy()
    cum = np.exp(joined["ret"].cumsum())
    lab = joined["regime"].values.astype(int)
    K = lab.max() + 1
    colors = plt.cm.Set1.colors

    ax.plot(dates, cum, linewidth=0.6, color="black", zorder=3)
    for r in range(K):
        mask = lab == r
        ax.fill_between(dates, 0, cum, where=mask, alpha=0.32,
                        color=colors[r % len(colors)], label=f"R{r}")
    for ep_name, start, end in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.08, color="gray", zorder=0)
    ax.set_ylabel("S&P 500 ($1 → cum)")
    ax.legend(fontsize=8, loc="upper left", ncol=K)
    ax.set_title(title, fontsize=11, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


print(f"\n{'='*72}\nGenerating 2-panel comparison figure...\n{'='*72}")
mkt = pd.read_parquet(C.MARKET_PATH)

fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
plot_timeline(axes[0], all_labels["ft_news_fusion_best"], mkt,
              "Pipeline A — news_fusion_best (BERTopic + PCA95 + KMeans K=5, sil≈0.28, dur≈200d)")
plot_timeline(axes[1], all_labels["ft_full_stack_best"], mkt,
              "Pipeline B — full_stack_best (Pipeline A + price, sil≈0.27, dur≈225d)")

# Add episode labels only on top panel
for ep_name, start, _ in EPISODES:
    axes[0].text(pd.Timestamp(start), axes[0].get_ylim()[1] * 0.95, ep_name,
                 fontsize=7, rotation=90, va="top", color="gray")

plt.tight_layout()
out_fig = os.path.join(C.IMAGES_DIR, "ft_best_comparison.png")
plt.savefig(out_fig, dpi=120)
plt.close("all")
print(f"Saved: {out_fig}")


# %% [markdown]
# ## How to use these labels in 08 (regime characterization tour)


# %%
print(f"""

{'=' * 72}
Done.  Outputs:
  _data/regimes_ft_news_fusion_best.parquet  (column: 'regime')
  _data/regimes_ft_full_stack_best.parquet   (column: 'regime')
  _images/ft_best_comparison.png

To include in the existing 4-panel tour (08_regime_characterization.py),
edit PIPELINE_LIST near the top of 08 to append:

    PIPELINE_LIST = [
        ("price-HMM",              "regimes_price.parquet",            "hmm_k5_full"),
        ("text_scalar_smooth-HMM", "regimes_text_scalar.parquet",      "hmm"),
        ("macro_only-HMM",         "regimes_macro_supplement.parquet", "macro_only"),
        ("djpr_plus_macro-HMM",    "regimes_macro_supplement.parquet", "djpr_plus_macro"),
        ("ft_news_fusion_best",    "regimes_ft_news_fusion_best.parquet", "regime"),
        ("ft_full_stack_best",     "regimes_ft_full_stack_best.parquet",  "regime"),
    ]

Then re-run 08 to get a 6-panel tour with the tuned configs included.
{'=' * 72}
""")
