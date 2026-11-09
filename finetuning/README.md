# Finetuning — Grid Search for News-Fusion Pipelines

Two standalone grid sweeps (`.py` files with `# %%` cell markers) that explore
the **{text representation × pre-clustering dim reduction × clustering method × K}**
space for two specific feature combinations. These are parallel to the main
pipeline (06/07/08), reading from existing artifacts under `_data/` but writing
their own results files.

## What each file does

| File | Pipeline | Feature stack | Expected dim |
|---|---|---|---|
| `_common.py` | (helper module) | Representation loaders, dim-reduction wrappers, clustering, evaluation, grid runner. Self-contained — copies helpers from `06_text_hybrid_models.py` and `07_evaluation.py` to avoid triggering main-pipeline imports. | — |
| `ft_news_fusion.py` | **A: news-only 4-way fusion** | DJPR scalar (5) + Macro scalar (5) + DJPR text (varies) + Macro text (varies) | 30-110 dim |
| `ft_full_stack.py` | **B: price + 4-way news** | Price (5) + Pipeline A | 35-115 dim |

Each grid file runs **90 configs**:
- 6 text representations: `LDA15`, `LDA20`, `BERTopic`, `FinBERT_PCA20`, `FinBERT_PCA50`, `FinBERT_UMAP10`
- 3 post-stack dim reductions: `none`, `pca95` (95% var), `umap10` (non-linear)
- 5 clustering setups: GMM (auto-K via BIC), HMM (auto-K via log-likelihood), KMeans K=3,4,5

## What this addresses

Two methodological gaps in the main pipeline (06/07):

1. **FinBERT was undertested.** The 768-d embeddings only feed `08`'s semantic
   retrieval; never as clustering input. The early `text_embed_roll-*`
   variants were dropped due to low silhouette, but only at PCA-20 + GMM.
   This sweep gives FinBERT a fair shot across PCA-20/50 and UMAP-10, with all
   three clustering methods.

2. **Pre-clustering dim reduction was hardcoded.** Only `hybrid_pca` in the
   main pipeline used PCA. This sweep adds `pca95` and `umap10` as a tunable
   knob applied to the assembled X.

## How to run

In Colab (or any Python 3 environment with the project dependencies):

```python
# Run both sweeps. Each takes ~15-25 min.
!python /content/drive/MyDrive/TTIC31220-Spring26/newsheadline/finetuning/ft_news_fusion.py
!python /content/drive/MyDrive/TTIC31220-Spring26/newsheadline/finetuning/ft_full_stack.py
```

Or, in VSCode / Jupyter, open either `.py` file — it will render as a notebook
(percent format, jupytext-compatible).

### Prerequisites

The sweeps **read** these artifacts (must exist before running):

- `_data/news_features_daily.parquet`
- `_data/news_features_macro_daily.parquet`
- `_data/lda_topics_k15_daily.parquet`, `lda_topics_k20_daily.parquet`
- `_data/lda_topics_macro_k15_daily.parquet`, `lda_topics_macro_k20_daily.parquet`
- `_data/bertopic_daily.parquet`, `bertopic_macro_daily.parquet`
- `_data/daily_finbert_embeddings.npz`, `daily_finbert_embeddings_macro.npz`
- `_data/market_features.parquet`

If any are missing, run `run_pipeline.py` with `--enable-macro` first.

## How to read results

Each sweep writes:

```
_data/ft_<name>_results.parquet   # one row per config (~90 rows × 14 cols)
_images/ft_<name>_top10.png       # horizontal bar chart, top-10 by silhouette
```

Quick inspection:

```python
import pandas as pd
r = pd.read_parquet("_data/ft_news_fusion_results.parquet")

# Top 5 by silhouette
print(r.sort_values("silhouette", ascending=False).head(5)[
    ["text_method","dim_red","cluster","K","silhouette","mean_duration","bic"]
])

# Best per text method (which text type wins overall)
print(r.loc[r.groupby("text_method")["silhouette"].idxmax()][
    ["text_method","cluster","silhouette","mean_duration"]
])

# Which dim_red helps?
print(r.groupby("dim_red")["silhouette"].mean())
```

Sample columns in the results parquet:

| Column | Meaning |
|---|---|
| `text_method` | LDA15 / LDA20 / BERTopic / FinBERT_PCA20 / FinBERT_PCA50 / FinBERT_UMAP10 |
| `dim_red` | none / pca95 / umap10 |
| `cluster` | GMM / HMM / KMeans |
| `K` | regime count selected (auto by BIC for GMM/HMM, explicit for KMeans) |
| `dim_in` | feature dimensionality after stacking, before reduction |
| `dim_out` | dimensionality after `dim_red` (= `dim_in` if `none`) |
| `n_obs` | aligned trading days (after dropna) |
| `silhouette` | sklearn silhouette score (higher = better cluster separation) |
| `dbi` | Davies-Bouldin index (lower = better) |
| `mean_duration` | average regime persistence in days (target 60-365 for interpretability) |
| `log_likelihood`, `bic`, `aic` | statistical fit (NaN for KMeans) |
| `elapsed_s` | wall time for this config |
| `error` | non-empty if the fit failed (config skipped) |

## Promoting a best config back to the main pipeline

If the sweep identifies a winner you want in the main `eval_summary.parquet`,
edit `06_text_hybrid_models.py` §12 to add a new entry:

```python
# Example: ft_news_fusion winner was FinBERT_PCA50 + umap10 + HMM
if macro_lda_feat is not None:                    # gate on macro corpus availability
    # 1. Reproduce the feature stack
    djpr_finbert = C.load_text_repr("FinBERT_PCA50", "djpr")
    macro_finbert = C.load_text_repr("FinBERT_PCA50", "macro")
    fused = (djpr_scalar.join(macro_scalar, how="inner")
                        .join(djpr_finbert, how="inner")
                        .join(macro_finbert, how="inner"))
    fused_cols = list(fused.columns)
    # 2. Add to results dict
    results["news_fusion_tuned"] = fit_text_one(
        "news_fusion_tuned", fused, fused_cols
    )
```

Then 07's macro-text pickup auto-includes it (via `regimes_macro_text_supplement.parquet`).

Or simpler: keep the tuning results separate and reference the parquet directly
in the report.

## Runtime + compute notes

- ~15-25 min per file on Colab CPU (no GPU needed — FinBERT embeddings are
  pre-computed; the sweep only does PCA / UMAP / clustering)
- UMAP is the slowest dim reduction (~5s vs PCA's <1s). Each `umap10` config
  costs ~10-30s extra.
- HMM with `n_trials=5` per K and 3 K values is the slowest clustering
  (~10-30s per config). GMM and KMeans are 2-5s each.
- Total memory budget: < 1GB (FinBERT embeddings are 6000 × 768 floats = 18MB).

## Limitations

- **Smoothing window is not swept.** Each representation uses its
  representation-appropriate default (set in `_common.py`).
- **No Bayesian / random search** — pure grid. If you want a finer search
  around a discovered optimum, edit `_common.default_grid()` to narrow the grid.
- **Auto-K only respects `[3,4,5]`** — GMM/HMM internal grid is hardcoded.
  KMeans variants explicitly test K=3,4,5 (so KMeans is over-represented in
  the grid; this is intentional to cover the no-likelihood-criterion case).
