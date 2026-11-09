"""Shared helpers for finetuning grid sweeps.

Provides:
- Path constants (read-only consumption of artifacts under _data/)
- Representation loaders for DJPR scalar, macro scalar, market, and text
  (LDA-15/20, BERTopic, FinBERT-PCA20/PCA50/UMAP10).  Built-in dim reduction
  baked into FinBERT loaders (768-d is too high for raw clustering).
- Dim-reduction wrapper for post-stack reduction (none / PCA-95var / UMAP-10).
- Clustering: GMM/HMM/KMeans (copied from 06_text_hybrid_models.py to keep
  finetuning isolated from main pipeline imports).
- Eval: silhouette, DBI, mean_duration, BIC, AIC, log_likelihood.
- Grid runner: takes a feature assembly function + grid spec, returns one row
  per config, saves results parquet + top-10 visualization.

This file is imported by ft_news_fusion.py and ft_full_stack.py.  Not
intended to run standalone.
"""

from __future__ import annotations

# ── Self-install dependencies (Colab compatibility) ─────────────────────────
# The main pipeline (run_pipeline.py) handles deps before running scripts.
# The finetuning files are standalone — they need to ensure deps before the
# heavy imports below (hmmlearn, umap, etc.).  Marker file avoids re-install
# on every cell execution within the same runtime.
import os
import sys
import time
import subprocess
import hashlib
import tempfile
from pathlib import Path


def _ensure_deps():
    candidates = [
        Path("/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/requirements-project.txt"),
        Path("G:/我的云端硬盘/TTIC31220-Spring26/newsheadline/requirements-project.txt"),
    ]
    try:
        candidates.append(Path(__file__).parent.parent / "requirements-project.txt")
    except NameError:
        pass
    candidates.append(Path.cwd() / "requirements-project.txt")
    candidates.append(Path.cwd().parent / "requirements-project.txt")

    req_file = next((p for p in candidates if p.exists()), None)
    if req_file is None:
        print("[ft-deps] requirements-project.txt not found; assuming env ready")
        return

    req_hash = hashlib.sha256(req_file.read_bytes()).hexdigest()[:12]
    marker = Path(tempfile.gettempdir()) / f"newsheadline_ft_deps_{req_hash}.installed"
    if marker.exists():
        return  # Already installed this runtime; silent skip

    print(f"[ft-deps] installing from {req_file} (one-time per runtime) ...")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
        cwd=str(req_file.parent),
    )
    if proc.returncode != 0:
        raise SystemExit(f"[ft-deps] pip install failed (exit {proc.returncode})")
    try:
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    except Exception:
        pass
    print(f"[ft-deps] installed; marker: {marker}")


_ensure_deps()

# ── Now safe to import heavy dependencies ───────────────────────────────────
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from hmmlearn.hmm import GaussianHMM


# ── Path constants ──────────────────────────────────────────────────────────
PROJECT_DIR = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline"
if not os.path.exists(PROJECT_DIR):
    # Fallback for local syntax check
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(PROJECT_DIR, "_data")
IMAGES_DIR = os.path.join(PROJECT_DIR, "_images")

DJPR_SCALAR_PATH  = f"{DATA_DIR}/news_features_daily.parquet"
MACRO_SCALAR_PATH = f"{DATA_DIR}/news_features_macro_daily.parquet"
MARKET_PATH       = f"{DATA_DIR}/market_features.parquet"

DJPR_LDA_15  = f"{DATA_DIR}/lda_topics_k15_daily.parquet"
DJPR_LDA_20  = f"{DATA_DIR}/lda_topics_k20_daily.parquet"
MACRO_LDA_15 = f"{DATA_DIR}/lda_topics_macro_k15_daily.parquet"
MACRO_LDA_20 = f"{DATA_DIR}/lda_topics_macro_k20_daily.parquet"

DJPR_BTOPIC  = f"{DATA_DIR}/bertopic_daily.parquet"
MACRO_BTOPIC = f"{DATA_DIR}/bertopic_macro_daily.parquet"

DJPR_FINBERT_NPZ  = f"{DATA_DIR}/daily_finbert_embeddings.npz"
MACRO_FINBERT_NPZ = f"{DATA_DIR}/daily_finbert_embeddings_macro.npz"


# ── Defaults (match main pipeline conventions) ──────────────────────────────
RANDOM_STATE = 42
DEFAULT_ROLL_SCALAR = 35
DEFAULT_ROLL_LDA    = 30
DEFAULT_ROLL_BTOPIC = 21
DEFAULT_ROLL_MACRO  = 30
DEFAULT_ROLL_FINBERT = 10   # FinBERT was originally 10d in DJPR pipeline

PRICE_FEATS = ["ret", "vol_21", "vix", "term_spread", "credit_spread"]
SCALAR_FEATS = ["css_mean", "css_std", "n_articles", "ess_mean", "sent_net"]

HIGH_DIM_THRESHOLD = 10   # HMM: dim >= 10 → diag covariance


# ── Trading-day index (shared anchor) ───────────────────────────────────────
def get_trading_days() -> pd.DatetimeIndex:
    mkt = pd.read_parquet(MARKET_PATH)
    return mkt[PRICE_FEATS].dropna().index


# ── Representation loaders ──────────────────────────────────────────────────
def load_price() -> pd.DataFrame:
    """Market price features.  Already smoothed inside market_features.parquet."""
    return pd.read_parquet(MARKET_PATH)[PRICE_FEATS]


def load_djpr_scalar(roll: int = DEFAULT_ROLL_SCALAR) -> pd.DataFrame:
    raw = pd.read_parquet(DJPR_SCALAR_PATH)[SCALAR_FEATS]
    return raw.reindex(get_trading_days()).rolling(
        window=roll, min_periods=max(3, roll // 4)
    ).mean()


def load_macro_scalar(roll: int = DEFAULT_ROLL_MACRO) -> pd.DataFrame:
    raw = pd.read_parquet(MACRO_SCALAR_PATH)[SCALAR_FEATS]
    return raw.reindex(get_trading_days()).rolling(
        window=roll, min_periods=max(3, roll // 4)
    ).mean().add_prefix("macro_")


def _apply_pca(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """PCA on rows of df (drops NaN rows first, returns reindexed result)."""
    clean = df.dropna()
    if clean.empty:
        return df.iloc[:0]   # empty same shape
    pca = PCA(n_components=min(n_components, clean.shape[1]),
              random_state=RANDOM_STATE)
    reduced = pca.fit_transform(StandardScaler().fit_transform(clean.values))
    out = pd.DataFrame(
        reduced, index=clean.index,
        columns=[f"pc{i}" for i in range(reduced.shape[1])],
    )
    return out.reindex(df.index)


def _apply_umap(df: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """UMAP non-linear dim reduction.  Slow — call sparingly."""
    from umap import UMAP
    clean = df.dropna()
    if clean.empty:
        return df.iloc[:0]
    reducer = UMAP(n_components=n_components, n_neighbors=30, min_dist=0.0,
                   metric="cosine", random_state=RANDOM_STATE)
    reduced = reducer.fit_transform(StandardScaler().fit_transform(clean.values))
    out = pd.DataFrame(
        reduced, index=clean.index,
        columns=[f"u{i}" for i in range(reduced.shape[1])],
    )
    return out.reindex(df.index)


def load_text_repr(method: str, corpus: str,
                   roll: int | None = None) -> pd.DataFrame:
    """Return a date-indexed DataFrame for a (text method, corpus) combination.

    method ∈ {'LDA15', 'LDA20', 'BERTopic', 'FinBERT_PCA20', 'FinBERT_PCA50',
              'FinBERT_UMAP10'}
    corpus ∈ {'djpr', 'macro'}
    roll: rolling window for smoothing.  None uses representation-appropriate default.

    For FinBERT methods, built-in dim reduction (PCA or UMAP) is applied to the
    raw 768-d before rolling/smoothing — Option B in plan (reduce-then-smooth).
    """
    if corpus not in ("djpr", "macro"):
        raise ValueError(f"corpus must be 'djpr' or 'macro', got {corpus!r}")

    trading_days = get_trading_days()
    prefix = "" if corpus == "djpr" else "macro_"

    if method == "LDA15":
        path = DJPR_LDA_15 if corpus == "djpr" else MACRO_LDA_15
        df = pd.read_parquet(path).reindex(trading_days)
        eff_roll = roll if roll is not None else DEFAULT_ROLL_LDA
    elif method == "LDA20":
        path = DJPR_LDA_20 if corpus == "djpr" else MACRO_LDA_20
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run 04_topic_models.py first")
        df = pd.read_parquet(path).reindex(trading_days)
        eff_roll = roll if roll is not None else DEFAULT_ROLL_LDA
    elif method == "BERTopic":
        path = DJPR_BTOPIC if corpus == "djpr" else MACRO_BTOPIC
        df = pd.read_parquet(path).reindex(trading_days)
        eff_roll = roll if roll is not None else DEFAULT_ROLL_BTOPIC
    elif method.startswith("FinBERT"):
        npz_path = DJPR_FINBERT_NPZ if corpus == "djpr" else MACRO_FINBERT_NPZ
        npz = np.load(npz_path, allow_pickle=True)
        df_768 = pd.DataFrame(
            npz["embeddings"].astype(np.float32),
            index=pd.to_datetime(npz["dates"]),
        ).reindex(trading_days)
        # Built-in dim reduction
        if method == "FinBERT_PCA20":
            df = _apply_pca(df_768, 20)
        elif method == "FinBERT_PCA50":
            df = _apply_pca(df_768, 50)
        elif method == "FinBERT_UMAP10":
            df = _apply_umap(df_768, 10)
        else:
            raise ValueError(f"unknown FinBERT variant {method!r}")
        eff_roll = roll if roll is not None else DEFAULT_ROLL_FINBERT
    else:
        raise ValueError(f"unknown text method {method!r}")

    # Apply rolling smoothing
    df = df.rolling(window=eff_roll, min_periods=max(3, eff_roll // 4)).mean()
    df = df.add_prefix(prefix)
    return df


# ── Post-stack dim reduction ────────────────────────────────────────────────
def reduce_dim_post(X: np.ndarray, method: str) -> tuple[np.ndarray, int]:
    """Apply optional dim reduction to the assembled, already-standardized X.
    method ∈ {'none', 'pca95', 'umap10'}.  Returns (reduced_X, n_components_out).
    """
    if method == "none":
        return X, X.shape[1]
    if method == "pca95":
        # Keep enough PCs for 95% variance
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        Z = pca.fit_transform(X)
        return Z, Z.shape[1]
    if method == "umap10":
        from umap import UMAP
        reducer = UMAP(n_components=10, n_neighbors=30, min_dist=0.0,
                       metric="cosine", random_state=RANDOM_STATE)
        Z = reducer.fit_transform(X)
        return Z, Z.shape[1]
    raise ValueError(f"unknown dim_red method {method!r}")


# ── Clustering (copied from 06_text_hybrid_models.py) ───────────────────────
def fit_best_gmm(X: np.ndarray, k_values: list[int] = [3, 4, 5]):
    """Return (best_k, gmm, labels) chosen by BIC."""
    results = {}
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            n_init=5, random_state=RANDOM_STATE,
        )
        gmm.fit(X)
        results[k] = (gmm.bic(X), gmm, gmm.predict(X))
    best_k = min(results, key=lambda k: results[k][0])
    return best_k, results[best_k][1], results[best_k][2]


def fit_best_hmm(X: np.ndarray, k_values: list[int] = [3, 4, 5], n_trials: int = 5):
    """Sticky-init HMM with adaptive covariance (diag for dim>=10)."""
    d = X.shape[1]
    cov_type = "diag" if d >= HIGH_DIM_THRESHOLD else "full"
    best_score, best_k, best_model, best_labels = -np.inf, None, None, None
    for k in k_values:
        for seed in range(n_trials):
            try:
                m = GaussianHMM(
                    n_components=k, covariance_type=cov_type,
                    n_iter=300, random_state=seed, tol=1e-5,
                    init_params="smc",
                )
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


def fit_kmeans(X: np.ndarray, k: int):
    """KMeans returns (k, None_model, labels)."""
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    return k, km, labels


# ── Evaluation (copied from 06/07) ──────────────────────────────────────────
def compute_stats(model, X: np.ndarray, K: int) -> dict:
    """log-likelihood / BIC / AIC.  Returns NaNs for KMeans (no likelihood)."""
    if model is None:
        return {"log_likelihood": np.nan, "bic": np.nan, "aic": np.nan}
    N, D = X.shape
    if hasattr(model, "bic") and hasattr(model, "aic"):
        ll = float(model.score(X) * N)
        return {
            "log_likelihood": ll,
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X)),
        }
    if hasattr(model, "score") and hasattr(model, "covariance_type"):
        ll = float(model.score(X))
        cov_params = {
            "full":      K * D * (D + 1) // 2,
            "diag":      K * D,
            "spherical": K,
            "tied":      D * (D + 1) // 2,
        }[model.covariance_type]
        n_params = (K - 1) + K * (K - 1) + K * D + cov_params
        bic = -2 * ll + n_params * np.log(N)
        aic = -2 * ll + 2 * n_params
        return {"log_likelihood": ll, "bic": float(bic), "aic": float(aic)}
    return {"log_likelihood": np.nan, "bic": np.nan, "aic": np.nan}


def clustering_quality(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Silhouette + Davies-Bouldin.  Auto PCA-compress >50d for silhouette efficiency."""
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    X_use = X
    if X.shape[1] > 50:
        X_use = PCA(n_components=20, random_state=RANDOM_STATE).fit_transform(X)
    sil = silhouette_score(X_use, labels, sample_size=5000, random_state=RANDOM_STATE)
    dbi = davies_bouldin_score(X_use, labels)
    return float(sil), float(dbi)


def mean_duration(labels: np.ndarray) -> float:
    """Average consecutive-day run length per regime."""
    if len(labels) == 0:
        return float("nan")
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


# ── Grid runner ─────────────────────────────────────────────────────────────
@dataclass
class GridConfig:
    text_method: str
    dim_red: str
    cluster: str
    K_explicit: int | None   # None for GMM/HMM auto-select; int for KMeans

    def label(self) -> str:
        k_str = f"K{self.K_explicit}" if self.K_explicit is not None else "Kauto"
        return f"{self.text_method}|{self.dim_red}|{self.cluster}|{k_str}"


def default_grid() -> list[GridConfig]:
    """6 text × 3 dim_red × 5 cluster-K combos = 90 configs.

    LDA/BERTopic GMM and KMeans skip K=3 (mega-cluster pathology from Knob 1).
    """
    grid = []
    text_methods = [
        "LDA15", "LDA20", "BERTopic",
        "FinBERT_PCA20", "FinBERT_PCA50", "FinBERT_UMAP10",
    ]
    dim_reds = ["none", "pca95", "umap10"]
    for tm in text_methods:
        for dr in dim_reds:
            # GMM auto-selects K via BIC
            grid.append(GridConfig(tm, dr, "GMM", None))
            # HMM auto-selects K via log-likelihood
            grid.append(GridConfig(tm, dr, "HMM", None))
            # KMeans: explicit K
            for k in [3, 4, 5]:
                grid.append(GridConfig(tm, dr, "KMeans", k))
    return grid


def run_one_config(cfg: GridConfig, feature_fn: Callable[[str], pd.DataFrame]) -> dict:
    """Run one grid cell. feature_fn(text_method) returns the assembled feature DF.
    Returns a dict suitable for DataFrame row construction."""
    t0 = time.time()
    try:
        feat_df = feature_fn(cfg.text_method)
        feat_df = feat_df.dropna()
        if len(feat_df) < 250:
            return _empty_row(cfg, error=f"too few rows after dropna ({len(feat_df)})")

        dim_in = feat_df.shape[1]
        X = StandardScaler().fit_transform(feat_df.values)
        X_red, dim_out = reduce_dim_post(X, cfg.dim_red)

        if cfg.cluster == "GMM":
            K, model, labels = fit_best_gmm(X_red, k_values=[3, 4, 5])
        elif cfg.cluster == "HMM":
            K, model, labels = fit_best_hmm(X_red, k_values=[3, 4, 5])
        elif cfg.cluster == "KMeans":
            K, model, labels = fit_kmeans(X_red, cfg.K_explicit)
        else:
            return _empty_row(cfg, error=f"unknown cluster {cfg.cluster!r}")

        if labels is None:
            return _empty_row(cfg, error="cluster fit returned None labels")

        sil, dbi = clustering_quality(X_red, labels)
        dur = mean_duration(labels)
        stats = compute_stats(model, X_red, K)

        elapsed = time.time() - t0
        return {
            "text_method":    cfg.text_method,
            "dim_red":        cfg.dim_red,
            "cluster":        cfg.cluster,
            "K":              int(K),
            "dim_in":         dim_in,
            "dim_out":        dim_out,
            "n_obs":          len(feat_df),
            "silhouette":     round(sil, 4),
            "dbi":            round(dbi, 4),
            "mean_duration":  round(dur, 1),
            "log_likelihood": round(stats["log_likelihood"], 1) if not np.isnan(stats["log_likelihood"]) else np.nan,
            "bic":            round(stats["bic"], 1) if not np.isnan(stats["bic"]) else np.nan,
            "aic":            round(stats["aic"], 1) if not np.isnan(stats["aic"]) else np.nan,
            "elapsed_s":      round(elapsed, 1),
            "error":          "",
        }
    except Exception as exc:
        return _empty_row(cfg, error=f"{type(exc).__name__}: {exc}")


def _empty_row(cfg: GridConfig, error: str) -> dict:
    return {
        "text_method":    cfg.text_method,
        "dim_red":        cfg.dim_red,
        "cluster":        cfg.cluster,
        "K":              cfg.K_explicit or -1,
        "dim_in":         -1,
        "dim_out":        -1,
        "n_obs":          0,
        "silhouette":     np.nan,
        "dbi":            np.nan,
        "mean_duration":  np.nan,
        "log_likelihood": np.nan,
        "bic":            np.nan,
        "aic":            np.nan,
        "elapsed_s":      np.nan,
        "error":          error,
    }


def run_grid(name: str, feature_fn: Callable[[str], pd.DataFrame],
             grid: list[GridConfig] | None = None,
             out_data_dir: str = DATA_DIR,
             out_images_dir: str = IMAGES_DIR) -> pd.DataFrame:
    """Run the full grid, save results parquet + top-10 visualization.

    Returns the full grid as a DataFrame.
    """
    if grid is None:
        grid = default_grid()
    print(f"=== {name}: running {len(grid)} configs ===\n")

    rows = []
    for i, cfg in enumerate(grid, 1):
        print(f"[{i:3d}/{len(grid)}] {cfg.label():40s}", end="  ", flush=True)
        row = run_one_config(cfg, feature_fn)
        if row.get("error"):
            print(f"ERROR: {row['error']}")
        else:
            print(f"sil={row['silhouette']:.3f}  dur={row['mean_duration']:.0f}  "
                  f"({row['elapsed_s']:.1f}s)")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_parquet = os.path.join(out_data_dir, f"ft_{name}_results.parquet")
    df.to_parquet(out_parquet)
    print(f"\nSaved: {out_parquet}  shape={df.shape}")

    # ── Top-10 visualization ────────────────────────────────────────────────
    valid = df[df["silhouette"].notna()].sort_values("silhouette", ascending=False)
    top = valid.head(10)
    if len(top) > 0:
        fig, ax = plt.subplots(figsize=(12, max(4, len(top) * 0.45)))
        labels = [f"{r.text_method}|{r.dim_red}|{r.cluster}|K{int(r.K)}"
                  for r in top.itertuples()]
        bars = ax.barh(range(len(top)), top["silhouette"].values, color="steelblue")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("silhouette score")
        ax.set_title(f"{name} — top 10 configs (n={len(grid)} total)")
        # Annotate with duration
        for i, (bar, r) in enumerate(zip(bars, top.itertuples())):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f" dur={r.mean_duration:.0f}d", va="center", fontsize=8, color="gray")
        plt.tight_layout()
        out_png = os.path.join(out_images_dir, f"ft_{name}_top10.png")
        plt.savefig(out_png, dpi=120)
        plt.close("all")
        print(f"Saved: {out_png}")

    # ── Print top-5 summary ─────────────────────────────────────────────────
    print(f"\n=== {name}: top 5 configs by silhouette ===")
    cols = ["text_method", "dim_red", "cluster", "K", "dim_out",
            "silhouette", "mean_duration", "dbi", "bic"]
    print(valid.head(5)[cols].to_string(index=False))
    print()

    # ── Sanity: how many errors? ────────────────────────────────────────────
    errors = df[df["error"] != ""]
    if len(errors) > 0:
        print(f"WARNING: {len(errors)} configs errored:")
        for _, r in errors.iterrows():
            print(f"  {r['text_method']}|{r['dim_red']}|{r['cluster']}: {r['error']}")

    return df


# ── Module sanity (only runs if executed directly, never on import) ─────────
if __name__ == "__main__":
    print("This module provides helpers for finetuning grid sweeps.")
    print("Import from ft_news_fusion.py or ft_full_stack.py, do not run directly.")
    print(f"\nData dir: {DATA_DIR}  exists: {os.path.exists(DATA_DIR)}")
    print(f"Trading days: {len(get_trading_days())}")
    print(f"Default grid size: {len(default_grid())} configs")
