# Auto-generated from 05_baseline_price.ipynb
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
# # 05 — Price-Only Baseline: HMM + GMM


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# ## 1. Load + Prepare Features


# %%
mkt = pd.read_parquet('/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data/market_features.parquet')

# Drop the first 21 rows (vol_21 has NaN from rolling window)
FEATURES = ['ret', 'vol_21', 'vix', 'term_spread', 'credit_spread']
df = mkt[FEATURES].dropna().copy()

scaler = StandardScaler()
X = scaler.fit_transform(df.values)

print(f"Feature matrix: {X.shape}  |  {df.index.min()} — {df.index.max()}")


# %% [markdown]
# ## 2. Gaussian Mixture Model (GMM)


# %%
gmm_results = {}

for k in range(2, 6):
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=5, random_state=42)
    gmm.fit(X)
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    labels = gmm.predict(X)
    gmm_results[k] = {'model': gmm, 'bic': bic, 'aic': aic, 'labels': labels}
    print(f"  GMM k={k}  BIC={bic:,.1f}  AIC={aic:,.1f}  log-lik={gmm.lower_bound_:.4f}")

best_k_gmm = min(gmm_results, key=lambda k: gmm_results[k]['bic'])
print(f"\nBest GMM K (by BIC): {best_k_gmm}")


# %% [markdown]
# ## 3. Gaussian HMM


# %%
def fit_hmm(X, k, cov_type='diag', n_iter=200, n_trials=5):
    """Fit GaussianHMM n_trials times; return best by log-likelihood."""
    best_score, best_model = -np.inf, None
    for seed in range(n_trials):
        try:
            m = GaussianHMM(
                n_components=k, covariance_type=cov_type,
                n_iter=n_iter, random_state=seed, tol=1e-4
            )
            m.fit(X)
            score = m.score(X)
            if score > best_score:
                best_score, best_model = score, m
        except Exception:
            pass
    return best_model, best_score


def hmm_bic(model, X):
    """Proper BIC for GaussianHMM: counts transition, initial, mean, and covariance parameters."""
    n, d = X.shape
    k = model.n_components
    cov_type = model.covariance_type
    trans_params = k * (k - 1)          # transition matrix (each row sums to 1)
    init_params  = k - 1                 # initial state distribution
    mean_params  = k * d
    if cov_type == 'diag':
        cov_params = k * d
    elif cov_type == 'full':
        cov_params = k * d * (d + 1) // 2
    elif cov_type == 'tied':
        cov_params = d * (d + 1) // 2
    else:  # spherical
        cov_params = k
    n_params = trans_params + init_params + mean_params + cov_params
    total_loglik = model.score(X) * n    # model.score() returns per-obs log-lik
    return -2 * total_loglik + n_params * np.log(n)


hmm_results = {}

for k in range(2, 6):
    for cov_type in ['diag', 'full']:
        model, score = fit_hmm(X, k, cov_type)
        if model is None:
            continue
        labels = model.predict(X)
        bic = hmm_bic(model, X)
        key = f'k{k}_{cov_type}'
        hmm_results[key] = {'model': model, 'loglik': score, 'bic': bic, 'labels': labels, 'k': k}
        print(f"  HMM {key}  log-lik/obs={score:.4f}  BIC={bic:,.1f}")

best_hmm_key = min(hmm_results, key=lambda k: hmm_results[k]['bic'])
print(f"\nBest HMM key (by BIC): {best_hmm_key}")


# %% [markdown]
# ## 4. BIC Selection Plot


# %%
fig, ax = plt.subplots(figsize=(8, 4))

ks = list(gmm_results.keys())
ax.plot(ks, [gmm_results[k]['bic'] for k in ks], 'o-', label='GMM BIC')
ax.plot(ks, [gmm_results[k]['aic'] for k in ks], 's--', label='GMM AIC')

ax.set_xlabel('Number of regimes K')
ax.set_ylabel('Information criterion')
ax.set_title('GMM model selection')
ax.legend()
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_images//bic_selection_price.png', dpi=100)
plt.close("all")


# %% [markdown]
# ## 5. Visualise Regimes


# %%
def plot_regimes(dates, returns, labels, title):
    k = labels.max() + 1
    cum = np.exp(np.cumsum(returns))
    colors = plt.cm.Set1.colors

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Cumulative return
    axes[0].plot(dates, cum, linewidth=0.7, color='black')
    for regime in range(k):
        mask = labels == regime
        axes[0].fill_between(dates, cum.min(), cum, where=mask,
                             alpha=0.25, color=colors[regime], label=f'Regime {regime}')
    axes[0].set_ylabel('S&P 500 (growth of $1)')
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)

    # Regime label time series
    axes[1].scatter(dates, labels, s=1, c=[colors[r] for r in labels], alpha=0.6)
    axes[1].set_ylabel('Regime')
    axes[1].set_yticks(range(k))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace('/', '_')
    plt.savefig(f'/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_images/regimes_{safe_title}.png', dpi=100)
    plt.close("all")


raw_ret = df['ret'].values
dates_arr = df.index.to_numpy()

plot_regimes(dates_arr, raw_ret, gmm_results[best_k_gmm]['labels'],
             f'GMM K={best_k_gmm} (price-only)')
plot_regimes(dates_arr, raw_ret, hmm_results[best_hmm_key]['labels'],
             f'HMM {best_hmm_key} (price-only)')


# %% [markdown]
# ## 6. Per-Regime Statistics


# %%
def regime_stats(df_feat: pd.DataFrame, labels: np.ndarray, name: str) -> pd.DataFrame:
    df_s = df_feat.copy()
    df_s['regime'] = labels
    records = []
    for r, grp in df_s.groupby('regime'):
        ann_ret  = grp['ret'].mean() * 252
        ann_vol  = grp['ret'].std() * np.sqrt(252)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
        avg_vix  = grp['vix'].mean()
        n_days   = len(grp)
        records.append({'model': name, 'regime': r, 'n_days': n_days,
                        'ann_ret': ann_ret, 'ann_vol': ann_vol,
                        'sharpe': sharpe, 'avg_vix': avg_vix})
    return pd.DataFrame(records)


stats_gmm = regime_stats(df, gmm_results[best_k_gmm]['labels'], f'GMM_K{best_k_gmm}')
stats_hmm = regime_stats(df, hmm_results[best_hmm_key]['labels'], f'HMM_{best_hmm_key}')

print("GMM regime stats:")
display(stats_gmm.round(4))
print("\nHMM regime stats:")
display(stats_hmm.round(4))


# %% [markdown]
# ## 7. Save Regime Labels


# %%
regimes = pd.DataFrame(index=df.index)
regimes[f'gmm_k{best_k_gmm}'] = gmm_results[best_k_gmm]['labels']
regimes[f'hmm_{best_hmm_key}'] = hmm_results[best_hmm_key]['labels']

# Also save all K for later comparison
for k in range(2, 6):
    regimes[f'gmm_k{k}'] = gmm_results[k]['labels']

regimes.to_parquet('/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data/regimes_price.parquet')
print("Saved: /content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data/regimes_price.parquet")
regimes.head()
