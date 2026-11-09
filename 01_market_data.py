# Auto-generated from 01_market_data.ipynb
# Intended to run in Google Colab with the original Google Drive paths preserved.

import os
import sys

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


def load_env_file(path: str) -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
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

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "").strip()
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD", "").strip()
WRDS_USERNAME_PLACEHOLDER = {"", "your_wrds_username_here"}
WRDS_PASSWORD_PLACEHOLDER = {"", "your_wrds_password_here"}

if WRDS_USERNAME not in WRDS_USERNAME_PLACEHOLDER:
    os.environ.setdefault("PGUSER", WRDS_USERNAME)
if WRDS_PASSWORD not in WRDS_PASSWORD_PLACEHOLDER:
    os.environ.setdefault("PGPASSWORD", WRDS_PASSWORD)

OUTPUT_PATH = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data/market_features.parquet"
if os.path.exists(OUTPUT_PATH):
    print(f"Skipping 01_market_data.py; output already exists: {OUTPUT_PATH}")
    sys.exit(0)


# %% [markdown]
# # 01 — Market Features


# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

START = '2000-01-01'
END   = '2025-12-31'


# %% [markdown]
# ## 1. S&P 500 Returns — WRDS / CRSP


# %%
import wrds

wrds_kwargs = {}
if WRDS_USERNAME not in WRDS_USERNAME_PLACEHOLDER:
    wrds_kwargs["wrds_username"] = WRDS_USERNAME

conn = wrds.Connection(**wrds_kwargs)   # uses .env/PGPASSWORD, ~/.pgpass, or prompts

crsp = conn.raw_sql(
    f"""
    SELECT date, sprtrn, vwretd
    FROM crsp.dsi
    WHERE date >= '{START}' AND date <= '{END}'
    ORDER BY date
    """,
    date_cols=['date']
)
conn.close()

crsp = crsp.set_index('date')
crsp['sprtrn'] = crsp['sprtrn'].astype(float)
crsp['vwretd'] = crsp['vwretd'].astype(float)
crsp['ret']    = np.log1p(crsp['sprtrn'])   # log return

print(f"CRSP rows: {len(crsp):,}  |  {crsp.index.min()} — {crsp.index.max()}")
crsp[['ret', 'vwretd']].head()


# %% [markdown]
# ## 2. Macro Variables — FRED


# %%
import pandas_datareader.data as web
from datetime import date

fred_map = {
    'vix':  'VIXCLS',
    'gs10': 'DGS10',
    'gs2':  'DGS2',
    'baa':  'DBAA',
    'aaa':  'DAAA',
}

fred_frames = {}
for name, code in fred_map.items():
    s = web.DataReader(code, 'fred', date(2000, 1, 1), date(2025, 12, 31))
    s.index = pd.to_datetime(s.index)
    fred_frames[name] = s[code].replace('.', np.nan).astype(float)

fred = pd.DataFrame(fred_frames)
fred = fred.ffill()   # fill weekend/holiday gaps forward

print(f"FRED rows: {len(fred):,}  |  NaNs: {fred.isnull().sum().to_dict()}")
fred.tail()


# %% [markdown]
# ## 3. Combine + Derived Features


# %%
df = crsp[['ret', 'vwretd']].join(fred, how='left')

df['vol_21']       = df['ret'].rolling(21).std() * np.sqrt(252)
df['term_spread']  = df['gs10'] - df['gs2']
df['credit_spread'] = df['baa'] - df['aaa']

df = df.drop(columns=['gs10', 'gs2', 'baa', 'aaa'])
df.index.name = 'date'

print(f"Final shape: {df.shape}")
print(f"Date range: {df.index.min()} — {df.index.max()}")
print(f"NaN counts:\n{df.isnull().sum()}")
df.describe()


# %% [markdown]
# ## 4. Quick Sanity Check


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

events = [
    ('Dot-com peak', '2000-03-10'),
    ('9/11',         '2001-09-11'),
    ('GFC',          '2008-09-15'),
    ('COVID',        '2020-03-16'),
    ('Rate hikes',   '2022-03-16'),
]

# Cumulative return
cum_ret = df['ret'].cumsum().apply(np.exp)
axes[0].plot(df.index, cum_ret, linewidth=0.8)
axes[0].set_ylabel('S&P 500 (growth of $1)')

# VIX
axes[1].plot(df.index, df['vix'], linewidth=0.6, color='orange')
axes[1].set_ylabel('VIX')

# Spreads
axes[2].plot(df.index, df['term_spread'],   linewidth=0.7, label='Term (10Y-2Y)')
axes[2].plot(df.index, df['credit_spread'], linewidth=0.7, label='Credit (Baa-Aaa)')
axes[2].axhline(0, color='k', linewidth=0.5, linestyle='--')
axes[2].set_ylabel('Spread (%)')
axes[2].legend(fontsize=8)

for ax in axes:
    for label, dt in events:
        ax.axvline(pd.Timestamp(dt), color='red', alpha=0.3, linewidth=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_images/market_features_check.png', dpi=100)
plt.close("all")


# %% [markdown]
# ## 5. Save


# %%
df.to_parquet('/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data/market_features.parquet')
print("Saved: /content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data/market_features.parquet")
print(f"Columns: {list(df.columns)}")
df.tail()
