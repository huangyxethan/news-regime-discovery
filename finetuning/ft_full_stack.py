# Fine-tuning sweep — Pipeline B: full 5-way stack (price + news fusion).
#
# Feature stack per config:
#   Market price (5d, pre-smoothed in market_features.parquet)
#   + DJPR scalar (5d, rolled 35d)  + Macro scalar (5d, rolled 30d)
#   + DJPR text (varies)            + Macro text (varies)
#
# Grid: same 90-config sweep as ft_news_fusion.py, but with price block added.
# Outputs:
#   _data/ft_full_stack_results.parquet
#   _images/ft_full_stack_top10.png
# Runtime: ~15-25 min on Colab CPU.

# %%
import os
try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass

# Robust path resolution: works in both `python file.py` AND Jupyter/VSCode cell
# execution (where __file__ may not be set or resolves to a kernel runtime path).
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
        print(f"[ft_full_stack] _common.py resolved at: {_cand}")
        break
else:
    raise ImportError(
        f"Cannot find _common.py.  Tried: {_FT_PATH_CANDIDATES}.  "
        "Make sure you are running from the project root or inside finetuning/."
    )
import _common as C

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# ## Pipeline B — full 5-way stack
#
# Adds market price features on top of the 4-way news fusion.  Tests whether
# adding price to the news-only kitchen sink yields a better regime
# representation than either alone.


# %%
print("=" * 72)
print("ft_full_stack — Pipeline B (price + 4-way news fusion)")
print("=" * 72)

# Pre-load all three constant sources
print("\nLoading market + scalar sources...")
price        = C.load_price()
djpr_scalar  = C.load_djpr_scalar()
macro_scalar = C.load_macro_scalar()
print(f"  Price:        {price.shape}        NaN rows: {price.isna().any(axis=1).sum()}")
print(f"  DJPR scalar:  {djpr_scalar.shape}  NaN rows: {djpr_scalar.isna().any(axis=1).sum()}")
print(f"  Macro scalar: {macro_scalar.shape}  NaN rows: {macro_scalar.isna().any(axis=1).sum()}")


def assemble_features(text_method: str) -> pd.DataFrame:
    """Build the Pipeline B feature DF for a given text representation."""
    djpr_text  = C.load_text_repr(text_method, "djpr")
    macro_text = C.load_text_repr(text_method, "macro")
    X = (price
         .join(djpr_scalar,  how="inner")
         .join(macro_scalar, how="inner")
         .join(djpr_text,    how="inner")
         .join(macro_text,   how="inner"))
    return X


# %% [markdown]
# ## Run the grid


# %%
results = C.run_grid("full_stack", assemble_features)


# %% [markdown]
# ## Comparison cues against Pipeline A
#
# - If top-5 silhouette here > top-5 in ft_news_fusion → price adds signal
# - If best mean_duration here > there → price stabilizes regimes
# - If FinBERT placements shift → indicates price interacts with embedding geometry


# %%
print("\n=== Top 10 by silhouette (full detail) ===")
top10 = results.sort_values("silhouette", ascending=False).head(10)
print(top10.to_string(index=False))
