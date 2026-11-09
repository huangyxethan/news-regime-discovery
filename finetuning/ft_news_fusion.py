# Fine-tuning sweep — Pipeline A: news-only 4-way fusion (no price).
#
# Feature stack per config:
#   DJPR scalar (5d, rolled 35d)  +  Macro scalar (5d, rolled 30d)
#   + DJPR text (varies)          +  Macro text (varies)
#
# Grid: 6 text methods × 3 post-stack dim_red × 5 cluster-K combos = 90 configs
# Outputs:
#   _data/ft_news_fusion_results.parquet
#   _images/ft_news_fusion_top10.png
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
        print(f"[ft_news_fusion] _common.py resolved at: {_cand}")
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
# ## Pipeline A — news-only 4-way fusion
#
# Combines two scalar sources (DJPR + macro sentiment) with two text sources
# (DJPR + macro topic distributions or FinBERT embeddings).  **No price.**
# The text representation is parametrically swept.


# %%
print("=" * 72)
print("ft_news_fusion — Pipeline A (news-only 4-way fusion)")
print("=" * 72)

# Pre-load the two scalar sources (constant across grid)
print("\nLoading scalar sources...")
djpr_scalar  = C.load_djpr_scalar()
macro_scalar = C.load_macro_scalar()
print(f"  DJPR scalar:  {djpr_scalar.shape}  NaN rows: {djpr_scalar.isna().any(axis=1).sum()}")
print(f"  Macro scalar: {macro_scalar.shape}  NaN rows: {macro_scalar.isna().any(axis=1).sum()}")


def assemble_features(text_method: str) -> pd.DataFrame:
    """Build the Pipeline A feature DF for a given text representation."""
    djpr_text  = C.load_text_repr(text_method, "djpr")
    macro_text = C.load_text_repr(text_method, "macro")
    # Join all four on date (inner intersection ensures no NaN-only days)
    X = (djpr_scalar
         .join(macro_scalar, how="inner")
         .join(djpr_text,    how="inner")
         .join(macro_text,   how="inner"))
    return X


# %% [markdown]
# ## Run the grid


# %%
results = C.run_grid("news_fusion", assemble_features)


# %% [markdown]
# ## Quick interpretation cues
#
# After looking at the top 5:
# - Is FinBERT in the top-5?  If yes: FinBERT-as-clustering-input works after all.
# - Which dim_red wins?  If `umap10` consistently → non-linear reduction helps.
# - Which cluster method wins?  HMM is expected for regime tasks; if GMM/KMeans wins,
#   the temporal-structure assumption may not be strong on this fusion.
# - Which K is selected?  Should mostly be 4-5 (Knob 1 logic), but BIC may pick 3
#   for very low-dim representations.


# %%
print("\n=== Top 10 by silhouette (full detail) ===")
top10 = results.sort_values("silhouette", ascending=False).head(10)
print(top10.to_string(index=False))
