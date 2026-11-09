# Auto-generated from 04_topic_models.ipynb
# Intended to run in Google Colab with the original Google Drive paths preserved.

import argparse
import os
import sys
import pickle

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

parser = argparse.ArgumentParser(description="LDA + BERTopic topic features.")
parser.add_argument("--enable-macro", action="store_true",
                    help="Also build macro-corpus LDA + BERTopic from "
                         "data/ravenpack_macro_cache_YYYY.parquet (requires "
                         "02d_news_macro.py to have run).")
args, _ = parser.parse_known_args()



# %% [markdown]
# # 04 — Topic Models: LDA + BERTopic


# %%
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import re
import random
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

PARQUET_PATH = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/data/ravenpack_djpr.parquet"
DATA_DIR     = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/_data"

LDA_OUTPUT_PATHS = {
    15: f"{DATA_DIR}/lda_topics_k15_daily.parquet",
    20: f"{DATA_DIR}/lda_topics_k20_daily.parquet",
}
BERTOPIC_OUTPUT_PATH = f"{DATA_DIR}/bertopic_daily.parquet"
HEADLINES_CACHE_PATH = f"{DATA_DIR}/topic_model_headlines_cache.pkl"
SKIP_LDA = all(os.path.exists(path) for path in LDA_OUTPUT_PATHS.values())
SKIP_BERTOPIC = os.path.exists(BERTOPIC_OUTPUT_PATH)

if SKIP_LDA and SKIP_BERTOPIC:
    print("Skipping 04_topic_models.py v1; LDA and BERTopic outputs already exist.")


# %% [markdown]
# ## Part A: LDA — Headline-Level Approach


# %%
# ── Filtering configuration ─────────────────────────────────────────────────

FINANCIAL_STOPWORDS = [
    "inc", "corp", "co", "ltd", "llc", "plc", "group",
    "shares", "stock", "stocks", "percent", "quarter",
    "billion", "million", "year", "new", "says", "said",
    "company", "companies", "market", "markets", "index",
    "report", "reports", "reported", "results", "fiscal",
    "update", "updated", "via", "dj", "ap", "reuters", "wsj",
    "press", "release", "announces", "announced",
    # Law-firm names
    "pomerantz", "bronstein", "gewirtz", "faruqi", "bernstein",
    "levi", "korsinsky", "scott", "rosen", "glancy", "hagens",
    # Generic financial noise
    "shareholder", "alert", "class", "action", "lawsuit",
    "attorney", "claim", "complaint", "allege", "investigation",
    "litigation", "reminds", "deadline", "encouraged", "contact",
    "inst", "holders", "dir", "holder", "vp", "registers",
    "sky", "barron", "mw", "pgr", "shrs", "mkt", "imbalance",
    "chart", "competitors", "compared",
]

NEWS_TYPE_BLACKLIST = {
    "TABULAR-MATERIAL",
    "LEGAL",
    "SHARE-TRANSACTIONS",
}

BOILERPLATE_RE = re.compile(
    r"shareholder.{0,15}alert"
    r"|class.{0,8}action"
    r"|files.{0,8}complaint"
    r"|reminds.{0,15}investor"
    r"|investigation.{0,8}(into|against|regarding)"
    r"|registers.{0,5}of.{0,15}(inc|corp|co\b)"
    r"|inst.{0,4}holders"
    r"|(dir|vp|svp|evp|chmn).{0,30}(registers|holder)"
    r"|(pomerantz|bronstein|faruqi|bernstein|korsinsky|hagens|glancy|rosen)"
    r"|most.{0,8}recent.{0,8}annual"
    r"|quarterly.{0,8}available"
    r"|if.{0,5}applicable"
    r"|suffered.{0,10}loss"
    r"|encouraged.{0,10}contact",
    re.IGNORECASE
)

ALL_STOPWORDS_SET = set(stopwords.words("english")) | set(FINANCIAL_STOPWORDS)


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(s.split())


def passes_filter(headline: str, min_content_words: int = 6) -> bool:
    if BOILERPLATE_RE.search(headline):
        return False
    words = clean_text(headline).split()
    content = [w for w in words if w not in ALL_STOPWORDS_SET and len(w) > 2]
    return len(content) >= min_content_words


# ?? Load + filter headlines year by year ?????????????????????????????????????
if os.path.exists(HEADLINES_CACHE_PATH):
    print(f"Loading cached filtered headlines: {HEADLINES_CACHE_PATH}")
    with open(HEADLINES_CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    day_headlines = cache["day_headlines"]
    dates_lda = cache["dates_lda"]
    all_flat = cache["all_flat"]
    total_headlines = len(all_flat)
    print(f"Loaded cache: {len(dates_lda)} days, {total_headlines:,} total headlines")
else:
    day_headlines: dict = {}

    print("Loading and filtering headlines year by year...")
    for yr in tqdm(range(2000, 2026), desc="Years"):
        batch = pd.read_parquet(
            PARQUET_PATH,
            filters=[
                ("timestamp_utc", ">=", datetime(yr, 1, 1)),
                ("timestamp_utc", "<",  datetime(yr + 1, 1, 1)),
                ("entity_type",   "=",  "COMP"),
                ("country_code",  "=",  "US"),
            ],
            columns=["timestamp_utc", "relevance", "headline", "news_type"],
        )
        n_raw = len(batch)
        batch = batch[batch["relevance"] >= 75].copy()
        batch = batch[~batch["news_type"].isin(NEWS_TYPE_BLACKLIST)]
        batch = batch[batch["headline"].notna()]
        mask = batch["headline"].apply(passes_filter)
        batch = batch[mask]
        tqdm.write(f"  {yr}: {n_raw:>7,} raw  ->  {len(batch):>6,} kept  ({100*len(batch)/max(n_raw,1):.0f}%)")

        batch["date"]  = pd.to_datetime(batch["timestamp_utc"]).dt.normalize()
        batch["clean"] = batch["headline"].apply(clean_text)

        for date, grp in batch.groupby("date"):
            texts = grp["clean"].tolist()
            if texts:
                if date in day_headlines:
                    day_headlines[date].extend(texts)
                else:
                    day_headlines[date] = texts

    dates_lda = sorted(day_headlines.keys())
    all_flat = [(d, h) for d, hs in day_headlines.items() for h in hs]
    total_headlines = len(all_flat)
    print(f"\nAfter filtering: {len(dates_lda)} days, {total_headlines:,} total headlines")
    print(f"Avg headlines/day: {total_headlines / max(len(dates_lda), 1):.0f}")
    print(f"Total individual headlines: {len(all_flat):,}")

    with open(HEADLINES_CACHE_PATH, "wb") as f:
        pickle.dump({
            "day_headlines": day_headlines,
            "dates_lda": dates_lda,
            "all_flat": all_flat,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved filtered headline cache: {HEADLINES_CACHE_PATH}")


# %% [markdown]
# ### A2. Fit LDA on Sampled Individual Headlines (K=15 and K=20)


# %%
if SKIP_LDA:
    print("Skipping LDA topic generation; outputs already exist:")
    for path in LDA_OUTPUT_PATHS.values():
        print(f"  {path}")
    lda_daily = pd.read_parquet(LDA_OUTPUT_PATHS[15])
else:
    # ── Sample headlines for LDA training ──────────────────────────────────────
    # n_jobs=1: avoids joblib writing mmap files to /tmp (tmpfs ~2GB on Colab).
    # SAMPLE_SIZE=300k and max_features=8000 are safe because single-threaded
    # LDA writes nothing to disk — the disk error was solely from n_jobs=-1.
    SAMPLE_SIZE = 300_000


    random.seed(42)
    sample_texts = [h for _, h in random.sample(all_flat, min(SAMPLE_SIZE, len(all_flat)))]
    print(f"Training sample size: {len(sample_texts):,}")

    # ── Build vocabulary ─────────────────────────────────────────────────────────
    stop_words_list = list(stopwords.words("english")) + FINANCIAL_STOPWORDS

    vectorizer = CountVectorizer(
        max_df=0.80,
        min_df=15,
        stop_words=stop_words_list,
        max_features=8000,
        ngram_range=(1, 2),
    )
    dtm_sample = vectorizer.fit_transform(sample_texts)
    vocab = vectorizer.get_feature_names_out()
    print(f"DTM shape: {dtm_sample.shape}  vocab: {len(vocab)}")

    # ── Fit LDA for K=15 and K=20 ────────────────────────────────────────────────
    # n_jobs=1 is the only change from the original; data size is unchanged.
    # Expected runtime: ~10-20 min per K on Colab CPU.
    lda_models = {}

    for K in [15, 20]:
        print(f"\nFitting LDA K={K}  (single-threaded, ~10-20 min per K)...")
        lda = LatentDirichletAllocation(
            n_components=K,
            max_iter=20,
            learning_method="online",
            random_state=42,
            n_jobs=1,
            verbose=1,
        )
        lda.fit(dtm_sample)
        perplexity = lda.perplexity(dtm_sample)
        lda_models[K] = {"model": lda, "perplexity": perplexity}
        print(f"  K={K}  perplexity={perplexity:,.1f}")

    print("\nDone fitting LDA models.")


    # %% [markdown]
    # ### A3. Top Words per Topic


    # %%
    def top_words(lda, vocab, n=10):
        rows = []
        for t_idx, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[-n:][::-1]
            rows.append({"topic": t_idx, "words": ", ".join(vocab[top_idx])})
        return pd.DataFrame(rows)


    # Preview top words for each K to pick the better model
    for K in lda_models:
        best_perp_k = lda_models[K]["perplexity"]
        print(f"\n--- K={K}  perplexity={best_perp_k:,.1f} ---")
        display(top_words(lda_models[K]["model"], vocab))


    # %% [markdown]
    # ### A4. Select Best K, Aggregate to Daily, and Save


    # %%
    # Select best K by perplexity
    K_selected = min(lda_models, key=lambda k: lda_models[k]["perplexity"])
    best_perp = lda_models[K_selected]["perplexity"]
    print(f"Selected K={K_selected}  (perplexity={best_perp:,.1f})")
    print("Top words per topic for K_selected:")
    display(top_words(lda_models[K_selected]["model"], vocab))


    def aggregate_to_daily(lda_model, K):
        """Transform each day's headlines through LDA and mean-pool to daily vector."""
        topic_cols = [f"topic_{i}" for i in range(K)]
        records = {}
        for date in tqdm(dates_lda, desc=f"Aggregating K={K}"):
            headlines = day_headlines[date]
            dtm_day = vectorizer.transform(headlines)
            topic_dists = lda_model.transform(dtm_day)   # (n_headlines, K)
            records[date] = topic_dists.mean(axis=0)
        df = pd.DataFrame.from_dict(records, orient="index", columns=topic_cols)
        df.index.name = "date"
        df.sort_index(inplace=True)
        return df


    # Save K=15 and K=20
    for K in [15, 20]:
        df_lda = aggregate_to_daily(lda_models[K]["model"], K)
        path = f"{DATA_DIR}/lda_topics_k{K}_daily.parquet"
        df_lda.to_parquet(path)
        print(f"Saved: lda_topics_k{K}_daily.parquet  shape={df_lda.shape}")

    lda_daily = pd.read_parquet(f"{DATA_DIR}/lda_topics_k{K_selected}_daily.parquet")
    print(f"Canonical LDA file: lda_topics_k{K_selected}_daily.parquet")
    lda_daily.head(3)


# %% [markdown]
# ---


# %%
if SKIP_BERTOPIC:
    print(f"Skipping BERTopic generation; output already exists: {BERTOPIC_OUTPUT_PATH}")
else:
    import torch
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Sample headlines for BERTopic training ───────────────────────────────────
    BERT_SAMPLE = 60_000
    random.seed(0)
    bert_sample = random.sample(all_flat, min(BERT_SAMPLE, len(all_flat)))
    bert_texts  = [h for _, h in bert_sample]
    print(f"BERTopic training sample: {len(bert_texts):,} individual headlines")

    # ── Embed with sentence-transformer (proper mean pooling, not CLS token) ─────
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print("Embedding headlines...")
    bert_embeddings = st_model.encode(
        bert_texts,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"Embeddings shape: {bert_embeddings.shape}")


    # %%
    # ── Fit BERTopic on individual headlines ────────────────────────────────────
    umap_model = UMAP(
        n_components=10,
        n_neighbors=30,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        min_topic_size=200,
        nr_topics=15,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(bert_texts, embeddings=bert_embeddings)

    n_topics_found = len(set(topics) - {-1})
    n_outliers     = sum(1 for t in topics if t == -1)
    print(f"Topics: {n_topics_found}  |  Outliers: {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

    print("\nBERTopic topic info:")
    display(topic_model.get_topic_info().head(20))


    # %%
    # ── Aggregate BERTopic to daily level via approximate_distribution ───────────
    # Uses c-TF-IDF (no GPU needed), processes each day's sampled headlines.
    topic_ids = sorted(set(topics) - {-1})
    bert_cols  = [f"btopic_{tid}" for tid in topic_ids]

    daily_bert_records = {}
    for date in tqdm(dates_lda, desc="Daily BERTopic aggregation"):
        day_texts_sample = day_headlines[date]
        if len(day_texts_sample) > 50:
            day_texts_sample = random.sample(day_texts_sample, 50)
        topic_distr, _ = topic_model.approximate_distribution(
            day_texts_sample, use_embedding_model=False
        )
        # approximate_distribution returns columns in ascending topic-ID order:
        # [-1, 0, 1, 2, ...] so outlier topic is at index 0 when present.
        n_cols = topic_distr.shape[1]
        if n_cols > len(topic_ids):
            topic_distr = topic_distr[:, 1:]          # drop outlier col at index 0
        if topic_distr.shape[1] > len(topic_ids):
            topic_distr = topic_distr[:, :len(topic_ids)]
        elif topic_distr.shape[1] < len(topic_ids):
            pad = np.zeros((topic_distr.shape[0], len(topic_ids) - topic_distr.shape[1]))
            topic_distr = np.hstack([topic_distr, pad])
        daily_bert_records[date] = topic_distr.mean(axis=0)

    df_bert = pd.DataFrame.from_dict(daily_bert_records, orient="index", columns=bert_cols)
    df_bert.index.name = "date"
    df_bert.sort_index(inplace=True)
    print(f"Daily BERTopic shape: {df_bert.shape}")
    df_bert.head(3)


    # %%
    df_bert.to_parquet(f"{DATA_DIR}/bertopic_daily.parquet")
    print(f"Saved: bertopic_daily.parquet  shape={df_bert.shape}")

    # Save model for nb08 semantic characterization
    try:
        import os
        model_dir = f"{DATA_DIR}/bertopic_model"
        os.makedirs(model_dir, exist_ok=True)
        topic_model.save(model_dir, serialization="safetensors",
                         save_ctfidf=True, save_embedding_model=False)
        print(f"Saved: bertopic_model/")
    except Exception as e:
        print(f"Model save failed (non-fatal): {e}")

    # Quick sanity: print top words per topic
    print("\nFinal topic keywords:")
    for tid in topic_ids[:12]:
        words = [w for w, _ in topic_model.get_topic(tid)[:8]]
        print(f"  Topic {tid:2d}: {words}")


# ── § Optional: Macro LDA + BERTopic (DJ Global Macro corpus) ────────────────
# Reads pre-filtered yearly caches written by 02d_news_macro.py.  Reuses the
# same STOPWORDS / BOILERPLATE_RE / passes_filter / clean_text from this file.
# Hyperparameters scaled down (macro corpus ~1/5 the size of DJPR):
#   SAMPLE_SIZE: 300k -> 100k    min_df: 15 -> 5      max_features: 8000 -> 5000
#   BERT_SAMPLE: 60k -> 30k      min_topic_size: 200 -> 100
LDA_MACRO_OUTPUT_PATHS = {
    15: f"{DATA_DIR}/lda_topics_macro_k15_daily.parquet",
    20: f"{DATA_DIR}/lda_topics_macro_k20_daily.parquet",
}
BERTOPIC_MACRO_OUTPUT_PATH = f"{DATA_DIR}/bertopic_macro_daily.parquet"
HEADLINES_CACHE_MACRO_PATH = f"{DATA_DIR}/topic_model_headlines_macro_cache.pkl"
MACRO_CACHE_DIR = "/content/drive/MyDrive/TTIC31220-Spring26/newsheadline/data"

if args.enable_macro:
    SKIP_LDA_MACRO = all(os.path.exists(p) for p in LDA_MACRO_OUTPUT_PATHS.values())
    SKIP_BERTOPIC_MACRO = os.path.exists(BERTOPIC_MACRO_OUTPUT_PATH)

    if SKIP_LDA_MACRO and SKIP_BERTOPIC_MACRO:
        print("\nSkipping macro topic models; outputs already exist.")
    else:
        print("\n=== Macro LDA + BERTopic (DJ Global Macro corpus) ===")

        # ── Load + filter macro headlines from yearly caches ────────────────
        if os.path.exists(HEADLINES_CACHE_MACRO_PATH):
            print(f"Loading cached macro headlines: {HEADLINES_CACHE_MACRO_PATH}")
            with open(HEADLINES_CACHE_MACRO_PATH, "rb") as f:
                cache_m = pickle.load(f)
            day_headlines_macro = cache_m["day_headlines"]
            dates_macro = cache_m["dates_lda"]
            all_flat_macro = cache_m["all_flat"]
            print(f"  cache: {len(dates_macro)} days, {len(all_flat_macro):,} headlines")
        else:
            day_headlines_macro: dict = {}
            print("Loading macro yearly caches (2000-2026)...")
            for yr in tqdm(range(2000, 2027), desc="Macro years"):
                cache_yr = f"{MACRO_CACHE_DIR}/ravenpack_macro_cache_{yr}.parquet"
                if not os.path.exists(cache_yr):
                    continue
                batch = pd.read_parquet(cache_yr)
                # 02d already filtered entity/country/relevance/news_type, so we
                # only need the textual passes_filter (boilerplate removal).
                batch = batch[batch["headline"].notna()]
                n_raw = len(batch)
                mask = batch["headline"].apply(passes_filter)
                batch = batch[mask].copy()
                tqdm.write(f"  {yr}: {n_raw:>7,} raw  ->  {len(batch):>6,} kept "
                           f"({100*len(batch)/max(n_raw,1):.0f}%)")
                batch["date"] = pd.to_datetime(batch["timestamp_utc"]).dt.normalize()
                batch["clean"] = batch["headline"].apply(clean_text)
                for date, grp in batch.groupby("date"):
                    texts = grp["clean"].tolist()
                    if texts:
                        if date in day_headlines_macro:
                            day_headlines_macro[date].extend(texts)
                        else:
                            day_headlines_macro[date] = texts
            dates_macro = sorted(day_headlines_macro.keys())
            all_flat_macro = [(d, h) for d, hs in day_headlines_macro.items() for h in hs]
            print(f"\nMacro filtered: {len(dates_macro)} days, "
                  f"{len(all_flat_macro):,} headlines, "
                  f"avg {len(all_flat_macro) / max(len(dates_macro), 1):.0f}/day")
            with open(HEADLINES_CACHE_MACRO_PATH, "wb") as f:
                pickle.dump({
                    "day_headlines": day_headlines_macro,
                    "dates_lda": dates_macro,
                    "all_flat": all_flat_macro,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved cache: {HEADLINES_CACHE_MACRO_PATH}")

        # ── Macro LDA (K=15, K=20) ─────────────────────────────────────────
        if not SKIP_LDA_MACRO:
            SAMPLE_SIZE_MACRO = 100_000
            random.seed(42)
            sample_texts_macro = [
                h for _, h in random.sample(
                    all_flat_macro, min(SAMPLE_SIZE_MACRO, len(all_flat_macro))
                )
            ]
            print(f"\nMacro LDA training sample: {len(sample_texts_macro):,}")

            vectorizer_macro = CountVectorizer(
                max_df=0.80,
                min_df=5,
                stop_words=list(stopwords.words("english")) + FINANCIAL_STOPWORDS,
                max_features=5000,
                ngram_range=(1, 2),
            )
            dtm_sample_macro = vectorizer_macro.fit_transform(sample_texts_macro)
            vocab_macro = vectorizer_macro.get_feature_names_out()
            print(f"Macro DTM: {dtm_sample_macro.shape}  vocab: {len(vocab_macro)}")

            lda_models_macro = {}
            for K in [15, 20]:
                print(f"\nFitting macro LDA K={K} ...")
                lda_m = LatentDirichletAllocation(
                    n_components=K,
                    max_iter=20,
                    learning_method="online",
                    random_state=42,
                    n_jobs=1,
                    verbose=1,
                )
                lda_m.fit(dtm_sample_macro)
                perp_m = lda_m.perplexity(dtm_sample_macro)
                lda_models_macro[K] = lda_m
                print(f"  K={K}  perplexity={perp_m:,.1f}")

            # Print top words per topic (sanity check — should be macro vocab)
            for K in lda_models_macro:
                print(f"\n--- Macro LDA K={K} top words ---")
                for t_idx, topic in enumerate(lda_models_macro[K].components_):
                    top_idx = topic.argsort()[-10:][::-1]
                    print(f"  Topic {t_idx:2d}: {', '.join(vocab_macro[top_idx])}")

            # Aggregate to daily for both Ks
            for K in [15, 20]:
                topic_cols = [f"topic_{i}" for i in range(K)]
                records = {}
                for date in tqdm(dates_macro, desc=f"Macro aggregate K={K}"):
                    dtm_day = vectorizer_macro.transform(day_headlines_macro[date])
                    records[date] = lda_models_macro[K].transform(dtm_day).mean(axis=0)
                df_lda_m = pd.DataFrame.from_dict(records, orient="index", columns=topic_cols)
                df_lda_m.index.name = "date"
                df_lda_m.sort_index(inplace=True)
                df_lda_m.to_parquet(LDA_MACRO_OUTPUT_PATHS[K])
                print(f"Saved: {LDA_MACRO_OUTPUT_PATHS[K]}  shape={df_lda_m.shape}")
        else:
            print("Macro LDA outputs exist; skipping LDA training.")

        # ── Macro BERTopic ─────────────────────────────────────────────────
        if not SKIP_BERTOPIC_MACRO:
            import torch
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from umap import UMAP

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\nMacro BERTopic device: {device}")

            BERT_SAMPLE_MACRO = 30_000
            random.seed(0)
            bert_sample_macro = random.sample(
                all_flat_macro, min(BERT_SAMPLE_MACRO, len(all_flat_macro))
            )
            bert_texts_macro = [h for _, h in bert_sample_macro]
            print(f"Macro BERTopic sample: {len(bert_texts_macro):,}")

            st_model_m = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            bert_embeddings_macro = st_model_m.encode(
                bert_texts_macro,
                batch_size=512,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            print(f"Macro BERTopic embeddings: {bert_embeddings_macro.shape}")

            umap_model_m = UMAP(
                n_components=10,
                n_neighbors=30,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            topic_model_macro = BERTopic(
                umap_model=umap_model_m,
                min_topic_size=100,
                nr_topics=15,
                calculate_probabilities=True,
                verbose=True,
            )
            topics_m, _ = topic_model_macro.fit_transform(
                bert_texts_macro, embeddings=bert_embeddings_macro
            )

            topic_ids_m = sorted(set(topics_m) - {-1})
            bert_cols_m = [f"btopic_{tid}" for tid in topic_ids_m]
            print(f"Macro BERTopic: {len(topic_ids_m)} topics found")

            daily_bert_records_m = {}
            for date in tqdm(dates_macro, desc="Macro daily BERTopic"):
                day_texts_sample = day_headlines_macro[date]
                if len(day_texts_sample) > 50:
                    day_texts_sample = random.sample(day_texts_sample, 50)
                topic_distr, _ = topic_model_macro.approximate_distribution(
                    day_texts_sample, use_embedding_model=False
                )
                n_cols = topic_distr.shape[1]
                if n_cols > len(topic_ids_m):
                    topic_distr = topic_distr[:, 1:]   # drop outlier col
                if topic_distr.shape[1] > len(topic_ids_m):
                    topic_distr = topic_distr[:, :len(topic_ids_m)]
                elif topic_distr.shape[1] < len(topic_ids_m):
                    pad = np.zeros(
                        (topic_distr.shape[0], len(topic_ids_m) - topic_distr.shape[1])
                    )
                    topic_distr = np.hstack([topic_distr, pad])
                daily_bert_records_m[date] = topic_distr.mean(axis=0)

            df_bert_m = pd.DataFrame.from_dict(
                daily_bert_records_m, orient="index", columns=bert_cols_m
            )
            df_bert_m.index.name = "date"
            df_bert_m.sort_index(inplace=True)
            df_bert_m.to_parquet(BERTOPIC_MACRO_OUTPUT_PATH)
            print(f"Saved: {BERTOPIC_MACRO_OUTPUT_PATH}  shape={df_bert_m.shape}")

            # Save model for nb08 use
            try:
                macro_model_dir = f"{DATA_DIR}/bertopic_macro_model"
                os.makedirs(macro_model_dir, exist_ok=True)
                topic_model_macro.save(
                    macro_model_dir,
                    serialization="safetensors",
                    save_ctfidf=True,
                    save_embedding_model=False,
                )
                print(f"Saved: bertopic_macro_model/")
            except Exception as e:
                print(f"Macro BERTopic model save failed (non-fatal): {e}")

            # Quick sanity: print top words per topic
            print("\nMacro BERTopic top words per topic:")
            for tid in topic_ids_m[:12]:
                words = [w for w, _ in topic_model_macro.get_topic(tid)[:8]]
                print(f"  Topic {tid:2d}: {words}")
        else:
            print("Macro BERTopic output exists; skipping.")
