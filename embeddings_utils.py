"""Shared utilities for clustering: load sample, build TF-IDF + SBERT embeddings,
extract cluster keywords. Used by ymeans2.py, dbscan_sweep.py, hierarchical_sweep.py,
and convergence_analysis.py so all three methods cluster the SAME rows under the
SAME representation, making cross-method comparison valid.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SEED = 42
OUTPUT_DIR = "outputs"
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_sample(csv_path="preprocessed_grievances.csv", n=5000, seed=SEED):
    """Load deterministic sample of n rows. Same sample for all methods.

    Adds a `row_idx` column tracking the row's position in the source CSV (after
    NA/empty drop), so we can join against llm_labels.csv (whose row_idx was
    assigned in llm_annotate.py via the same logic).
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    df["row_idx"] = df.index.astype(int)
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return sample


def build_tfidf(texts, max_features=1000, ngram_range=(1, 2)):
    """TF-IDF + L2 normalize. With L2-normalized vectors, Euclidean distance is
    monotonic with cosine distance, so KMeans (Euclidean) becomes spherical KMeans.
    Returns (X_normalized, vectorizer)."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    X_norm = normalize(X, norm="l2", axis=1)
    return X_norm, vectorizer


def build_sbert(texts, model_name=SBERT_MODEL_NAME, cache_path=None, batch_size=64):
    """Multilingual sentence-transformers embeddings, L2-normalized. Cached to disk."""
    from sentence_transformers import SentenceTransformer

    ensure_output_dir()
    if cache_path is None:
        cache_path = os.path.join(OUTPUT_DIR, f"sbert_cache_{len(texts)}.npy")

    if os.path.exists(cache_path):
        print(f"[sbert] loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"[sbert] encoding {len(texts)} texts with {model_name} ...")
    print("[sbert] (first run downloads ~470 MB of model weights)")
    model = SentenceTransformer(model_name)
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, emb)
    print(f"[sbert] saved to {cache_path}")
    return emb


def cluster_keywords(texts, labels, vectorizer=None, n_words=8):
    """Top n_words TF-IDF keywords per cluster. If no vectorizer supplied, fits one."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        vectorizer.fit(texts)
    df = pd.DataFrame({"text": list(texts), "label": list(labels)})
    out = {}
    for cid in sorted(df["label"].unique()):
        if cid == -1:
            continue
        joined = " ".join(df[df["label"] == cid]["text"].tolist())
        if not joined.strip():
            out[int(cid)] = []
            continue
        vec = vectorizer.transform([joined]).toarray()[0]
        words = vectorizer.get_feature_names_out()
        top_idx = vec.argsort()[-n_words:][::-1]
        out[int(cid)] = [words[i] for i in top_idx if vec[i] > 0]
    return out


def cluster_size_distribution(labels):
    """Returns dict {cluster_id: count} including noise (-1) if present."""
    s = pd.Series(labels).value_counts().sort_index()
    return s.to_dict()
