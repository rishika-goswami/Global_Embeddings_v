"""Agglomerative clustering with linkage x distance_threshold sweep (cosine).

Sweeps {average, complete, single} x {0.4..0.9} and picks the combo whose K
is closest to a target K. Also produces dendrograms for the chosen combo
and for each linkage at a fixed threshold (side-by-side comparison).

Original hierarchical1.py is left untouched.

Note: scikit-learn does NOT allow linkage='ward' with metric='cosine' (ward
requires Euclidean), so we exclude it. Single-linkage is included for
completeness even though it tends to produce chaining.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram
from scipy import sparse


DEFAULT_LINKAGES = ["average", "complete", "single"]
# Cosine-distance scales: TF-IDF clusters lie around 0.4-0.95, SBERT around 0.1-0.5.
THRESHOLDS_TFIDF = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
THRESHOLDS_SBERT = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
DEFAULT_THRESHOLDS = THRESHOLDS_TFIDF  # backwards-compat


def _ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def sweep_agglomerative(X, linkages=None, thresholds=None, seed=42, verbose=True):
    """Returns sweep_df, models_map keyed by (linkage, threshold)."""
    linkages = linkages or DEFAULT_LINKAGES
    thresholds = thresholds or DEFAULT_THRESHOLDS
    X_dense = _ensure_dense(X)

    rows = []
    models_map = {}
    for linkage in linkages:
        for thr in thresholds:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=thr,
                metric="cosine",
                linkage=linkage,
                compute_distances=True,
            )
            labels = model.fit_predict(X_dense)
            k = model.n_clusters_
            if k >= 2:
                try:
                    sil = silhouette_score(
                        X_dense, labels, metric="cosine",
                        sample_size=min(2000, X_dense.shape[0]), random_state=seed,
                    )
                except Exception:
                    sil = float("nan")
                try:
                    db = davies_bouldin_score(X_dense, labels)
                except Exception:
                    db = float("nan")
            else:
                sil = float("nan")
                db = float("nan")

            rows.append({
                "linkage": linkage,
                "threshold": thr,
                "n_clusters": int(k),
                "silhouette": sil,
                "davies_bouldin": db,
            })
            models_map[(linkage, thr)] = (model, labels)
            if verbose:
                print(f"[agg] {linkage:<8s} thr={thr}  k={k:4d}  sil={sil}  db={db}")

    return pd.DataFrame(rows), models_map


def pick_combo(sweep_df, target_k, min_silhouette=-1.0):
    """Pick (linkage, threshold) closest to target_k. Soft silhouette floor."""
    candidates = sweep_df[(sweep_df["n_clusters"] >= 2) & (sweep_df["silhouette"].fillna(-2) >= min_silhouette)].copy()
    if len(candidates) == 0:
        candidates = sweep_df[sweep_df["n_clusters"] >= 2].copy()
        if len(candidates) == 0:
            return None
    candidates["k_dist"] = (candidates["n_clusters"] - target_k).abs()
    candidates = candidates.sort_values(by=["k_dist", "silhouette"], ascending=[True, False])
    best = candidates.iloc[0]
    return str(best["linkage"]), float(best["threshold"]), best


def build_linkage_matrix(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current += 1
            else:
                current += counts[child_idx - n_samples]
        counts[i] = current
    return np.column_stack([model.children_, model.distances_, counts]).astype(float)


def plot_dendrogram(model, threshold=None, title="Dendrogram", outfile=None,
                    truncate_p=5, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    plt.title(title)
    lm = build_linkage_matrix(model)
    dendrogram(lm, truncate_mode="level", p=truncate_p)
    plt.xlabel("Cluster size (truncated)")
    plt.ylabel("Cosine distance")
    if threshold is not None:
        plt.axhline(y=threshold, color="r", linestyle="--",
                    label=f"Threshold = {threshold} ({model.n_clusters_} classes)")
        plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_dendrogram_grid(models_by_linkage, threshold, title_prefix, outfile, figsize=(18, 6)):
    """Side-by-side dendrograms across linkages at a fixed threshold."""
    fig, axes = plt.subplots(1, len(models_by_linkage), figsize=figsize)
    if len(models_by_linkage) == 1:
        axes = [axes]
    for ax, (linkage, model) in zip(axes, models_by_linkage.items()):
        plt.sca(ax)
        ax.set_title(f"{title_prefix} — linkage='{linkage}'\n(thr={threshold}, k={model.n_clusters_})")
        lm = build_linkage_matrix(model)
        dendrogram(lm, truncate_mode="level", p=5, ax=ax)
        ax.axhline(y=threshold, color="r", linestyle="--")
        ax.set_xlabel("Cluster size (truncated)")
        ax.set_ylabel("Cosine distance")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    from embeddings_utils import load_sample, build_tfidf, ensure_output_dir, OUTPUT_DIR

    ensure_output_dir()
    sample = load_sample(n=5000)
    X, _ = build_tfidf(sample["cleaned_text"])
    sweep_df, _ = sweep_agglomerative(X)
    sweep_df.to_csv(os.path.join(OUTPUT_DIR, "agglomerative_sweep_tfidf.csv"), index=False)
    print(sweep_df.to_string())
