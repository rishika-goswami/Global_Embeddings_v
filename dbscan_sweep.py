"""DBSCAN with parameter sweep (cosine metric).

Sweeps eps x min_samples and reports n_clusters / n_noise / silhouette per
combination, then picks the combination whose K is closest to a target K
(passed in by the orchestrator) under a noise ceiling.

Original dbscan1.py is left untouched.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


# TF-IDF and SBERT live on different cosine-distance scales. SBERT embeddings
# are dense and semantically clustered, so neighborhood distances are tiny
# (~0.05–0.4); TF-IDF on short text has cosine distances ~0.5–1.0. We need
# different eps grids per representation.
EPS_GRID_TFIDF = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
EPS_GRID_SBERT = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
DEFAULT_EPS = EPS_GRID_TFIDF  # backwards-compat
DEFAULT_MIN_SAMPLES = [3, 5, 8, 10, 15]


def _to_dense_if_small(X, max_dense_elems=20_000_000):
    if sparse.issparse(X) and X.shape[0] * X.shape[1] <= max_dense_elems:
        return X.toarray()
    return X


def k_distance_curve(X, k=5):
    """Sorted k-th NN distance — for picking eps via the elbow heuristic."""
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    distances, _ = nn.kneighbors(X)
    kth = np.sort(distances[:, k - 1])
    return kth


def sweep_dbscan(X, eps_grid=None, min_samples_grid=None, seed=42, verbose=True):
    """Run all combos. Returns DataFrame and dict of {(eps, min_samples): labels}."""
    eps_grid = eps_grid or DEFAULT_EPS
    min_samples_grid = min_samples_grid or DEFAULT_MIN_SAMPLES
    X_for_sil = _to_dense_if_small(X)

    rows = []
    labels_map = {}
    for eps in eps_grid:
        for ms in min_samples_grid:
            db = DBSCAN(eps=eps, min_samples=ms, metric="cosine").fit(X)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            n = len(labels)

            if n_clusters >= 2 and (labels != -1).sum() >= 2:
                mask = labels != -1
                try:
                    sil = silhouette_score(
                        X_for_sil[mask] if not sparse.issparse(X_for_sil) else X_for_sil[mask],
                        labels[mask],
                        metric="cosine",
                        sample_size=min(2000, mask.sum()),
                        random_state=seed,
                    )
                except Exception:
                    sil = float("nan")
            else:
                sil = float("nan")

            rows.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": round(100 * n_noise / n, 2),
                "silhouette": sil,
            })
            labels_map[(eps, ms)] = labels
            if verbose:
                print(f"[dbscan] eps={eps} ms={ms:2d} -> k={n_clusters:3d}  noise={n_noise:5d} ({100*n_noise/n:.1f}%)  sil={sil}")

    return pd.DataFrame(rows), labels_map


def pick_combo(sweep_df, target_k, max_noise_pct=25.0):
    """Pick the (eps, min_samples) closest to target_k with noise <= max_noise_pct.

    Returns (eps, min_samples, row) or None if no combo qualifies.
    Tie-breaker: lower noise pct, then higher silhouette.
    """
    candidates = sweep_df[(sweep_df["n_clusters"] >= 2) & (sweep_df["noise_pct"] <= max_noise_pct)].copy()
    if len(candidates) == 0:
        # Fall back: ignore noise ceiling
        candidates = sweep_df[sweep_df["n_clusters"] >= 2].copy()
        if len(candidates) == 0:
            return None
    candidates["k_dist"] = (candidates["n_clusters"] - target_k).abs()
    candidates = candidates.sort_values(
        by=["k_dist", "noise_pct", "silhouette"], ascending=[True, True, False]
    )
    best = candidates.iloc[0]
    return float(best["eps"]), int(best["min_samples"]), best


if __name__ == "__main__":
    from embeddings_utils import load_sample, build_tfidf, ensure_output_dir, OUTPUT_DIR
    import os

    ensure_output_dir()
    sample = load_sample(n=5000)
    X, _ = build_tfidf(sample["cleaned_text"])
    sweep_df, _ = sweep_dbscan(X)
    sweep_df.to_csv(os.path.join(OUTPUT_DIR, "dbscan_sweep_tfidf.csv"), index=False)
    print(sweep_df.to_string())
