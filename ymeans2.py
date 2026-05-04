"""Y-Means v2 — cosine-aware autonomous K selection.

Differences from the original ymeans.py:
- L2-normalized inputs make plain KMeans equivalent to spherical (cosine) KMeans.
- Standard BIC: BIC = n * log(SSE/n) + k * log(n), instead of the heuristic in v1
  that was dominated by the dimensionality penalty.
- Reports silhouette (cosine) at every K alongside BIC; the chosen K balances
  both signals.
- Sweeps the full [k_min, k_max] range and saves the curve, instead of stopping
  at the first BIC uptick.

Returns the fitted KMeans model plus the per-K sweep so the orchestrator can plot it.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import sparse


def _to_dense_if_small(X, max_dense_elems=20_000_000):
    """Silhouette on cosine needs an array; densify only if it'll fit."""
    if sparse.issparse(X):
        if X.shape[0] * X.shape[1] <= max_dense_elems:
            return X.toarray()
        return X
    return X


def _bic(kmeans, X):
    """Standard BIC: n*log(SSE/n) + k*log(n). Lower is better."""
    n = X.shape[0]
    k = kmeans.n_clusters
    sse = kmeans.inertia_
    if sse <= 0 or n <= 1:
        return np.inf
    return n * np.log(sse / n) + k * np.log(n)


def _find_knee(ks, values):
    """Kneedle-style knee finder for a (mostly) monotonic decreasing curve.

    Normalizes x and y to [0,1], rotates so an ideal monotonic decrease lies
    on the diagonal, returns the K at maximum perpendicular distance — i.e.
    the point of maximum diminishing returns.
    """
    ks = np.asarray(ks, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(ks) < 3:
        return int(ks[np.argmin(v)])
    x = (ks - ks.min()) / (ks.max() - ks.min())
    # Reverse y so it's increasing for distance calc
    y = (v - v.min()) / (v.max() - v.min())
    # Difference from diagonal y=x — knee is the point of max distance below the diag (concave down)
    diff = y - x
    return int(ks[np.argmin(diff)])


def _interior_silhouette_peak(sweep, k_min_floor=3):
    """Silhouette peak excluding K=2 (which is often degenerate for high-dim data).

    Returns None if no valid peak.
    """
    valid = sweep.dropna(subset=["silhouette"])
    valid = valid[valid["k"] >= k_min_floor]
    if len(valid) == 0:
        return None
    return int(valid.loc[valid["silhouette"].idxmax(), "k"])


def run_ymeans2(X, k_min=2, k_max=15, seed=42, verbose=True, n_init=3):
    """Sweep K, return (best_model, sweep_df, chosen_k).

    K selection (Y-Means autonomous K, adapted for high-dim text):
      1. If BIC has an INTERIOR minimum (not at k_min or k_max), use it — that
         is a true Y-Means signal that adding more clusters hurts.
      2. Otherwise BIC is monotonic — find its knee (Kneedle algorithm).
      3. Cross-check against silhouette's interior peak (k>=3, excluding the
         degenerate K=2 trivial split). If both agree (within ±2), great.
         If they disagree, prefer the BIC-based answer (BIC is theoretically
         grounded; silhouette can favor coarse splits in high-dim cosine space).
    """
    rows = []
    models = {}
    X_for_sil = _to_dense_if_small(X)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X)
        bic = _bic(km, X)
        try:
            sil = silhouette_score(X_for_sil, km.labels_, metric="cosine",
                                   sample_size=min(2000, X.shape[0]), random_state=seed)
        except Exception as e:
            sil = float("nan")
            if verbose:
                print(f"[ymeans2] silhouette failed at k={k}: {e}")
        rows.append({"k": k, "bic": bic, "silhouette": sil, "inertia": km.inertia_})
        models[k] = km
        if verbose:
            print(f"[ymeans2] k={k:2d}  BIC={bic:.2f}  sil={sil:.4f}")

    sweep = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # Step 1: interior BIC minimum?
    bic_argmin_pos = int(sweep["bic"].idxmin())
    is_interior = (bic_argmin_pos != 0) and (bic_argmin_pos != len(sweep) - 1)
    if is_interior:
        k_bic = int(sweep.iloc[bic_argmin_pos]["k"])
        bic_method = "interior-min"
    else:
        k_bic = _find_knee(sweep["k"].values, sweep["bic"].values)
        bic_method = "knee"

    # Step 2: silhouette interior peak (k>=3) for cross-check
    k_sil = _interior_silhouette_peak(sweep, k_min_floor=3)

    # Step 3: combine
    if k_sil is None or abs(k_bic - k_sil) <= 2:
        chosen = k_bic
        note = f"BIC ({bic_method})={k_bic}, silhouette interior={k_sil}; using BIC"
    else:
        chosen = k_bic
        note = f"BIC ({bic_method})={k_bic} vs silhouette interior={k_sil} disagree; using BIC (theoretical anchor)"
        if verbose:
            print(f"[ymeans2] WARN: {note}")

    # Don't return k_min if we're at the edge — clip to interior if possible
    if chosen == k_min and k_sil is not None and k_sil > k_min:
        chosen = k_sil
        note += f" — but k=k_min is degenerate, falling back to silhouette k={k_sil}"

    if verbose:
        print(f"[ymeans2] chosen k={chosen} ({note})")

    return models[chosen], sweep, chosen


if __name__ == "__main__":
    # Standalone smoke test on TF-IDF
    from embeddings_utils import load_sample, build_tfidf, cluster_keywords, ensure_output_dir, OUTPUT_DIR
    import os

    ensure_output_dir()
    sample = load_sample(n=5000)
    print(f"Loaded {len(sample)} samples")

    X, vec = build_tfidf(sample["cleaned_text"])
    model, sweep, chosen_k = run_ymeans2(X)

    sample["ymeans2_cluster"] = model.labels_
    sweep.to_csv(os.path.join(OUTPUT_DIR, "ymeans2_sweep_tfidf.csv"), index=False)

    keywords = cluster_keywords(sample["cleaned_text"], sample["ymeans2_cluster"], vec)
    print(f"\n--> Y-Means v2 chose k={chosen_k}")
    print("\nClass keywords:")
    for cid, words in keywords.items():
        size = (sample["ymeans2_cluster"] == cid).sum()
        print(f"  Class {cid} ({size} records): {', '.join(words)}")

    sample[["registration_no", "subject_content_text", "cleaned_text", "ymeans2_cluster"]].to_csv(
        os.path.join(OUTPUT_DIR, "ymeans2_results_tfidf.csv"), index=False
    )
    print(f"\nResults saved to {OUTPUT_DIR}/ymeans2_results_tfidf.csv")
