"""Orchestrator: runs Y-Means2, DBSCAN sweep, Agglomerative sweep on a single
shared sample under both TF-IDF and SBERT embeddings, then computes
cross-method convergence and produces all plots and the analytical report.

This is the main deliverable — the script that answers "how many classes?"

Run:  python convergence_analysis.py [--sample-size 5000] [--smoke]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, davies_bouldin_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

from embeddings_utils import (
    load_sample, build_tfidf, build_sbert, cluster_keywords,
    cluster_size_distribution, ensure_output_dir, OUTPUT_DIR,
)
from ymeans2 import run_ymeans2
from dbscan_sweep import (
    sweep_dbscan, pick_combo as pick_dbscan, k_distance_curve,
    EPS_GRID_TFIDF, EPS_GRID_SBERT, DEFAULT_MIN_SAMPLES,
)
from hierarchical_sweep import (
    sweep_agglomerative, pick_combo as pick_agg,
    plot_dendrogram, plot_dendrogram_grid, build_linkage_matrix,
    THRESHOLDS_TFIDF, THRESHOLDS_SBERT, DEFAULT_LINKAGES,
)


def to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


# ---------------- Plots ----------------

def plot_ymeans_curves(sweep_tfidf, sweep_sbert, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sweep, name in zip(axes, [sweep_tfidf, sweep_sbert], ["TF-IDF", "SBERT"]):
        ax2 = ax.twinx()
        ax.plot(sweep["k"], sweep["bic"], "o-", color="tab:blue", label="BIC")
        ax2.plot(sweep["k"], sweep["silhouette"], "s--", color="tab:red", label="silhouette")
        ax.set_xlabel("k")
        ax.set_ylabel("BIC", color="tab:blue")
        ax2.set_ylabel("silhouette (cosine)", color="tab:red")
        ax.set_title(f"Y-Means2 sweep — {name}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kdistance(curve, eps_lines, outfile, title):
    plt.figure(figsize=(10, 5))
    plt.plot(curve)
    for eps in eps_lines:
        plt.axhline(y=eps, color="r", linestyle="--", alpha=0.4, label=f"eps={eps}")
    plt.xlabel("points (sorted)")
    plt.ylabel("k-th NN cosine distance")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def plot_param_grid_dbscan(sweep_df, outfile, title):
    pivot = sweep_df.pivot(index="min_samples", columns="eps", values="n_clusters")
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def plot_param_grid_agg(sweep_df, outfile, title):
    pivot = sweep_df.pivot(index="linkage", columns="threshold", values="n_clusters")
    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cosine_centroid_heatmap(centroids_by_method, outfile):
    n = len(centroids_by_method)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, C) in zip(axes, centroids_by_method.items()):
        sim = cosine_similarity(C)
        sns.heatmap(sim, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                    cbar_kws={"label": "cosine sim"})
        ax.set_title(f"{name}\ncentroid-cosine ({sim.shape[0]} clusters)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_sizes(sizes_by_method, outfile):
    n = len(sizes_by_method)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, sizes) in zip(axes, sizes_by_method.items()):
        items = sorted(sizes.items())
        labels = [str(k) for k, _ in items]
        vals = [v for _, v in items]
        ax.bar(labels, vals)
        ax.set_title(name)
        ax.set_xlabel("cluster id")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ari_heatmap(labels_by_run, outfile, title="Cross-method ARI"):
    names = list(labels_by_run.keys())
    n = len(names)
    M = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = adjusted_rand_score(labels_by_run[a], labels_by_run[b])
    plt.figure(figsize=(1.5 * n + 4, 1.2 * n + 2))
    sns.heatmap(M, annot=True, fmt=".2f", xticklabels=names, yticklabels=names, cmap="viridis",
                vmin=0, vmax=1, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    return pd.DataFrame(M, index=names, columns=names)


def plot_convergence_summary(k_table, outfile):
    """k_table: dict {label: K}"""
    plt.figure(figsize=(10, 5))
    names = list(k_table.keys())
    vals = [k_table[n] for n in names]
    bars = plt.bar(range(len(names)), vals, color=["#1f77b4" if "LLM" not in n else "#d62728" for n in names])
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.ylabel("K (number of classes)")
    plt.title("Convergence summary — K found by each method × embedding")
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v), ha="center")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------- Compute centroids for cosine heatmap ----------------

def compute_centroids(X, labels):
    X_dense = to_dense(X)
    cids = sorted([c for c in set(labels) if c != -1])
    if not cids:
        return None
    return np.vstack([X_dense[labels == c].mean(axis=0) for c in cids])


# ---------------- Main ----------------

def run_pipeline(sample_size=5000, smoke=False, csv_path="preprocessed_grievances.csv"):
    ensure_output_dir()

    if smoke:
        sample_size = 500
    sample = load_sample(csv_path=csv_path, n=sample_size)
    print(f"\n=== Convergence pipeline === sample n={len(sample)} ===\n")

    # Smoke test sanity: how many tokens per row?
    tok_counts = sample["cleaned_text"].str.split().map(len)
    print(f"[stopword-check] mean tokens/row = {tok_counts.mean():.1f}, median = {tok_counts.median():.0f}, "
          f"<3 tokens: {(tok_counts < 3).sum()} rows")
    if tok_counts.mean() < 5:
        print("[stopword-check] WARN: average row length < 5 tokens — stopwording may be too aggressive.")

    # Build embeddings
    X_tfidf, vec_tfidf = build_tfidf(sample["cleaned_text"])

    # Drop rows with zero-vector TF-IDF (cosine is undefined on zero vectors and
    # crashes Agglomerative). Happens when a row's tokens are all out-of-vocabulary
    # or all stopwords. Zero-row count grows with dataset size.
    nz_mask = np.asarray((X_tfidf.power(2).sum(axis=1) > 0)).flatten() if sparse.issparse(X_tfidf) \
              else (np.linalg.norm(X_tfidf, axis=1) > 0)
    n_zero = int((~nz_mask).sum())
    if n_zero > 0:
        print(f"[zero-rows] dropping {n_zero} rows with all-zero TF-IDF vectors")
        sample = sample[nz_mask].reset_index(drop=True)
        X_tfidf = X_tfidf[nz_mask]

    X_sbert = build_sbert(sample["cleaned_text"].tolist(),
                         cache_path=os.path.join(OUTPUT_DIR, f"sbert_cache_{len(sample)}.npy"))

    embeddings = {"tfidf": X_tfidf, "sbert": X_sbert}

    results = {}        # results[method][embedding] = labels array
    chosen_k = {}       # chosen_k[method][embedding] = int
    sweeps = {"ymeans2": {}, "dbscan": {}, "agg": {}}
    chosen_models = {"agg": {}}

    # ---- Y-MEANS2 ----
    for embname, X in embeddings.items():
        print(f"\n[ymeans2/{embname}] ...")
        model, sweep, k = run_ymeans2(X, k_min=2, k_max=15)
        sweep.to_csv(os.path.join(OUTPUT_DIR, f"ymeans2_sweep_{embname}.csv"), index=False)
        results.setdefault("ymeans2", {})[embname] = np.asarray(model.labels_)
        chosen_k.setdefault("ymeans2", {})[embname] = k
        sweeps["ymeans2"][embname] = sweep
    plot_ymeans_curves(sweeps["ymeans2"]["tfidf"], sweeps["ymeans2"]["sbert"],
                       os.path.join(OUTPUT_DIR, "ymeans_bic_curve.png"))

    # ---- DBSCAN sweep ----
    eps_grids = {"tfidf": EPS_GRID_TFIDF, "sbert": EPS_GRID_SBERT}
    for embname, X in embeddings.items():
        print(f"\n[dbscan/{embname}] sweeping ...")
        sweep_df, labels_map = sweep_dbscan(X, eps_grid=eps_grids[embname])
        sweep_df.to_csv(os.path.join(OUTPUT_DIR, f"dbscan_sweep_{embname}.csv"), index=False)
        sweeps["dbscan"][embname] = sweep_df
        target_k = chosen_k["ymeans2"][embname]
        choice = pick_dbscan(sweep_df, target_k=target_k, max_noise_pct=25.0)
        if choice is None:
            print(f"[dbscan/{embname}] WARN: no qualifying combo found")
            results.setdefault("dbscan", {})[embname] = np.full(X.shape[0], -1)
            chosen_k.setdefault("dbscan", {})[embname] = 0
            continue
        eps, ms, row = choice
        print(f"[dbscan/{embname}] picked eps={eps} ms={ms} -> k={row['n_clusters']} (target was {target_k})")
        results.setdefault("dbscan", {})[embname] = np.asarray(labels_map[(eps, ms)])
        chosen_k.setdefault("dbscan", {})[embname] = int(row["n_clusters"])

        # k-distance plot for the chosen min_samples
        curve = k_distance_curve(X, k=ms)
        plot_kdistance(
            curve, eps_lines=sorted(set([eps] + [0.3, 0.5, 0.7])),
            outfile=os.path.join(OUTPUT_DIR, f"dbscan_k_distance_{embname}.png"),
            title=f"k-distance plot — {embname.upper()} (k={ms})",
        )
        plot_param_grid_dbscan(
            sweep_df, os.path.join(OUTPUT_DIR, f"dbscan_param_grid_{embname}.png"),
            f"DBSCAN n_clusters — {embname.upper()}"
        )

    # ---- Agglomerative sweep + dendrograms ----
    thr_grids = {"tfidf": THRESHOLDS_TFIDF, "sbert": THRESHOLDS_SBERT}
    for embname, X in embeddings.items():
        print(f"\n[agg/{embname}] sweeping ...")
        sweep_df, models_map = sweep_agglomerative(X, thresholds=thr_grids[embname])
        sweep_df.to_csv(os.path.join(OUTPUT_DIR, f"agglomerative_sweep_{embname}.csv"), index=False)
        sweeps["agg"][embname] = sweep_df
        target_k = chosen_k["ymeans2"][embname]
        choice = pick_agg(sweep_df, target_k=target_k)
        if choice is None:
            print(f"[agg/{embname}] WARN: no qualifying combo")
            results.setdefault("agg", {})[embname] = np.full(X.shape[0], 0)
            chosen_k.setdefault("agg", {})[embname] = 1
            continue
        linkage, thr, row = choice
        model, labels = models_map[(linkage, thr)]
        print(f"[agg/{embname}] picked linkage={linkage} thr={thr} -> k={row['n_clusters']} (target was {target_k})")
        results.setdefault("agg", {})[embname] = np.asarray(labels)
        chosen_k.setdefault("agg", {})[embname] = int(row["n_clusters"])
        chosen_models["agg"][embname] = (linkage, thr, model)

        # Param grid heatmap
        plot_param_grid_agg(
            sweep_df, os.path.join(OUTPUT_DIR, f"agglomerative_param_grid_{embname}.png"),
            f"Agglomerative n_clusters — {embname.upper()}"
        )

        # Chosen dendrogram
        plot_dendrogram(
            model, threshold=thr,
            title=f"Agglomerative dendrogram — {embname.upper()} (linkage={linkage}, thr={thr}, k={model.n_clusters_})",
            outfile=os.path.join(OUTPUT_DIR, f"dendrogram_{embname}_chosen.png"),
        )

        # Side-by-side dendrograms across all 3 linkages at the chosen threshold
        models_at_thr = {}
        for link in ["average", "complete", "single"]:
            if (link, thr) in models_map:
                m, _ = models_map[(link, thr)]
                models_at_thr[link] = m
        plot_dendrogram_grid(
            models_at_thr, threshold=thr,
            title_prefix=f"{embname.upper()}",
            outfile=os.path.join(OUTPUT_DIR, f"dendrogram_{embname}_all_linkages.png"),
        )

    # ---- Cross-method comparison ----
    labels_by_run = {}
    for method, embmap in results.items():
        for embname, labels in embmap.items():
            labels_by_run[f"{method}/{embname}"] = labels

    ari_df = plot_ari_heatmap(
        labels_by_run, outfile=os.path.join(OUTPUT_DIR, "cross_method_ari_heatmap.png"),
        title="Cross-method ARI (3 methods × 2 embeddings)"
    )
    ari_df.to_csv(os.path.join(OUTPUT_DIR, "cross_method_ari.csv"))

    # NMI table too
    names = list(labels_by_run.keys())
    nmi_M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            nmi_M[i, j] = normalized_mutual_info_score(labels_by_run[a], labels_by_run[b])
    nmi_df = pd.DataFrame(nmi_M, index=names, columns=names)
    nmi_df.to_csv(os.path.join(OUTPUT_DIR, "cross_method_nmi.csv"))

    # ---- Centroid cosine heatmaps ----
    centroids = {}
    for run_name, labels in labels_by_run.items():
        method, emb = run_name.split("/")
        X = embeddings[emb]
        C = compute_centroids(X, labels)
        if C is not None and C.shape[0] >= 2:
            centroids[run_name] = C
    if centroids:
        plot_cosine_centroid_heatmap(centroids, os.path.join(OUTPUT_DIR, "cosine_similarity_heatmap.png"))

    # ---- Cluster size distribution ----
    sizes = {name: cluster_size_distribution(lab) for name, lab in labels_by_run.items()}
    plot_cluster_sizes(sizes, os.path.join(OUTPUT_DIR, "cluster_size_distribution.png"))

    # ---- Per-run summary table ----
    rows = []
    for run_name, labels in labels_by_run.items():
        method, emb = run_name.split("/")
        X = embeddings[emb]
        X_dense = to_dense(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        if n_clusters >= 2:
            mask = labels != -1
            try:
                sil = silhouette_score(X_dense[mask], labels[mask], metric="cosine",
                                       sample_size=min(2000, mask.sum()), random_state=42)
            except Exception:
                sil = float("nan")
            try:
                db = davies_bouldin_score(X_dense[mask], labels[mask])
            except Exception:
                db = float("nan")
        else:
            sil = float("nan")
            db = float("nan")
        rows.append({
            "method": method, "embedding": emb, "k": n_clusters,
            "noise_pct": round(100 * n_noise / len(labels), 2),
            "silhouette": round(sil, 4) if not np.isnan(sil) else None,
            "davies_bouldin": round(db, 4) if not np.isnan(db) else None,
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "method_summary.csv"), index=False)
    print("\n=== Method summary ===")
    print(summary_df.to_string(index=False))

    # ---- Per-cluster keywords for each chosen run ----
    keyword_dump = {}
    for run_name, labels in labels_by_run.items():
        kw = cluster_keywords(sample["cleaned_text"], labels, vec_tfidf, n_words=8)
        keyword_dump[run_name] = kw
    with open(os.path.join(OUTPUT_DIR, "cluster_keywords.json"), "w") as f:
        json.dump(keyword_dump, f, indent=2)

    # ---- LLM comparison (if labels exist) ----
    llm_path = os.path.join(OUTPUT_DIR, "llm_labels.csv")
    llm_k = None
    if os.path.exists(llm_path):
        llm = pd.read_csv(llm_path)
        # Join on row_idx (truly unique). registration_no isn't — same ID appears ~10x.
        join_key = "row_idx" if "row_idx" in llm.columns else "registration_no"
        merged = sample.merge(llm[[join_key, "llm_category", "llm_confidence"]],
                              on=join_key, how="left")
        if merged["llm_category"].notna().sum() >= 100:
            llm_labels = merged["llm_category"].astype("category").cat.codes.values
            llm_k = int(merged["llm_category"].nunique())
            print(f"\n[llm-compare] LLM K = {llm_k}")
            llm_ari_rows = []
            for run_name, labels in labels_by_run.items():
                ari = adjusted_rand_score(llm_labels, labels)
                nmi = normalized_mutual_info_score(llm_labels, labels)
                llm_ari_rows.append({"run": run_name, "ARI_vs_llm": ari, "NMI_vs_llm": nmi})
            llm_cmp = pd.DataFrame(llm_ari_rows)
            llm_cmp.to_csv(os.path.join(OUTPUT_DIR, "llm_vs_clustering.csv"), index=False)
            print(llm_cmp.to_string(index=False))
        else:
            print("[llm-compare] not enough rows in sample matched llm_labels.csv to compare")
    else:
        print(f"[llm-compare] {llm_path} not found — run llm_annotate.py to add the LLM anchor")

    # ---- Convergence summary plot ----
    k_table = {}
    for method in ["ymeans2", "dbscan", "agg"]:
        for emb in ["tfidf", "sbert"]:
            k_table[f"{method}/{emb}"] = chosen_k.get(method, {}).get(emb, 0)
    if llm_k is not None:
        k_table["LLM (Pass A+B)"] = llm_k
    plot_convergence_summary(k_table, os.path.join(OUTPUT_DIR, "convergence_summary.png"))

    # ---- Markdown report ----
    write_report(sample, summary_df, ari_df, nmi_df, k_table, keyword_dump,
                 chosen_k, sweeps, llm_k=llm_k)

    print(f"\nAll outputs in {OUTPUT_DIR}/")


def write_report(sample, summary_df, ari_df, nmi_df, k_table, keyword_dump,
                 chosen_k, sweeps, llm_k=None):
    out = []
    out.append("# Convergence Report — Clustering of MORLY Grievances\n")
    out.append(f"**Sample size:** {len(sample)} rows\n")

    out.append("## K found by each method × embedding\n")
    out.append("| Run | K |")
    out.append("|---|---|")
    for k, v in k_table.items():
        out.append(f"| {k} | {v} |")
    out.append("")

    out.append("## Method summary (silhouette, davies-bouldin, noise%)\n")
    out.append(summary_df.to_markdown(index=False))
    out.append("")

    # Convergence assessment
    ks = [chosen_k[m][e] for m in ["ymeans2", "dbscan", "agg"] for e in ["tfidf", "sbert"]
          if chosen_k.get(m, {}).get(e, 0) > 0]
    if ks:
        spread = max(ks) - min(ks)
        recommended = int(np.median(ks))
        out.append(f"## Convergence assessment\n")
        out.append(f"- K range across methods: **{min(ks)} – {max(ks)}** (spread = {spread})")
        out.append(f"- Recommended K (median): **{recommended}**")
        if llm_k is not None:
            out.append(f"- LLM Pass-A+B taxonomy K: **{llm_k}**")
        if spread <= 2:
            out.append(f"- ✅ Methods converged within ±{spread} of each other.")
        elif spread <= 4:
            out.append(f"- ⚠️ Moderate spread ({spread}); inspect cluster keywords to decide.")
        else:
            out.append(f"- ❌ High spread ({spread}); the data does not strongly support a single K.")
        out.append("")

    out.append("## Cross-method ARI (higher = more agreement)\n")
    out.append(ari_df.round(2).to_markdown())
    out.append("")
    out.append("## Cross-method NMI\n")
    out.append(nmi_df.round(2).to_markdown())
    out.append("")

    out.append("## Top keywords per cluster (per run)\n")
    for run, kw in keyword_dump.items():
        out.append(f"### {run}\n")
        for cid, words in sorted(kw.items()):
            out.append(f"- **Class {cid}**: {', '.join(words)}")
        out.append("")

    out.append("## Files produced\n")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        out.append(f"- `{f}`")

    with open(os.path.join(OUTPUT_DIR, "convergence_report.md"), "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true",
                        help="Run on 500-row sub-sample for quick smoke test")
    parser.add_argument("--csv", default="preprocessed_grievances.csv")
    args = parser.parse_args()
    run_pipeline(sample_size=args.sample_size, smoke=args.smoke, csv_path=args.csv)
