# Grievance Clustering — Convergence Across Three Algorithms

This repository decides **how many natural classes** exist in the (Indian
Railways, Ministry of Railways) grievance dataset of 15,582 mixed
English / Hinglish complaints. Rather than picking a `K` arbitrarily, three
independent clustering algorithms are tuned until they (almost) agree — with an
LLM-annotated open-coded taxonomy serving as an external anchor. The reference
methodology follows the MDPI Informatics paper at
<https://www.mdpi.com/2227-9709/12/3/82>.

The pipeline runs each algorithm twice: once on TF-IDF features and once on
multilingual sentence-transformer embeddings, so the choice of representation
is also evaluated.

---

## What was added today

| File | Purpose |
|---|---|
| [embeddings_utils.py](embeddings_utils.py) | Shared loaders so all three algorithms cluster the **same** 5,000-row sample under the **same** features. Builds TF-IDF (L2-normalized) and SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) embeddings and caches SBERT to disk. |
| [ymeans2.py](ymeans2.py) | New cosine-aware Y-Means. Replaces Euclidean k-means with spherical k-means (L2-norm + standard k-means ≡ cosine), uses standard BIC `n·log(SSE/n) + k·log(n)` instead of the original heuristic, and reports silhouette alongside BIC. Original [ymeans.py](ymeans.py) untouched. |
| [dbscan_sweep.py](dbscan_sweep.py) | Sweeps `eps × min_samples` per representation (different grids for TF-IDF and SBERT — they live on different cosine-distance scales), then picks the combo whose `K` is closest to the Y-Means anchor under a 25 % noise ceiling. Also produces the k-distance elbow plot. Original [dbscan1.py](dbscan1.py) untouched. |
| [hierarchical_sweep.py](hierarchical_sweep.py) | Sweeps `linkage × distance_threshold` (linkages: average / complete / single — Ward is excluded because scikit-learn forbids Ward + cosine), reports silhouette + Davies-Bouldin, picks combo nearest the anchor, and produces the dendrograms (chosen + side-by-side comparison across all three linkages). Original [hierarchical1.py](hierarchical1.py) untouched. |
| [convergence_analysis.py](convergence_analysis.py) | Orchestrator. Runs Y-Means → reads its `K` → tunes DBSCAN and Agglomerative around it → computes pairwise ARI / NMI between all six runs → renders 12+ plots → writes the convergence report. |
| [llm_annotate.py](llm_annotate.py) | Two-pass open-coded annotation with `gpt-4o-mini`. Pass A discovers free-text categories on a 300-row sample then asks the LLM to consolidate them into 5–10 buckets. Pass B labels all 15,582 rows in parallel with tqdm + checkpoint resume + per-retry rate-limit logging. |
| [requirements.txt](requirements.txt) | Adds `sentence-transformers`. |

The original scripts (`ymeans.py`, `dbscan1.py`, `hierarchical1.py`,
`preprocess_data.py`, etc.) are unmodified — every new piece is additive.

---

## How to run

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2a. Quick run on a 5,000-row sample (~6 min, cleaner convergence — recommended for first pass)
python3.12 -u convergence_analysis.py --sample-size 5000

# 2b. Full run on all ~15,582 rows (~75–90 min on CPU, more honest but Agglomerative degenerates)
python3.12 -u convergence_analysis.py --sample-size 20000

# Smoke test variant (500-row sub-sample, ~30 sec, useful before changing parameters):
python3.12 -u convergence_analysis.py --smoke

# 3. Open-coded LLM annotation of the full 15,582-row dataset.
#    --workers 10 --batch-size 1 is the sustainable setting for OpenAI Tier-1 limits.
python3.12 llm_annotate.py --workers 10 --batch-size 1 --skip-confirm

# 4. After llm_labels.csv exists, re-run step 2 — the orchestrator now folds the LLM
#    K into the convergence summary and the cross-method ARI/NMI tables.
python3.12 -u convergence_analysis.py --sample-size 20000
```

The `-u` flag forces unbuffered Python stdout so `tee`'d log files reflect progress
in real time. Passing `--sample-size 20000` to a 15,582-row dataset just caps to
the dataset size — there is no separate "all rows" flag.

You can also run any individual algorithm standalone (each module has a
`__main__` smoke entry point):

```bash
python3.12 ymeans2.py
python3.12 dbscan_sweep.py
python3.12 hierarchical_sweep.py
```

---

## The three clustering methods

All three operate on **cosine** similarity (or its complement, cosine distance)
because text-similarity is angle-driven; magnitude carries little semantic
weight, especially after L2-normalization.

### 1. Y-Means (autonomous K, BIC-driven) — anchor

- Spherical k-means: input is L2-normalized so plain k-means' Euclidean
  objective is monotonic with cosine distance.
- Sweep `K ∈ [2, 15]`, fit k-means at every `K`, record BIC and silhouette.
- Pick `K`:
  1. If BIC has an **interior minimum** (a true low between `K=2` and `K=15`),
     use it.
  2. Otherwise BIC is monotonic — fall back to a Kneedle-style elbow finder
     and cross-check against silhouette's interior peak (`K ≥ 3`, since
     `K=2` is often a degenerate split for high-dim cosine data).

### 2. DBSCAN — density-based, sweep over `eps × min_samples`

- Fixed parameter: `metric='cosine'`.
- Tunes `eps` (neighborhood radius in cosine distance) and `min_samples`
  (density threshold to seed a cluster). Per representation:

  | embedding | `eps` grid | `min_samples` grid |
  |---|---|---|
  | TF-IDF | `{0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8}` | `{3, 5, 8, 10, 15}` |
  | SBERT | `{0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35}` | `{3, 5, 8, 10, 15}` |

  SBERT cosine distances are roughly 5× smaller than TF-IDF for short noisy
  text, hence the smaller grid.
- Pick combo whose number of clusters is closest to Y-Means' `K` while keeping
  noise ≤ 25 %.

### 3. Agglomerative (hierarchical) — sweep over `linkage × distance_threshold`

- Fixed parameter: `metric='cosine'`.
- Tunes `linkage ∈ {average, complete, single}` and `distance_threshold`.
  Ward linkage is excluded because scikit-learn forbids Ward with cosine.

  | embedding | `distance_threshold` grid |
  |---|---|
  | TF-IDF | `{0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}` |
  | SBERT | `{0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50}` |

- Pick combo whose `K` is closest to the Y-Means anchor with a soft silhouette
  floor.

### Finding the point of (almost) convergence

The orchestrator runs Y-Means first, takes its `K` as a target, then tunes
DBSCAN and Agglomerative parameters until their `K` lands as close to that
target as their respective sweeps allow. The clustering converges when the
spread between methods is small (`≤ 3` clusters within an embedding,
`≤ 6` across both). The data is **not** forced — if a method's
sweep cannot produce the target `K`, the closest achievable `K` is reported
and flagged.

---

## Selected parameters and the K each method (almost) converged on

The pipeline was run twice — once on a deterministic 5,000-row sample for a fast
diagnostic, and once on the full **15,582 rows**. The 15k run is what the rest
of this README is built around; the 5k numbers are kept as a comparison column
because the convergence behavior changed materially between the two scales.

| Method | Embedding | Metric | Selected parameters (15k) | K (15k) | K (5k) |
|---|---|---|---|---|---|
| **Y-Means2** | TF-IDF (1000 feat, bigrams, L2-norm) | cosine (spherical k-means) | `n_init=3`, `K_max=15`, BIC + silhouette sweep, Kneedle fallback | **15** | 15 |
| **Y-Means2** | SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) | cosine | same algorithm | **15** | 15 |
| **DBSCAN** | TF-IDF | `cosine` | `eps=0.4`, `min_samples=10`, noise = **16.4 %** | **13** | 12 |
| **DBSCAN** | SBERT | `cosine` | `eps=0.15`, `min_samples=3`, noise = **6.5 %** | **21** | 18 |
| **Agglomerative** | TF-IDF | `cosine` | `linkage=single`, `distance_threshold=0.6` | **4** ⚠ | 15 |
| **Agglomerative** | SBERT | `cosine` | `linkage=single`, `distance_threshold=0.35` | **11** | 16 |

**Interpretation of the cosine threshold** (since `distance_threshold = 1 − similarity`):
- DBSCAN/TF-IDF `eps=0.4` → cosine similarity ≥ 60 % to be neighbours.
- DBSCAN/SBERT `eps=0.15` → cosine similarity ≥ 85 % to be neighbours
  (SBERT vectors are denser/closer, hence the smaller eps).
- Agg/TF-IDF `threshold=0.6` → clusters merge until inter-cluster cosine
  similarity drops below 40 %.
- Agg/SBERT `threshold=0.35` → merge until SBERT cosine similarity drops
  below 65 %.

### Where convergence (almost) holds at 15k

| Embedding | K range (excluding Agg/TF-IDF outlier) | Spread |
|---|---|---|
| TF-IDF | 13 (DBSCAN) – 15 (Y-Means) | **2** |
| SBERT | 11 (Agg) – 21 (DBSCAN) | **10** |
| **All 6 runs (raw)** | **4 – 21** | **17** |
| **All 6 runs (Agg/TF-IDF set aside)** | **11 – 21** | **10** |

The cleanest signal: **Y-Means picks K=15 under every embedding and every
sample size we tried**. DBSCAN sits a few clusters above (SBERT) or below
(TF-IDF) that anchor. Agglomerative on SBERT (K=11) brackets it from below.

### Where it does *not* converge at 15k

- **Agglomerative on TF-IDF degenerates at this scale** (the ⚠ above). The
  single-linkage dendrogram has a *cliff*: at `threshold=0.5` it has
  136 clusters, at `threshold=0.6` it collapses straight to 4, and there is
  no parameter combination in between that lands at K ≈ 15. Average and
  complete linkage stay above 354 clusters even at the largest swept
  threshold (`0.95`). At 5k the same single-linkage path landed neatly at
  K=15; at 15k it can't. **Treat the K=4 figure as a degenerate artifact
  of the cliff, not as evidence that the data has 4 classes.**
- **Across embeddings, K spreads widely** (11–21 if we ignore Agg/TF-IDF).
  SBERT consistently surfaces more clusters than TF-IDF because it captures
  semantic similarity — *"refund delay"* and *"paisa nahi mila"* are
  near-neighbours under SBERT but not under TF-IDF, so SBERT can keep them
  in one merged group while TF-IDF splits them off, and density-based
  methods see the SBERT structure as more, smaller clumps.
- **Cross-method ARI is low** even when `K` agrees. Maximum off-diagonal ARI
  on 15k is **0.27** (Y-Means/TF-IDF ↔ Y-Means/SBERT — same algorithm,
  different features). Every other pair is essentially zero. Methods agree
  on the *count* but slice the data along different boundaries — see
  [outputs/cross_method_ari.csv](outputs/cross_method_ari.csv) and the
  heatmap.

### What changed from 5k → 15k

| Aspect | 5k | 15k | Change |
|---|---|---|---|
| Y-Means TF-IDF / SBERT K | 15 / 15 | 15 / 15 | unchanged |
| DBSCAN/TF-IDF K, noise | K=12, 24.4 % | K=13, 16.4 % | slightly higher K, less noise |
| DBSCAN/SBERT K, noise | K=18, 11.2 % | K=21, 6.5 % | more clusters, less noise |
| Agg/TF-IDF K, silhouette | K=15, 0.003 | K=4, 0.016 | **degenerated** |
| Agg/SBERT K, silhouette | K=16, 0.124 | K=11, 0.098 | smaller K, slightly worse |
| Y-Means/SBERT silhouette, DB | 0.20, 2.33 | 0.225, 2.22 | both improved |
| Overall K spread | 6 | 17 (or 10 ignoring Agg/TF-IDF) | **wider** at 15k |

Two qualitative findings: **DBSCAN is more confident at 15k** (less noise on
both embeddings) and **Y-Means is identical** — but **Agglomerative single-linkage
becomes unreliable at this scale** because the chaining-cliff that was
manageable at 5k becomes prohibitive at 15k. If Agglomerative is needed at
15k+, Ward linkage on a Euclidean (PCA-reduced) representation is the right
move — cosine + single-linkage simply doesn't scale.

---

## Generated graphs and what they show

| File | What it shows |
|---|---|
| [outputs/ymeans_bic_curve.png](outputs/ymeans_bic_curve.png) | Two-panel plot: BIC and silhouette as functions of `K` for both TF-IDF and SBERT. The silhouette peak indicates the cosine-cluster quality optimum; the BIC trajectory shows where the model penalty starts outweighing the SSE gain. |
| [outputs/dbscan_k_distance_tfidf.png](outputs/dbscan_k_distance_tfidf.png) / [outputs/dbscan_k_distance_sbert.png](outputs/dbscan_k_distance_sbert.png) | Sorted k-th nearest-neighbour cosine distance — the "elbow" of this curve is a principled `eps` for DBSCAN. |
| [outputs/dbscan_param_grid_tfidf.png](outputs/dbscan_param_grid_tfidf.png) / [outputs/dbscan_param_grid_sbert.png](outputs/dbscan_param_grid_sbert.png) | Heatmap of `n_clusters` over the full `eps × min_samples` sweep — shows the parameter region that produces a stable cluster count. |
| [outputs/agglomerative_param_grid_tfidf.png](outputs/agglomerative_param_grid_tfidf.png) / [outputs/agglomerative_param_grid_sbert.png](outputs/agglomerative_param_grid_sbert.png) | Heatmap of `n_clusters` over `linkage × threshold`. Single-linkage drops to small `K` very steeply; average/complete plateau at hundreds of clusters until very high thresholds. |
| [outputs/dendrogram_tfidf_chosen.png](outputs/dendrogram_tfidf_chosen.png) / [outputs/dendrogram_sbert_chosen.png](outputs/dendrogram_sbert_chosen.png) | The hierarchical-clustering dendrogram for the *chosen* linkage and threshold, with a horizontal line at the cut height annotated with the resulting `K`. |
| [outputs/dendrogram_tfidf_all_linkages.png](outputs/dendrogram_tfidf_all_linkages.png) / [outputs/dendrogram_sbert_all_linkages.png](outputs/dendrogram_sbert_all_linkages.png) | 3-panel side-by-side dendrograms (average / complete / single) at the chosen threshold — visualizes how aggressively each linkage merges at the same cut height. |
| [outputs/cosine_similarity_heatmap.png](outputs/cosine_similarity_heatmap.png) | For each method × embedding, the pairwise cosine similarity of cluster centroids. Highly off-diagonal values mean the clusters bleed into each other; near-zero off-diagonals mean the clusters are well separated. |
| [outputs/cluster_size_distribution.png](outputs/cluster_size_distribution.png) | Bar charts of `count per cluster` for every run — shows class balance, dominance of a single mega-cluster (common for DBSCAN), or tail of small singleton classes. |
| [outputs/cross_method_ari_heatmap.png](outputs/cross_method_ari_heatmap.png) | 6×6 matrix of Adjusted Rand Index between all method × embedding combinations — answers "do methods agree on partitioning, not just on count?" |
| [outputs/convergence_summary.png](outputs/convergence_summary.png) | One-shot bar chart of the `K` each run found, including the LLM Pass-A taxonomy size for comparison. The headline visualization. |

---

## Results (15,582-row run)

### Method summary table

This is `outputs/method_summary.csv` rendered. Lower Davies–Bouldin and higher
silhouette are better.

| method | embedding | K | noise % | silhouette | davies–bouldin |
|---|---|---|---|---|---|
| Y-Means2 | tfidf | **15** | 0.00 | 0.0817 | 4.4621 |
| Y-Means2 | sbert | **15** | 0.00 | 0.2245 | 2.2207 |
| DBSCAN | tfidf | **13** | 16.39 | -0.0296 | 2.0813 |
| DBSCAN | sbert | **21** | 6.51 | -0.1973 | 1.3787 |
| Agglomerative | tfidf | **4** ⚠ | 0.00 | 0.0158 | 0.9859 |
| Agglomerative | sbert | **11** | 0.00 | 0.0983 | 0.8055 |

### TF-IDF vs SBERT

- **K count:** SBERT consistently surfaces *more* clusters than TF-IDF —
  semantic embedding lets it split classes that TF-IDF lumps together because
  of token-overlap. The difference grows with data size: at 5k SBERT was
  +1–3 clusters; at 15k it's +6–8 (DBSCAN: 13 vs 21; Agg: 4 vs 11).
- **Cluster quality:** SBERT scores significantly better on both
  silhouette and Davies–Bouldin under Y-Means (silhouette 0.224 vs 0.082,
  DB 2.22 vs 4.46) and Agglomerative. SBERT centroids are more compact and
  more separated.
- **DBSCAN exception:** DBSCAN/SBERT silhouette is negative (-0.20) because
  its choice of `eps=0.15, min_samples=3` sits in a steep region of the
  k-distance curve — many borderline points get pulled into
  weakly-cohesive clusters. The trade-off: noise drops to 6.5 %.
- **Robustness across data size:** Y-Means is the only method whose K is
  identical at 5k and 15k, on both embeddings. DBSCAN drifts slightly higher
  (+1, +3). Agglomerative drifts strongly downward (-11, -5).

### Dendrograms

- **TF-IDF dendrogram (single linkage at threshold 0.6).** At 15k the cut
  yields **K=4** — a degenerate result. The dendrogram shows the chaining
  cliff at threshold ≈ 0.55–0.6 where the structure collapses from ~136
  clusters straight to 4. There is no flat plateau where a sensible K=15
  cut could land on this representation at this scale.
- **SBERT dendrogram (single linkage at threshold 0.35, ≈ 65 % similarity)**
  yields K=11 and is more readable. A clear hierarchy emerges between
  thresholds 0.30 and 0.40, where coarse semantic groups
  (payment / cleanliness / staff / scheduling) split off cleanly. SBERT's
  semantic compactness is what makes hierarchical clustering still
  workable here.
- **Side-by-side linkage comparison** (`dendrogram_*_all_linkages.png`):
  - TF-IDF: average and complete linkages stay at 354+ clusters even at
    `threshold=0.95`; single linkage cliffs from 136 to 4 to 1 across one
    threshold step. None of them produce a usable mid-range K.
  - SBERT: average linkage smoothly drops 2,366 → 52 across the threshold
    sweep; complete linkage 2,680 → 203; single linkage 1,390 → 4. SBERT
    gives a real hierarchy.

### Number of classes — recommendation

Reconciling all 7 signals (3 algorithms × 2 embeddings + the LLM Pass-A
taxonomy of 10 buckets):

| Source | K |
|---|---|
| Y-Means / TF-IDF | 15 |
| Y-Means / SBERT | 15 |
| DBSCAN / TF-IDF | 13 |
| DBSCAN / SBERT | 21 |
| Agglomerative / TF-IDF | 4 ⚠ (degenerate) |
| Agglomerative / SBERT | 11 |
| LLM Pass-A taxonomy | **10** |

**Two defensible answers:**

- **K = 15 (algorithmic median, supported by Y-Means)** — the count three of
  six clustering runs land on or near. Use this if you want the
  unsupervised-clustering answer.
- **K = 10 (LLM taxonomy + Agg/SBERT bracket)** — the open-coded
  categorization from the LLM, plus Agg/SBERT at K=11 as the closest
  hierarchical bracket. The cluster keywords from Y-Means at K=15 collapse
  cleanly onto these 10 buckets when you fold sub-splits like
  *"refund delay"* and *"refund cancel"* into one bucket. Use this if you
  want a production triage taxonomy.

**My informed opinion:** ship **10 classes** for routing/triage and let the
finer sub-clusters from K=15 surface as confidence-weighted secondary tags.
Cross-method ARI is too low (max 0.27) to claim that any specific 15-way
partition is "the" partition — but the *count* of about 10–15 thematically
distinct concerns is well-supported.

Reasoning for the 10-class production taxonomy:

| # | Class | Trigger keywords (Hinglish + English) |
|---|---|---|
| 1 | Payment failure | payment nahi, paisa, account se gaya, payment fail |
| 2 | Refund delay | refund nahi mila, refund delay, paise wapas |
| 3 | Ticket booking issue | booking, ticket payment, confirm nahi, error |
| 4 | Train delay / scheduling | train late, schedule, timing, delay |
| 5 | Train cancellation / changes | cancel, baar baar change, bina notice |
| 6 | Staff misbehavior | staff rude, behavior, attitude, badtameez |
| 7 | Staff training inadequate | staff training zarurat, training nahi, galat info |
| 8 | Cleanliness | safai nahi, ganda, washroom, dirty, toilet |
| 9 | Station / platform infrastructure | station kharab, building purani, facilities, platform halat |
| 10 | Service quality (catch-all) | bekar, samasya, dikkat, generic complaint |

---

## Limitations and caveats to keep in mind

- **BIC was monotonically decreasing** for both Y-Means runs in `K ∈ [2, 15]`,
  meaning the algorithm fell to its Kneedle elbow finder. The current Kneedle
  implementation biases toward the high-K edge in this regime, so the chosen
  `K=15` should be read as "BIC says ≥15" rather than "BIC-optimal at exactly
  15". Bumping `K_max` to 25 or switching to a gap statistic would let the
  algorithm find a true minimum if one exists.
- **DBSCAN/TF-IDF noise ≈ 16 %** at 15k (was 24 % at 5k). Better than before,
  but ~1 in 6 points still don't fit any dense region; the rest of the sweep
  either collapses to one cluster (`eps ≥ 0.6`) or sheds too much noise
  (`eps ≤ 0.35`).
- **Agglomerative + cosine + single-linkage breaks at 15k.** Average and
  complete linkages produce hundreds of clusters even at the highest swept
  threshold; single-linkage chains and then cliffs from 136 → 4 across one
  threshold step. The `K=4` figure is a numerical artifact, not a finding
  about the data. For hierarchical clustering at 15k+, switch to Ward
  linkage on Euclidean (PCA-reduced) embeddings.
- **Cross-method ARI is very low** (max 0.27 at 15k) even when `K` agrees.
  Methods agree on the *count* but slice the data along different
  boundaries. Treat the convergent K as a *count*, not as a validated
  *partition*.
- **Synthetic data has duplicate `registration_no`s** (~10× repetition per
  ID). All resume / dedup logic in `llm_annotate.py` uses positional
  `row_idx`, not `registration_no`, to handle this.
- **Run time at 15k is ~75–90 minutes on a single CPU core** — most of it
  spent in the Agglomerative sweep (24 combos × an `O(n²) ≈ 1.8 GB` distance
  matrix per combo). The 5k run finishes in ~6 minutes. If you only need
  Y-Means + DBSCAN you can comment out the Agglomerative stage in
  `convergence_analysis.py` and bring the full-15k run down to ~15–20
  minutes.

---

## File map

```
preprocessed_grievances.csv       # 15,582 cleaned rows, columns: registration_no, subject_content_text, cleaned_text
embeddings_utils.py               # shared TF-IDF + SBERT loaders
ymeans2.py                        # cosine-aware Y-Means with corrected BIC
dbscan_sweep.py                   # DBSCAN sweep (cosine)
hierarchical_sweep.py             # Agglomerative sweep + dendrograms (cosine)
llm_annotate.py                   # gpt-4o-mini two-pass open-coded annotation
convergence_analysis.py           # orchestrator — run this
outputs/                          # all generated plots, CSVs, and the markdown report
  method_summary.csv              # the headline table
  convergence_report.md           # full English narrative of one run
  convergence_summary.png         # K-comparison bar chart
  ... (see "Generated graphs" section above)
```
