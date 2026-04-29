import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

# 1. Load Preprocessed Data
df = pd.read_csv('preprocessed_grievances.csv')
df = df.dropna(subset=['cleaned_text'])

# 2. Vectorization
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])

# Sampling 5000 records for consistency
sample_size = min(5000, X.shape[0])
X_sample = X[:sample_size].toarray()
sample_df = df.iloc[:sample_size].copy()

# 3. Apply Hierarchical Clustering with 20% Similarity Logic
# Distance Threshold = 0.8
model = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=0.8, 
    metric='cosine', 
    linkage='complete',
    compute_distances=True
)

print(f"Running Hierarchical Clustering (20% Similarity / 0.8 Distance)...")
sample_df['agg_cluster'] = model.fit_predict(X_sample)

# 4. THE COUNT
n_classes_at_dotted_line = model.n_clusters_

print("\n" + "="*40)
print(f"NUMBER OF CLASSES AT THE DOTTED LINE (0.8): {n_classes_at_dotted_line}")
print("="*40)

# 5. Visualization
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

plt.figure(figsize=(12, 8))
plt.title(f"Hierarchical Clustering Dendrogram\n(Threshold: 20% Similarity / 0.8 Distance)")
plot_dendrogram(model, truncate_mode="level", p=5)
plt.xlabel("Cluster Size")
plt.ylabel("Cosine Distance")

# The Dotted Line
plt.axhline(y=0.8, color='r', linestyle='--', label=f'Threshold Line ({n_classes_at_dotted_line} Classes)')
plt.legend()
plt.show()

# 6. Semantic Check for Broad Themes
def get_cluster_keywords(data, cluster_col, vectorizer, n_words=8):
    group_text = data.groupby(cluster_col)['cleaned_text'].apply(lambda x: ' '.join(x))
    cluster_matrix = vectorizer.transform(group_text)
    words = vectorizer.get_feature_names_out()
    cluster_labels = {}
    for i, row in enumerate(cluster_matrix.toarray()):
        top_indices = row.argsort()[-n_words:][::-1]
        cluster_labels[i] = [words[idx] for idx in top_indices]
    return cluster_labels

keywords = get_cluster_keywords(sample_df, 'agg_cluster', tfidf)
top_clusters = sample_df['agg_cluster'].value_counts().head(5).index

print("\nTop 5 Broad Themes Found:")
for cid in top_clusters:
    count = sample_df['agg_cluster'].value_counts()[cid]
    print(f"Class {cid} ({count} records): {', '.join(keywords[cid])}")