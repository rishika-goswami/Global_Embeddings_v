import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('preprocessed_grievances.csv')
df = df.dropna(subset=['cleaned_text'])

# Note: Hierarchical clustering is very memory intensive O(N^2). 
# If your machine runs out of memory, reduce this sample_size.
sample_size = min(3000, len(df))
sample_df = df.head(sample_size).copy()

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_sample = tfidf.fit_transform(sample_df['cleaned_text']).toarray()

# Automate cluster division by setting a distance threshold instead of hardcoding K
# distance_threshold=1.5 means "don't merge groups if they are more than 1.5 distance apart"
agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='complete', compute_full_tree=True)
sample_df['agg_cluster'] = agg.fit_predict(X_sample)

# Analyze results
found_clusters = agg.n_clusters_
print(f"Hierarchical Clustering automatically stopped at {found_clusters} semantic classes based on the distance threshold.")

print("\nDistribution of records across hierarchical classes:")
print(sample_df['agg_cluster'].value_counts().head(10)) # Print top 10 largest

sample_df.to_csv('hierarchical_results.csv', index=False)