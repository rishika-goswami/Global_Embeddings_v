import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

df = pd.read_csv('preprocessed_grievances.csv')
df = df.dropna(subset=['cleaned_text'])

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])

# Run DBSCAN (eps is the maximum distance between records to be considered neighbors)
# min_samples is how many similar records are needed to form a cluster
dbscan = DBSCAN(eps=0.7, min_samples=10)
df['dbscan_cluster'] = dbscan.fit_predict(X)

# Analyze results
n_clusters = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'].values else 0)
n_noise = list(df['dbscan_cluster']).count(-1)

print(f"DBSCAN automatically found {n_clusters} highly specific micro-clusters.")
print(f"Records classified as noise (unique complaints): {n_noise}")

# Print the top 5 largest clusters it found (excluding noise)
cluster_counts = df[df['dbscan_cluster'] != -1]['dbscan_cluster'].value_counts().head(5)
print("\nTop 5 Largest Density Clusters:")
print(cluster_counts)

df.to_csv('dbscan_results.csv', index=False)