import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_bic(kmeans, X):
    """
    Computes the Bayesian Information Criterion (BIC) for a clustering.
    Lower BIC suggests a better model.
    """
    labels = kmeans.labels_
    n_clusters = kmeans.n_clusters
    n_samples, n_features = X.shape
    
    # Calculate variance
    if n_clusters <= 1:
        return np.inf
        
    # Sum of squared errors
    sse = kmeans.inertia_
    
    # BIC formula approximation
    # BIC = SSE + log(n_samples) * (n_clusters * n_features)
    # Lower is better
    bic = sse + np.log(n_samples) * (n_clusters * n_features)
    return bic

def run_ymeans(X_matrix, max_k=15):
    """
    Simulates Y-Means by iteratively splitting clusters and 
    evaluating them using the BIC.
    """
    print("Starting Y-Means autonomous splitting...")
    current_k = 2
    best_bic = np.inf
    best_model = None
    
    # Iterative splitting strategy
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_matrix)
        current_bic = compute_bic(kmeans, X_matrix)
        
        print(f"Evaluating K={k} | BIC: {current_bic:.2f}")
        
        if current_bic < best_bic:
            best_bic = current_bic
            best_model = kmeans
        else:
            # If BIC starts increasing, we've likely passed the optimal K
            # Y-Means logic: Stop when complexity outweighs gain
            print(f"BIC increased at K={k}. Stability reached.")
            break
            
    return best_model

# 1. Load Preprocessed Data
df = pd.read_csv('preprocessed_grievances.csv')
df = df.dropna(subset=['cleaned_text'])

# 2. Vectorization
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])

# 3. Execute Y-Means
ymeans_model = run_ymeans(X)

# 4. Results
optimal_k = ymeans_model.n_clusters
df['ymeans_cluster'] = ymeans_model.labels_

print(f"\n--> Y-Means automatically identified {optimal_k} classes.")

# Semantic Interpretation
order_centroids = ymeans_model.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()

print("\nAutomatically Identified Semantic Classes:")
for i in range(optimal_k):
    top_words = [terms[ind] for ind in order_centroids[i, :8]]
    print(f"Class {i}: {', '.join(top_words)}")

# Export
df.to_csv('ymeans_results.csv', index=False)
print("\nResults saved to ymeans_results.csv")