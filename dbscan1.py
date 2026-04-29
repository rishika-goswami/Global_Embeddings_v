import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# Ensure required NLTK resources are available
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def run_dbscan_analysis(input_csv='preprocessed_grievances.csv'):
    # 1. Load Preprocessed Data
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['cleaned_text'])

    # 2. Limit to 5000 Samples
    # This matches the scale used in the hierarchical clustering test
    sample_size = min(5000, len(df))
    sample_df = df.iloc[:sample_size].copy()
    print(f"Processing {sample_size} samples for DBSCAN...")

    # 3. Vectorization (TF-IDF)
    # TF-IDF with bigrams captures the context of Hinglish phrases better
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_sample = tfidf.fit_transform(sample_df['cleaned_text'])

    # 4. Apply DBSCAN with 50% Similarity Logic
    # eps = 1 - Similarity (1 - 0.5 = 0.5)
    # metric='cosine' is essential for angular similarity in text
    # min_samples=5 ensures we only form clusters for recurring issues
    dbscan = DBSCAN(
        eps=0.5, 
        min_samples=5, 
        metric='cosine'
    )

    print("Executing DBSCAN (eps=0.5, metric='cosine')...")
    sample_df['dbscan_cluster'] = dbscan.fit_predict(X_sample)

    # 5. Result Analysis
    labels = sample_df['dbscan_cluster']
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"\nResults for 5000 Samples:")
    print(f"-> Dense Semantic Classes Found: {n_clusters_}")
    print(f"-> Uncategorized Records (Noise): {n_noise_}")

    # 6. Extract Top Keywords per Class
    def get_top_keywords(data, vectorizer, n_words=6):
        keywords = {}
        unique_clusters = [c for c in sorted(data['dbscan_cluster'].unique()) if c != -1]
        
        for cluster_id in unique_clusters:
            # Combine text for the specific cluster
            subset = data[data['dbscan_cluster'] == cluster_id]['cleaned_text']
            combined_text = " ".join(subset)
            
            # Use a simple count-based check for the most frequent words in this group
            tokens = combined_text.split()
            freq = pd.Series(tokens).value_counts().head(n_words).index.tolist()
            keywords[cluster_id] = freq
            
        return keywords

    # Get keywords for the largest clusters
    top_cluster_ids = sample_df[sample_df['dbscan_cluster'] != -1]['dbscan_cluster'].value_counts().head(10).index
    cluster_themes = get_top_keywords(sample_df, tfidf)

    print("\nTop Identified Classes:")
    for cid in top_cluster_ids:
        count = sample_df['dbscan_cluster'].value_counts()[cid]
        themes = ", ".join(cluster_themes[cid])
        print(f"Class {cid} ({count} records): {themes}")

    # 7. Save to CSV
    output_filename = 'dbscan_50pct_5000samples.csv'
    sample_df[['registration_no', 'subject_content_text', 'dbscan_cluster']].to_csv(output_filename, index=False)
    print(f"\nOutput saved to {output_filename}")

if __name__ == "__main__":
    # Ensure you have run the preprocessing script first to generate 'preprocessed_grievances.csv'
    run_dbscan_analysis('preprocessed_grievances.csv')