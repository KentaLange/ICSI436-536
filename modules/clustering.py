import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compare_clustering(original_data, pca_data, dataset_name, n_clusters=3):
    """
    Compare clustering performance using KMeans and Silhouette Score.
    """
    print(f"\n--- Clustering Comparison for {dataset_name} ---")
    
    # Option to suppress warnings regarding thread leaks on M1 Macs, but standard code is fine.
    # KMeans on Original Data
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_orig = kmeans_orig.fit_predict(original_data)
    sil_orig = silhouette_score(original_data, labels_orig)
    print(f"Silhouette Score (Original Data, {original_data.shape[1]} features): {sil_orig:.4f}")
    
    # KMeans on PCA Data
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pca = kmeans_pca.fit_predict(pca_data)
    sil_pca = silhouette_score(pca_data, labels_pca)
    print(f"Silhouette Score (PCA Data, {pca_data.shape[1]} features): {sil_pca:.4f}")
    
    return sil_orig, sil_pca
