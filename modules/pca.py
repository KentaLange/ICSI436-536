import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
def compute_pca(x_scaled, dataset_name):
    # Fit PCA on scaled data
    pca_full = PCA()
    pca_full.fit(x_scaled)

    print("Eigenvalues:", np.round(pca_full.explained_variance_, 4))
    print("Explained variance ratio:", np.round(pca_full.explained_variance_ratio_, 4))

    # Explained variance plot
    plt.figure()
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"{dataset_name} - PCA: Cumulative Explained Variance")
    plt.ylim(0, 1.05)
    plt.savefig(rf"C:\Users\patelka\OneDrive - State University of New York\Desktop\ICSI_536\ICSI436-536\output\{dataset_name}_pca_variance.png")
    plt.show()

    # 2D projection
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(x_scaled)
    plt.figure()
    plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{dataset_name} - PCA-2D Projection")
    plt.savefig(rf"C:\Users\patelka\OneDrive - State University of New York\Desktop\ICSI_536\ICSI436-536\output\{dataset_name}_pca_2d.png")
    plt.show()
