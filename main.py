import pandas as pd
from modules.load_data import get_iris_data_set, get_blob_dataset, get_moons_dataset, get_wine_dataset
from modules.standardize import standardize_data
from modules.plot_data import plot_corr
from modules.pca import compute_pca
from modules.clustering import compare_clustering

if __name__ == "__main__":
    """
    #      Iris Dataset     #
    df_iris, iris = get_iris_data_set()
    standard_iris_df = standardize_data(df_iris)
    feature_names = iris.feature_names
    plot_corr(
        df=df_iris,
        feature_names=feature_names,
        dataset_name="Iris"
    )
    compute_pca(
        x_scaled=df_iris,
        dataset_name="Iris"
    )

    #    Blob Dataset    #
    x_blob = get_blob_dataset()
    blob_feature_names = list(x_blob.columns)
    plot_corr(
        df=x_blob,
        feature_names=blob_feature_names,
        dataset_name="Blob"
    )
    compute_pca(
        x_scaled=x_blob,
        dataset_name="Blob"
    )

    #     Moon Dataset    #
    df_moons = get_moons_dataset()
    moon_feature_name = list(df_moons.columns)
    plot_corr(
        df=df_moons,
        feature_names=moon_feature_name,
        dataset_name="Moon"
    )
    compute_pca(
        x_scaled=df_moons,
        dataset_name="Moon"
    )
    """

    #    Wine Dataset    #
    df_wine = get_wine_dataset()
    wine_feature_names = list(df_wine.columns)
    
    # Standardize data before PCA processing
    standard_wine_df = standardize_data(df_wine)
    
    plot_corr(
        df=df_wine,
        feature_names=wine_feature_names,
        dataset_name="Wine"
    )
    
    wine_pca_3d = compute_pca(
        x_scaled=standard_wine_df,
        dataset_name="Wine"
    )
    
    if wine_pca_3d is not None:
        compare_clustering(
            original_data=standard_wine_df,
            pca_data=wine_pca_3d,
            dataset_name="Wine",
            n_clusters=3
        )