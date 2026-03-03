import pandas as pd
from sklearn.datasets import load_iris, make_blobs, make_moons, load_wine

def get_iris_data_set():
    # Real dataset: Iris
    iris = load_iris()
    x_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    print("IRIS shape:", x_iris.shape)        # (samples, features)
    print(x_iris.head())                      # first few rows
    return x_iris, iris

def get_blob_dataset():
    # Synthetic dataset: blobs
    x_blobs, _ = make_blobs(n_samples=1000, n_features=6, centers=4, random_state=42)
    x_blobs = pd.DataFrame(x_blobs, columns=[f"f{i}" for i in range(x_blobs.shape[1])])
    print("\nBLOBS shape:", x_blobs.shape)
    print(x_blobs.head())
    return x_blobs

def get_moons_dataset():
    # Synthetic dataset: moons (2 features only)
    x_moons, _ = make_moons(n_samples=400, noise=0.08, random_state=42)
    x_moons = pd.DataFrame(x_moons, columns=["x1", "x2"])
    print("\nMOONS shape:", x_moons.shape)
    print(x_moons.head())
    return x_moons

def get_wine_dataset():
    wine = load_wine()
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    print("\nWine Shape:", df_wine.shape)
    print(df_wine.head())
    return df_wine