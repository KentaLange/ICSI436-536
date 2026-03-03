import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def standardize_data(dataset):
    x_standard_data = scaler.fit_transform(dataset.values)
    print("Scaled mean (approx 0):", np.round(x_standard_data.mean(axis=0), 3))
    print("Scaled std  (approx 1):", np.round(x_standard_data.std(axis=0), 3))
    return x_standard_data
