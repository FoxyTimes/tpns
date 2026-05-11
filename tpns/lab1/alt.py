import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("../iris.data", header=None)

data.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

X = data[["sepal_length", "sepal_width", "petal_length"]]
y = data["class"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

info_gain = mutual_info_classif(X, y_encoded, discrete_features=False)

def split_info(feature, n_bins=2):
    binned = pd.cut(feature, bins=n_bins, labels=False, duplicates='drop')
    values, counts = np.unique(binned, return_counts=True)
    probs = counts / len(feature)
    return -np.sum(probs * np.log2(probs))


for feature, ig in zip(X.columns, info_gain):
    si = split_info(X[feature])
    gr = ig / si

    print(f"{feature}")
    print(f"  Gain Ratio:", gr)