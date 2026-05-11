import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer

data_set = pd.read_csv("../iris.data", header=None)

data_set.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

x = data_set.iloc[:, [0, 1, 2]].values

categories = data_set.columns[[0, 1, 2]]

y = data_set.iloc[:, -1].values

x = np.where(x == '?', np.nan, x)
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

classes = np.unique(y)


def get_t(massive_):
    t_s = []
    for i in range(1, len(massive_)):
        if massive_[i - 1][0] != massive_[i][0]:
            t_s.append((massive_[i - 1][0] + massive_[i][0]) / 2)
    return t_s


def get_p(massive_):
    result = []
    for i in range(len(classes)):
        result.append(0)
    for i in range(len(massive_)):
        for j in range(len(classes)):
            if massive_[i][1] == classes[j]:
                result[j] += 1
    for i in range(len(result)):
        result[i] = result[i] / len(massive_)
    return result


def get_entropy(massive_):
    entropy = 0
    p_s = get_p(massive_)
    for i in range(len(p_s)):
        if p_s[i] != 0:
            entropy -= p_s[i] * math.log(p_s[i], 2)
    return entropy


def get_splitinfo(total, left, right):
    split = 0
    p_left = len(left) / total
    p_right = len(right) / total

    if p_left != 0:
        split -= p_left * math.log2(p_left)

    if p_right != 0:
        split -= p_right * math.log2(p_right)

    return split


for v in range(x.shape[1]):

    massive = list(zip(x[:, v], y))
    massive.sort()

    t_s = get_t(massive)

    best_gain_ratio = 0
    best_t = None

    entropy_total = get_entropy(massive)

    for t in t_s:
        left = [m for m in massive if m[0] <= t]
        right = [m for m in massive if m[0] > t]

        if len(left) == 0 or len(right) == 0:
            continue

        entropy_left = get_entropy(left)
        entropy_right = get_entropy(right)

        weighted_entropy = ((len(left) / len(massive)) * entropy_left +
                            (len(right) / len(massive)) * entropy_right)

        info_gain = entropy_total - weighted_entropy

        split_info = get_splitinfo(len(massive), left, right)

        if split_info == 0:
            continue

        gain_ratio = info_gain / split_info

        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_t = t

    print(f"\nПризнак: {categories[v]}")
    print("Лучший порог:", best_t)
    print("Лучший Gain Ratio:", best_gain_ratio)


