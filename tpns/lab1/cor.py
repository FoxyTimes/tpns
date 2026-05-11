import pandas as pd

data_set = pd.read_csv("../iris.data", header=None)

data_set.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

data = data_set.iloc[:, 0:4].values.tolist()


def get_avg_col(num):
    sum_col = 0
    i = 0
    while len(data) != i:
        sum_col += float(data[i][num])
        i += 1
    return sum_col / i


def cor(x, y):
    x_ = get_avg_col(x)
    y_ = get_avg_col(y)

    sum_chisl = 0
    sum_znam_x = 0
    sum_znam_y = 0

    for i in range(0, len(data)):
        x_i = float(data[i][x])
        y_i = float(data[i][y])

        sum_chisl += (x_i - x_) * (y_i - y_)
        sum_znam_x += (x_i - x_) ** 2
        sum_znam_y += (y_i - y_) ** 2

    return sum_chisl / (sum_znam_x ** 0.5 * sum_znam_y ** 0.5)


thermal_matrix = []

for i in range(0, len(data[0])):
    thermal_matrix.append([])
    for j in range(0, len(data[0])):
        thermal_matrix[i].append(cor(i, j))


feature_names = data_set.columns[:4]

print("Корреляционная матрица:\n")

for i in range(len(thermal_matrix)):
    for j in range(len(thermal_matrix[i])):
        print(f"{thermal_matrix[i][j]:.6f}", end=" ")
    print()