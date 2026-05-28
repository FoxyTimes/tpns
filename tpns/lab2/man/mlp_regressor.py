import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MultilayerPerceptronRegressor:
    def __init__(self, layer_sizes, learning_rate=0.5, max_epochs=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            scale = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(float)

    def _forward(self, X):
        activations = [X]
        a = X

        for i in range(self.num_layers - 2):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
            activations.append(a)

        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        activations.append(z_out)

        return activations

    def _backprop(self, y, activations):
        num_samples = y.shape[0]
        deltas = [None] * (self.num_layers - 1)

        delta_output = (2 * (activations[-1] - y.reshape(-1, 1))) / num_samples
        deltas[-1] = delta_output

        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[i], self.weights[i].T) * self.relu_derivative(activations[i])
            deltas[i - 1] = delta

        for i in range(self.num_layers - 1):
            grad_w = np.dot(activations[i].T, deltas[i]) / num_samples
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / num_samples

            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def fit(self, x, y):
        y = y.astype(float)
        batch_size = random.randint(0, 64)

        for epoch in range(self.max_epochs):


            indices = np.random.permutation(x.shape[0])
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for start_batch in range(0, x.shape[0], batch_size):
                end_batch = min(start_batch + batch_size, x.shape[0])

                X = x_shuffled[start_batch:end_batch]
                Y = y_shuffled[start_batch:end_batch]

                activations = self._forward(X)
                self._backprop(Y, activations)

            if epoch % 100 == 0:
                predictions = activations[-1]
                mse = np.mean((predictions.flatten() - Y) ** 2)
                rmse = np.sqrt(mse)
                print(f"Эпоха {epoch:4d}: MSE = {mse:.4f}, RMSE = {rmse:.4f}")

        return self

    def predict(self, X):
        activations = self._forward(X)
        return activations[-1].flatten()


def load_laptop_data():
    data = pd.read_csv("../Laptop_price.csv")
    return data


def main():
    data = load_laptop_data()

    brand_dummies = pd.get_dummies(data['Brand'], prefix='Brand')

    X = pd.concat([data.drop(['Brand', 'Price'], axis=1), brand_dummies], axis=1)
    y = data['Price'].values

    print(f"\nПризнаки: {list(X.columns)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_train_scaled, test_size=0.15, random_state=42
    )

    print(f"\nОбучающая выборка: {X_train.shape[0]}")
    print(f"Тестовая выборка: {X_test.shape[0]}\n")

    mlp = MultilayerPerceptronRegressor(
        layer_sizes=[X_train.shape[1], 8, 1],
        learning_rate=0.5,
        max_epochs=1000
    )

    mlp.fit(X_train, y_train)

    train_pred = mlp.predict(X_train)
    test_pred = mlp.predict(X_test)

    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))

    print(f"\nRMSE на обучении: {train_rmse * scaler_y.scale_[0]:.2f}")
    print(f"RMSE на тесте: {test_rmse * scaler_y.scale_[0]:.2f}")


    test_pred_original = scaler_y.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    print(f"\n{'#':<4} {'Предсказано':>12} {'Реальное':>12} {'Ошибка':>12}")

    for i in range(len(test_pred_original)):
        error = abs(test_pred_original[i] - y_test_original[i])
        print(f"{i + 1:<4} {test_pred_original[i]:>12.2f} {y_test_original[i]:>12.2f} {error:>12.2f}")

    print(f"Всего тестовых примеров: {len(test_pred_original)}")
    print(f"Средняя ошибка: {np.mean(np.abs(test_pred_original - y_test_original)):.2f}")
    print(f"Макс. ошибка: {np.max(np.abs(test_pred_original - y_test_original)):.2f}")


if __name__ == "__main__":
    main()

