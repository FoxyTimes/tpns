import numpy as np

from sklearn.model_selection import train_test_split


class MultilayerPerceptron:
    def __init__(self, layer_sizes, learning_rate=0.1, max_epochs=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.num_layers = len(layer_sizes)

        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, X):
        activations = [X]
        a = X

        for i in range(self.num_layers - 2):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)

        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        a_out = self.softmax(z_out)
        activations.append(a_out)

        return activations

    def _backprop(self, y_onehot, activations):
        num_samples = y_onehot.shape[0]
        deltas = [None] * (self.num_layers - 1)

        delta_output = (activations[-1] - y_onehot) / num_samples
        deltas[-1] = delta_output

        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[i], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            deltas[i - 1] = delta

        for i in range(self.num_layers - 1):
            grad_w = np.dot(activations[i].T, deltas[i]) / num_samples
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / num_samples

            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

    def fit(self, x, y):
        num_classes = len(np.unique(y))
        y_onehot = np.eye(num_classes)[y]
        batch_size = 64

        for epoch in range(self.max_epochs):
            indices = np.random.permutation(x.shape[0])
            x_shuffled = x[indices]
            y_shuffled = y_onehot[indices]

            for start_batch in range(0, x.shape[0], batch_size):
                end_batch = min(start_batch + batch_size, x.shape[0])

                X = x_shuffled[start_batch:end_batch]
                Y = y_shuffled[start_batch:end_batch]

                activations = self._forward(X)
                self._backprop(Y, activations)


            if epoch % 100 == 0:
                accuracy = self.evaluate(x, y)
                print(f"Эпоха {epoch:4d}: Accuracy = {accuracy:.4f}")

        return self

    def predict_proba(self, X):
        activations = self._forward(X)
        return activations[-1]

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


def load_iris_data():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    return X, y, class_names


def main():
    print("Многослойный персептрон - классификация Iris")

    X, y, class_names = load_iris_data()

    print(f"\nДатасет: Iris")
    print(f"Количество образцов: {X.shape[0]}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Классы: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nОбучающая выборка: {X_train.shape[0]} образцов")
    print(f"Тестовая выборка: {X_test.shape[0]} образцов\n")

    mlp = MultilayerPerceptron(
        layer_sizes=[4, 64, 3],
        learning_rate=0.5,
        max_epochs=400
    )

    mlp.fit(X_train, y_train)

    print("\nРезультаты")

    train_accuracy = mlp.evaluate(X_train, y_train)
    test_accuracy = mlp.evaluate(X_test, y_test)

    print(f"\nТочность на обучающей выборке: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Точность на тестовой выборке:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")



if __name__ == "__main__":
    main()
