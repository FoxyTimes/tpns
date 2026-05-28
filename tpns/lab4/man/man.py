import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        scale = np.sqrt(6.0 / (kernel_size * kernel_size * in_channels + out_channels))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.biases = np.zeros((out_channels,))

        self.input_cache = None
        self.output_shape = None

    def forward(self, X):
        batch_size, in_channels, height, width = X.shape
        out_channels, _, kernel_size, _ = self.weights.shape

        out_height = height - kernel_size + 1
        out_width = width - kernel_size + 1

        output = np.zeros((batch_size, out_channels, out_height, out_width))

        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        for ic in range(in_channels):
                            patch = X[b, ic, i:i+kernel_size, j:j+kernel_size]
                            kernel = self.weights[oc, ic]
                            output[b, oc, i, j] += np.sum(patch * kernel)
                output[b, oc] += self.biases[oc]

        self.input_cache = X
        self.output_shape = output.shape
        return output

    def backward(self, dL_dout):
        batch_size, out_channels, out_height, out_width = dL_dout.shape
        in_channels, _, kernel_size, _ = self.weights.shape
        batch_size, in_channels, height, width = self.input_cache.shape

        dL_dweights = np.zeros_like(self.weights)
        dL_dbiases = np.zeros_like(self.biases)
        dL_dinput = np.zeros_like(self.input_cache)

        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        for ic in range(in_channels):
                            patch = self.input_cache[b, ic, i:i+kernel_size, j:j+kernel_size]
                            dL_dweights[oc, ic] += patch * dL_dout[b, oc, i, j]
                            dL_dinput[b, ic, i:i+kernel_size, j:j+kernel_size] += \
                                self.weights[oc, ic] * dL_dout[b, oc, i, j]
                dL_dbiases[oc] += np.sum(dL_dout[b, oc])

        dL_dweights /= batch_size
        dL_dbiases /= batch_size

        return dL_dinput, dL_dweights, dL_dbiases


class AvgPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input_cache = None
        self.output_shape = None

    def forward(self, X):
        batch_size, channels, height, width = X.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        pool_region = X[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.mean(pool_region)

        self.input_cache = X
        self.output_shape = output.shape
        return output

    def backward(self, dL_dout):
        batch_size, channels, out_height, out_width = dL_dout.shape
        batch_size, channels, height, width = self.input_cache.shape

        dL_dinput = np.zeros_like(self.input_cache)
        pool_area = self.pool_size * self.pool_size

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        dL_dinput[b, c, h_start:h_end, w_start:w_end] += \
                            dL_dout[b, c, i, j] / pool_area

        return dL_dinput


class LeNet5:
    def __init__(self, learning_rate=0.01, optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.conv1 = Conv2D(1, 6, 5)
        self.pool1 = AvgPool2D(2, 2)

        self.conv2 = Conv2D(6, 16, 5)
        self.pool2 = AvgPool2D(2, 2)

        self.fc1_weights = np.random.randn(16 * 4 * 4, 120) * 0.01
        self.fc1_biases = np.zeros((120,))

        self.fc2_weights = np.random.randn(120, 84) * 0.01
        self.fc2_biases = np.zeros((84,))

        self.fc3_weights = np.random.randn(84, 10) * 0.01
        self.fc3_biases = np.zeros((10,))

        self.cache = {}
        self.optimizer_state = {}
        self._init_optimizer_state()

    def _named_parameters(self):
        return {
            'conv1_weights': self.conv1.weights,
            'conv1_biases': self.conv1.biases,
            'conv2_weights': self.conv2.weights,
            'conv2_biases': self.conv2.biases,
            'fc1_weights': self.fc1_weights,
            'fc1_biases': self.fc1_biases,
            'fc2_weights': self.fc2_weights,
            'fc2_biases': self.fc2_biases,
            'fc3_weights': self.fc3_weights,
            'fc3_biases': self.fc3_biases,
        }

    def _init_optimizer_state(self):
        self.optimizer_state = {}
        for name, param in self._named_parameters().items():
            self.optimizer_state[name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param),
            }

    def _apply_gradients(self, grads):
        params = self._named_parameters()

        if self.optimizer == "adam":
            self.t += 1

        for name, grad in grads.items():
            param = params[name]

            if self.optimizer == "adam":
                state = self.optimizer_state[name]
                state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * grad
                state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (grad ** 2)

                m_hat = state['m'] / (1 - self.beta1 ** self.t)
                v_hat = state['v'] / (1 - self.beta2 ** self.t)
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                param -= self.learning_rate * grad

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(float)

    @staticmethod
    def softmax(z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    def forward(self, X):
        conv1_out = self.conv1.forward(X)
        relu1_out = self.relu(conv1_out)
        pool1_out = self.pool1.forward(relu1_out)

        conv2_out = self.conv2.forward(pool1_out)
        relu2_out = self.relu(conv2_out)
        pool2_out = self.pool2.forward(relu2_out)

        batch_size = pool2_out.shape[0]
        flattened = pool2_out.reshape(batch_size, -1)

        fc1_out = np.dot(flattened, self.fc1_weights) + self.fc1_biases
        relu3_out = self.relu(fc1_out)

        fc2_out = np.dot(relu3_out, self.fc2_weights) + self.fc2_biases
        relu4_out = self.relu(fc2_out)

        fc3_out = np.dot(relu4_out, self.fc3_weights) + self.fc3_biases
        output = self.softmax(fc3_out)

        self.cache = {
            'X': X,
            'conv1_out': conv1_out,
            'relu1_out': relu1_out,
            'pool1_out': pool1_out,
            'conv2_out': conv2_out,
            'relu2_out': relu2_out,
            'pool2_out': pool2_out,
            'flattened': flattened,
            'fc1_out': fc1_out,
            'relu3_out': relu3_out,
            'fc2_out': fc2_out,
            'relu4_out': relu4_out,
            'fc3_out': fc3_out,
            'output': output
        }

        return output

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        grads = {}

        output = self.cache['output']
        dL_dfc3_out = output - y_true

        relu4_out = self.cache['relu4_out']
        dL_dfc3_weights = np.dot(relu4_out.T, dL_dfc3_out) / batch_size
        dL_dfc3_biases = np.sum(dL_dfc3_out, axis=0) / batch_size
        dL_drelu4_out = np.dot(dL_dfc3_out, self.fc3_weights.T)

        grads['fc3_weights'] = dL_dfc3_weights
        grads['fc3_biases'] = dL_dfc3_biases

        dL_dfc2_out = dL_drelu4_out * self.relu_derivative(self.cache['fc2_out'])
        relu3_out = self.cache['relu3_out']
        dL_dfc2_weights = np.dot(relu3_out.T, dL_dfc2_out) / batch_size
        dL_dfc2_biases = np.sum(dL_dfc2_out, axis=0) / batch_size
        dL_drelu3_out = np.dot(dL_dfc2_out, self.fc2_weights.T)

        grads['fc2_weights'] = dL_dfc2_weights
        grads['fc2_biases'] = dL_dfc2_biases

        dL_dfc1_out = dL_drelu3_out * self.relu_derivative(self.cache['fc1_out'])
        flattened = self.cache['flattened']
        dL_dfc1_weights = np.dot(flattened.T, dL_dfc1_out) / batch_size
        dL_dfc1_biases = np.sum(dL_dfc1_out, axis=0) / batch_size
        dL_dflattened = np.dot(dL_dfc1_out, self.fc1_weights.T)

        grads['fc1_weights'] = dL_dfc1_weights
        grads['fc1_biases'] = dL_dfc1_biases

        pool2_out_shape = self.cache['pool2_out'].shape
        dL_dpool2_out = dL_dflattened.reshape(pool2_out_shape)

        dL_drelu2_out = self.pool2.backward(dL_dpool2_out)

        dL_dconv2_out = dL_drelu2_out * self.relu_derivative(self.cache['conv2_out'])
        dL_dpool1_out, dL_dconv2_weights, dL_dconv2_biases = self.conv2.backward(dL_dconv2_out)
        grads['conv2_weights'] = dL_dconv2_weights
        grads['conv2_biases'] = dL_dconv2_biases

        dL_drelu1_out = self.pool1.backward(dL_dpool1_out)

        dL_dconv1_out = dL_drelu1_out * self.relu_derivative(self.cache['conv1_out'])
        _, dL_dconv1_weights, dL_dconv1_biases = self.conv1.backward(dL_dconv1_out)
        grads['conv1_weights'] = dL_dconv1_weights
        grads['conv1_biases'] = dL_dconv1_biases

        self._apply_gradients(grads)

    def train_epoch(self, X_train, y_train, batch_size=32):
        num_samples = X_train.shape[0]
        indices = np.random.permutation(num_samples)

        total_loss = 0
        num_batches = 0

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            output = self.forward(X_batch)

            loss = -np.mean(np.sum(y_batch * np.log(output + 1e-8), axis=1))
            total_loss += loss
            num_batches += 1
            self.backward(y_batch)

        return total_loss / num_batches

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == y_test_labels)
        return accuracy

    def save_weights(self, filepath):
        weights_dict = {
            'conv1_weights': self.conv1.weights,
            'conv1_biases': self.conv1.biases,
            'conv2_weights': self.conv2.weights,
            'conv2_biases': self.conv2.biases,
            'fc1_weights': self.fc1_weights,
            'fc1_biases': self.fc1_biases,
            'fc2_weights': self.fc2_weights,
            'fc2_biases': self.fc2_biases,
            'fc3_weights': self.fc3_weights,
            'fc3_biases': self.fc3_biases,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            't': self.t,
            'optimizer_state': self.optimizer_state,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights_dict, f)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights_dict = pickle.load(f)

        self.conv1.weights = weights_dict['conv1_weights']
        self.conv1.biases = weights_dict['conv1_biases']
        self.conv2.weights = weights_dict['conv2_weights']
        self.conv2.biases = weights_dict['conv2_biases']
        self.fc1_weights = weights_dict['fc1_weights']
        self.fc1_biases = weights_dict['fc1_biases']
        self.fc2_weights = weights_dict['fc2_weights']
        self.fc2_biases = weights_dict['fc2_biases']
        self.fc3_weights = weights_dict['fc3_weights']
        self.fc3_biases = weights_dict['fc3_biases']
        self.learning_rate = weights_dict.get('learning_rate', self.learning_rate)
        self.optimizer = weights_dict.get('optimizer', self.optimizer)
        self.beta1 = weights_dict.get('beta1', self.beta1)
        self.beta2 = weights_dict.get('beta2', self.beta2)
        self.eps = weights_dict.get('eps', self.eps)
        self.t = weights_dict.get('t', 0)

        self._init_optimizer_state()
        loaded_state = weights_dict.get('optimizer_state')
        if isinstance(loaded_state, dict):
            for name in self.optimizer_state:
                if name in loaded_state and 'm' in loaded_state[name] and 'v' in loaded_state[name]:
                    if loaded_state[name]['m'].shape == self.optimizer_state[name]['m'].shape and \
                       loaded_state[name]['v'].shape == self.optimizer_state[name]['v'].shape:
                        self.optimizer_state[name] = loaded_state[name]
        print(f"Weights loaded from {filepath}")


def load_mnist_data():
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("Please install torchvision: pip install torchvision")
        return None, None, None, None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                               transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                              transform=transform, download=True)

    X_train = np.array([img.numpy() for img, _ in train_dataset])
    y_train_labels = np.array([label for _, label in train_dataset])

    X_test = np.array([img.numpy() for img, _ in test_dataset])
    y_test_labels = np.array([label for _, label in test_dataset])

    y_train = np.eye(10)[y_train_labels]
    y_test = np.eye(10)[y_test_labels]

    return X_train, y_train, X_test, y_test


def main():
    import os

    print("Loading MNIST data")
    X_train, y_train, X_test, y_test = load_mnist_data()

    if X_train is None:
        return

    X_train = X_train[:2000]
    y_train = y_train[:2000]

    X_test = X_test[:200]
    y_test = y_test[:200]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    model = LeNet5(learning_rate=0.001)
    weights_file = "lenet5_weights.pkl"

    if os.path.exists(weights_file):
        print(f"\nLoading pre-trained weights from {weights_file}...")
        model.load_weights(weights_file)

        print("\nEvaluating pre-trained model on test set...")
        predictions = model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == y_test_labels)
        print(f"Test Accuracy: {accuracy:.4f}")
    else:
        print("\nNo pre-trained weights found. Training new model...")
        num_epochs = 5
        batch_size = 32

        print("\nTraining...")
        for epoch in range(num_epochs):
            loss = model.train_epoch(X_train, y_train, batch_size)
            accuracy = model.evaluate(X_test[:1000], y_test[:1000])
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        print(f"\nSaving weights to {weights_file}...")
        model.save_weights(weights_file)

        print("\nEvaluating on full test set...")
        predictions = model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == y_test_labels)
        print(f"Test Accuracy: {accuracy:.4f}")

    y_test_labels = np.argmax(y_test, axis=1)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test_labels, predictions)
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - LeNet5 (NumPy)")
    plt.show()


if __name__ == "__main__":
    main()

