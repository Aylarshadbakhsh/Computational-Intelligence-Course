import numpy as np


class Perceptron:
    def __init__(self, input_size, eta, e_max) -> None:
        self.input_size = input_size
        self.eta = eta
        self.e_max = e_max
    def sigmoid(self, x):
        return 2 * (1 / (1 + np.exp(-x))) - 1

    def initialize_random_weight(self, ):
        self.weights = np.random.rand(1, self.input_size)

    def update_weights(self, yp, yt, x):
        self.weights += (0.5 * self.eta * (yt - yp.item()) * (1 - yp.item() ** 2) * (x))

    def predict(self, data):
        y = data
        return self.sigmoid(np.dot(self.weights, y))

    def train(self, epochs, data, label):
        self.initialize_random_weight()

        for e in range(epochs):
            error = 0
            for k in range(len(data)):
                y = self.predict(data[k])
                error += 0.5 * (label[k] - y) ** 2
                self.update_weights(y, label[k], data[k])
            print(f"Epoch #{e + 1}: Error: {error}")
            if error < self.e_max:
                break
        print("Traning process finished successfully.")