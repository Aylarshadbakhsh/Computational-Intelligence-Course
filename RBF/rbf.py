import numpy as np
import matplotlib.pyplot as plt
from kmean import KMeans


# class RBFRegressor(object):
#     def __init__(self, k, lr=0.01, epochs=100, inferStds=True):
#         self.k = k
#         self.lr = lr
#         self.epochs = epochs
#         self.km = KMeans(K=2, max_iters=150)
#         self.inferStds = inferStds
#         self.w = np.random.randn(k)
#         self.b = np.random.randn(1)
    
#     @staticmethod
#     def rbf(x, c, s):
#         return np.exp(-1 / (2 * s**2) * (x-c)**2)

#     def fit(self, X, y):
#         self.centers = self.km.predict(np.reshape(X, (-1, 1)))
#         self.stds = self.km.stddv()
#         self.stds = np.array(self.stds)
#         self.centers = self.centers.flatten()
#         self.stds = self.stds.flatten()
 
#         # training
#         for epoch in range(self.epochs):
#             for i in range(X.shape[0]):
#                 a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
#                 F = a.T.dot(self.w) + self.b
#                 error = -(y[i] - F).flatten()
#                 self.w = self.w - self.lr * a * error
#                 self.b = self.b - self.lr * error
                
#     def predict(self, X):
#         y_pred = []
#         for i in range(X.shape[0]):
#             a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
#             F = a.T.dot(self.w) + self.b
#             y_pred.append(F)
#         return np.array(y_pred)




class RBFClassifier(object):
    def __init__(self, k, lr, epochs, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.inferStds = inferStds
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
        self.km = KMeans(K=2, max_iters=150)

    @staticmethod
    def sigmoid(x):
        return 2 * (1 / (1 + np.exp(-x))) - 1
    
    @staticmethod
    def rbf(x, c, s):
        return np.exp(np.sum(-1 / (2 * s**2) * (x-c)**2))

    def fit(self, X, y):
        self.centers = self.km.predict(X)
        self.stds = self.km.stddv()

        # training
        loss=0
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                F = self.sigmoid(F)
                loss = ((y[i] - F).flatten() ** 2)
                # backward pass
                error = -(y[i] - F).flatten()
                # online update
                self.w =self.w- (0.5 * self.lr *error *a* ((1 -F) ** 2))
                self.b = self.b - (0.5*self.lr * error)

            if epoch%10==0:
             print(f"EPOCH {epoch} | Loss : {loss}")

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)




class RBFRegressor(object):
    def __init__(self, k, lr=0.01, epochs=100, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.inferStds = inferStds
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
        self.km = KMeans(K=2, max_iters=150)

    @staticmethod
    def rbf(x, c, s):
        return np.exp(-1 / (2 * s**2) * (x-c)**2)

    def fit(self, X, y):
        self.centers = self.km.predict(np.reshape(X, (-1, 1)))
        self.stds = self.km.stddv()
        self.stds = np.array(self.stds)
        self.centers = self.centers.flatten()
        self.stds = self.stds.flatten()

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b
                loss = (y[i] - F).flatten() ** 2
                error = -(y[i] - F).flatten()
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
    
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)