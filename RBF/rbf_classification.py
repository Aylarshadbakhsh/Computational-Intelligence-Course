import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from rbf import RBFClassifier
from sklearn.datasets import make_blobs, make_gaussian_quantiles

# Data
X, y = make_blobs(centers=2, n_samples=500, n_features=2, shuffle=True, random_state=40)
XX,YY=np.split(X, 2, axis=1)
colors = ['red','blue']
plt.figure()
plt.scatter(XX, YY, c=y, cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel("Feature No. 1")
plt.ylabel("Feature No. 2")
plt.title('Labeled Data')
plt.show()

# Model
rbfnet = RBFClassifier(lr=0.01, k=2,epochs=500)
# Train
rbfnet.fit(X, y)
# Prediction
y_pred = rbfnet.predict(X)
# Plots
y_pred=y_pred.flatten()
fig = plt.figure(figsize=(8,8))
plt.scatter(XX, YY, c=y_pred, cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel("Feature No. 1")
plt.ylabel("Feature No. 2")
plt.title('classified with RBF Classifier')
plt.show()

# Gaussian Distribution
# Data
num_samples_total = 1000
gaussian_mean = (0, 0)
num_classes_total = 2
num_features_total = 2
X, y = make_gaussian_quantiles(n_features=num_features_total, n_classes=num_classes_total, n_samples=num_samples_total, mean=gaussian_mean)
XX,YY = np.split(X,2,axis=1)
colors = ['red','blue']
plt.figure()
plt.scatter(XX, YY, c=y, cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel("Feature No. 1")
plt.ylabel("Feature No. 2")
plt.title('Labeled Data')
plt.show()

# Model
rbfnet = RBFClassifier(lr=0.1, k=2, epochs=100)
# Train
rbfnet.fit(X, y)
# Prediciton
y_pred = rbfnet.predict(X)
# Plots
y_pred=y_pred.flatten()
fig = plt.figure(figsize=(8,8))
plt.scatter(XX, YY, c=y_pred, cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel("Feature No. 1")
plt.ylabel("Feature No. 2")
plt.title('classified with RBF Classifier')
plt.show()
