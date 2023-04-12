import numpy as np
import matplotlib.pyplot as plt
from rbf import RBFRegressor

# Data
NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = np.sin(2 * np.pi * X)  + noise

# # Model
rbfnet = RBFRegressor(lr=1e-2, k=2)
# Train
rbfnet.fit(X, y)
# Prediction
y_pred = rbfnet.predict(X)
# Plotting
plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBF-Net')
plt.legend()
plt.tight_layout()
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Sinusoidal Wave Estimation with RBF Regressor")
plt.show()
