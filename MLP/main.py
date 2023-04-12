from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
from MLP import NeuralNetworkWith2Layers,NeuralNetworkWith3Layers

# Modelling XOR gate with a 2 layered ANN

#Train data
X = np.array([[1, 1, 0], [1, 0, 1],[0,0,0],[1, 1, 1],  [0, 1, 0],[0, 1, 1], [1, 0, 0],[0, 0,1]])
y = np.array([[0], [0], [0],[1], [1], [0], [1],[1]]).reshape(-1, 1)
train_set = list(zip(X, y))
Xtest = np.array([[0, 0,0], [1, 1, 1], [0, 0, 1]])
ytest = np.array([[0], [1], [1]]).reshape(-1, 1)
test_set = list(zip(Xtest, ytest))

model_xor = NeuralNetworkWith2Layers([3, 5, 1]) # Model 
parameters, total_costs = model_xor.train(train_set, 0.9, 10000) # Training

# Testing
for X, y in test_set:
    X = X.reshape(-1, 1)
    yp, _ = model_xor.forward_model(X, parameters, 3)
    print(f"Prediction: {yp} | True Value: {y}")

# California Housing Dataset prediction using 
'''data, target = fetch_california_housing(return_X_y=True)

data, target = shuffle(data, target, random_state=124)

data = data[:1000]
target = target[:1000]
# Splitting data into train and test
train_data, test_data, train_target,test_target = train_test_split(np.array(data), target, test_size=0.2)
train_set = list(zip(train_data, train_target))
test_set = list(zip(test_data, test_target))
model = NeuralNetworkWith3Layers([8, 12, 12, 1])
parameters, total_costs = model.train(train_set, 0.05, 40)
predictions, test_cost = model.test(test_set, parameters)
print(f"Test cost: {test_cost}")'''