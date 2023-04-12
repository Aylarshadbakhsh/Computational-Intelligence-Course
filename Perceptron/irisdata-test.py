import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Perceptron1 import Perceptron
df = pd.read_csv("iris.data")
# Four Inputs: sepal length, sepal width, petal length, petal width
inputs= df.iloc[0:120, [0, 1, 2, 3]].values
# Three classes: Iris-setosa, Iris-versicolor, Iris-virginica
original_classes = df.iloc[0:120, 4].values
# Convert the three Classes into two classes iris-setosa=1 , iris-virginica and iris-versicolor=-1
classes= np.where(original_classes == "Iris-setosa", 1,-1)
bias = -np.ones((120, 1))
data=np.append(inputs, bias, axis=1)
label = classes
model = Perceptron(input_size=5, eta=0.1, e_max=0)
model.train(1000, data, label)
test_data=df.iloc[30:130, [0, 1, 2, 3]].values
original_classes2 = df.iloc[30:130, 4].values
bias = -np.ones((100, 1))
test_data = np.append(test_data, bias, axis=1)
test_label = np.where(original_classes2 == "Iris-setosa", 1, -1)
for data, label in zip(test_data, test_label):
    print(model.predict(data), label)