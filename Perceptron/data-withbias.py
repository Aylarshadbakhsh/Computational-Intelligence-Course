import numpy as np
from Perceptron1 import Perceptron
bias=-np.ones((10,1))
data = np.array([[2.7810836, 2.550537003],
                [1.465489372, 2.362125076],
                [3.396561688, 4.400293529],
                [1.38807019, 1.850220317],
                [3.06407232, 3.005305973],
                [7.627531214, 2.759262235],
                [5.332441248, 2.088626775],
                [6.922596716, 1.77106367,],
                [8.675418651, -0.242068655],
                [7.673756466, 3.508563011]])
data=np.append(data, bias, axis=1)
label = [-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]

model = Perceptron(input_size=3, eta=0.1, e_max=0)
model.train(1000, data, label)

test_data = np.array([[3.396561688, 4.400293529],
                    [1.38807019, 1.850220317],
                    [3.06407232, 3.005305973],
                    [7.627531214, 2.759262235],
                    [5.332441248, 2.088626775]])
bias=-np.ones((5,1))
test_data=np.append(test_data, bias, axis=1)
test_label = [-1, -1, -1, 1, 1]
for data, label in zip(test_data, test_label):
    print(model.predict(data), label)

'''#AND
data2 = np.array([[1, 1, -1],
                [1, 0, -1],
                [0, 1, -1],
                [0, 0, -1]])
label2 = [1, -1, -1, -1, ]

model = Perceptron(input_size=3, eta=0.1, e_max=0)
model.train(3000, data2, label2)
test_data2 = data2
test_label2 = label2

for data2, label2 in zip(test_data2, test_label2):
    print(model.predict(data2), label2)
#OR
data2 = np.array([[1, 1, -1],
                    [1, 0, -1],
                    [0, 1, -1],
                    [0, 0, -1]])
label2 = [1, 1, 1, -1, ]

model = Perceptron(input_size=3, eta=0.1, e_max=0)
model.train(3000, data2, label2)
test_data2 = data2
test_label2 = label2

for data2, label2 in zip(test_data2, test_label2):
    print(model.predict(data2), label2)
#XOR
data2 = np.array([[1, 1, -1],
                    [1, 0, -1],
                    [0, 1, -1],
                    [0, 0, -1]])
label2 = [1, -1, -1, 1, ]

model = Perceptron(input_size=3, eta=0.1, e_max=0)
model.train(3000, data2, label2)
test_data2 = data2
test_label2 = label2

for data2, label2 in zip(test_data2, test_label2):
    print(model.predict(data2), label2)'''