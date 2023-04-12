import numpy as np

class NeuralNetworkWith2Layers():
    def __init__(self, layers):
        self.layers = layers


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def initialize_parameters(self):
        parameters = {}
        center = 0
        margin = 1
        
        for i in range(1, len(self.layers)):
            # draw random samples from a normal (Gaussian) distribution
            parameters['W'+str(i)] = np.random.normal(center, margin, size = (self.layers[i], self.layers[i-1]))
            # zero bias vector
            parameters['b' + str(i)] = np.zeros((self.layers[i],1))  
        return parameters  


    # return output of the network from forward calculations
    def forward_model(self, new_a, parameters, L):
        caches = []

        # claculate forward process for each layer
        for l in range(1, L):
            prev_a = new_a 
            # extract weight and biase from the list of parameters
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            # new a is calculated based on the formula, using sigmoid as activation function
            Z = np.dot(W, prev_a).reshape(-1, 1) + b
            new_a = self.sigmoid(Z)
            # cache -> ((a, W, b), z)
            cache = ((prev_a, W, b), Z)

            caches.append(cache)
                
        return new_a, caches 


    # calculate SSE cost 
    def compute_cost(self, output, Y):
        cost = ((output - Y)**2).sum()
        return cost


    # calculate derivation of sigmoid
    def sigmoid_deriv(self, z):
        a = self.sigmoid(z)
        return a * (1 - a)
    

    # create np zeros for all needed gradients based on size of layers
    def create_gradients_zeros(self):
        grad_a1 = np.zeros((self.layers[1], 1))
        grad_W2 = np.zeros((self.layers[2], self.layers[1]))
        grad_W1 = np.zeros((self.layers[1], self.layers[0]))
        grad_b2 = np.zeros((self.layers[2], 1))         
        grad_b1 = np.zeros((self.layers[1], 1))
        
        return grad_a1, grad_W2, grad_W1, grad_b2, grad_b1


    # extract parameters which has been saved during forwardfeeding from the cache
    def extract_parameters(self, caches):
        # cache -> ((a, W, b), z)
        a1 = caches[1][0][0]
        a0 = caches[0][0][0]

        W2 = caches[1][0][1]
        W1 = caches[0][0][1]
        
        z2 = caches[1][1]
        z1 = caches[0][1]
        
        return a1, a0, W2, W1, z2, z1
    
    
    # calculate gradients of wights and biases
    # calculate gradients of wights and biases
    def backpropagation_for_loop(self, caches, output, y):
        a2=output
        a1, a0, W2, W1, z2, z1 = self.extract_parameters(caches)
        grad_a1, grad_W2, grad_W1, grad_b2, grad_b1 = self.create_gradients_zeros()

        # calculat gradients of out put of layer(a)
        for k in range(self.layers[1]):
            for j in range(self.layers[2]):
                grad_a1[k, 0] += W2[j, k] * self.sigmoid_deriv(z2[j, 0]) * (2 * a2[j,0]- 2 * y[j,0])

        
        # calculate gradients of weights
        for j in range(self.layers[2]):
            for k in range(self.layers[1]):
                grad_W2[j, k] += a1[k, 0] * self.sigmoid_deriv(z2[j, 0]) * (2 * a2[j,0]- 2 * y[j,0]) 

        for k in range(self.layers[1]):
            for m in range(self.layers[0]):
                grad_W1[k, m] += a0[m, 0] * self.sigmoid_deriv(z1[k, 0]) * grad_a1[k, 0]

        # calculate gradients of biases
        for j in range(self.layers[2]):
            grad_b2[j ,0] += 1 * self.sigmoid_deriv(z2[j, 0]) * (2 * a2[j,0]- 2 * y[j,0]) 
            
        for k in range(self.layers[1]):
            grad_b1[k ,0] += 1 * self.sigmoid_deriv(z1[k, 0]) * grad_a1[k, 0]
            
        # define a dictionare
        # keys -> label of gradients
        # values -> gradients
        gradients = {}
        gradients["db1"] = grad_b1
        gradients["db2"] = grad_b2
        gradients["dW1"] = grad_W1
        gradients["dW2"] = grad_W2
        
        return gradients


    # apply stochastic gradient descent on input train_set and update weights
    def train(self, train_set, learning_rate, epochs):
        # initialize W and b
        parameters = self.initialize_parameters()
        
        total_costs = []
        layers_len = len(self.layers)
        for i in range(epochs):
            cost = 0
            grad_a1, grad_W2, grad_W1, grad_b2, grad_b1 = self.create_gradients_zeros()
                
            for X, y in train_set:
                y = y.reshape(-1, 1)
                X = X.reshape(-1, 1)
                output, caches = self.forward_model(X, parameters, layers_len)
                gradients = self.backpropagation_for_loop(caches, output, y)

                # extract gradients and add them
                grad_b2 += gradients["db2"]
                grad_b1 += gradients["db1"]
                grad_W2 += gradients["dW2"]
                grad_W1 += gradients["dW1"]

                # cost of this item in batch added to total cost oc this batch
                cost += self.compute_cost(output, y)
                
                # update parameters, weights and biases
                for l in range(layers_len-1):
                    parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (gradients["dW" + str(l+1)]/ len(train_set))
                    parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (gradients["db" + str(l+1)]/ len(train_set))
            
            total_costs.append(cost/len(train_set))
            if i%10==0:
                print(f"EPOCH {i} | Loss : {cost}")
                    
        return parameters, total_costs     



class NeuralNetworkWith3Layers():
    def __init__(self, layers):
        self.layers = layers


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def initialize_parameters(self):
        parameters = {}
        center = 0
        margin = 1
        
        for i in range(1, len(self.layers)):
            # draw random samples from a normal (Gaussian) distribution
            parameters['W'+str(i)] = np.random.normal(center, margin, size = (self.layers[i], self.layers[i-1]))
            # zero bias vector
            parameters['b' + str(i)] = np.zeros((self.layers[i],1))  
        return parameters  


    # return output of the network from forward calculations
    def forward_model(self, new_a, parameters, L):
        caches = []

        # claculate forward process for each layer
        for l in range(1, L):
            prev_a = new_a 
            # extract weight and biase from the list of parameters
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            # new a is calculated based on the formula, using sigmoid as activation function
            Z = np.dot(W, prev_a).reshape(-1, 1) + b
            if l < L-1:
                new_a = self.sigmoid(Z)
            else:
                new_a = Z
            # cache -> ((a, W, b), z)
            cache = ((prev_a, W, b), Z)

            caches.append(cache)
        
                
        return new_a, caches 


    # calculate SSE cost 
    def compute_cost(self, output, Y):
        cost = ((output - Y)**2).sum()
        return cost


    # calculate derivation of sigmoid
    def sigmoid_deriv(self, z):
        a = self.sigmoid(z)
        return a * (1 - a)
    

    # create np zeros for all needed gradients based on size of layers
    def create_gradients_zeros(self):
        grad_a2 = np.zeros((self.layers[2], 1))
        grad_a1 = np.zeros((self.layers[1], 1))
        grad_W3 = np.zeros((self.layers[3], self.layers[2]))
        grad_W2 = np.zeros((self.layers[2], self.layers[1]))
        grad_W1 = np.zeros((self.layers[1], self.layers[0]))
        grad_b3 = np.zeros((self.layers[3], 1))
        grad_b2 = np.zeros((self.layers[2], 1))         
        grad_b1 = np.zeros((self.layers[1], 1))
        
        return grad_a2, grad_a1, grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1


    # extract parameters which has been saved during forwardfeeding from the cache
    def extract_parameters(self, caches):
        # cache -> ((a, W, b), z)
        a2 = caches[2][0][0]
        a1 = caches[1][0][0]
        a0 = caches[0][0][0]

        W3 = caches[2][0][1]
        W2 = caches[1][0][1]
        W1 = caches[0][0][1]
        
        z3 = caches[2][1]
        z2 = caches[1][1]
        z1 = caches[0][1]
        
        return a2, a1, a0, W3, W2, W1, z3, z2, z1
    
    
    # calculate gradients of wights and biases
    # calculate gradients of wights and biases
    def backpropagation_for_loop(self, caches, output, y):
        a3=output
        a2, a1, a0, W3, W2, W1, z3, z2, z1 = self.extract_parameters(caches)
        grad_a2, grad_a1, grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1 = self.create_gradients_zeros()

        # calculat gradients of out put of layer(a)
        for k in range(self.layers[2]):
            for j in range(self.layers[3]):
                grad_a2[k, 0] += W3[j, k] * (2 * a3[j,0]- 2 * y[j,0])

        for m in range(self.layers[1]):
            for k in range(self.layers[2]):
                grad_a1[m, 0] += W2[k, m] * self.sigmoid_deriv(z2[k, 0]) * grad_a2[k, 0]

        # calculate gradients of weights
        for j in range(self.layers[3]):
            for k in range(self.layers[2]):
                grad_W3[j, k] += a2[k, 0] * (2 * a3[j,0]- 2 * y[j,0]) 

        for k in range(self.layers[2]):
            for m in range(self.layers[1]):
                grad_W2[k, m] += a1[m, 0] * self.sigmoid_deriv(z2[k, 0]) * grad_a2[k, 0]
        
        for m in range(self.layers[1]):
            for v in range(self.layers[0]):
                grad_W1[m, v] += a0[v, 0] * self.sigmoid_deriv(z1[m, 0]) * grad_a1[m, 0]

        # calculate gradients of biases
        for j in range(self.layers[3]):
            grad_b3[j ,0] += 1 * (2 * a3[j,0]- 2 * y[j,0]) 
            
        for k in range(self.layers[2]):
            grad_b2[k ,0] += 1 * self.sigmoid_deriv(z2[k, 0]) * grad_a2[k, 0]
                
        for m in range(self.layers[1]):
            grad_b1[m ,0] += 1 * self.sigmoid_deriv(z1[m, 0]) * grad_a1[m, 0]
            
        # define a dictionare
        # keys -> label of gradients
        # values -> gradients
        gradients = {}
        gradients["db1"] = grad_b1
        gradients["db2"] = grad_b2
        gradients["db3"] = grad_b3
        gradients["dW1"] = grad_W1
        gradients["dW2"] = grad_W2
        gradients["dW3"] = grad_W3
        
        return gradients


    # apply stochastic gradient descent on input train_set and update weights
    def train(self, train_set, learning_rate, epochs):
        # initialize W and b
        parameters = self.initialize_parameters()
        
        total_costs = []
        layers_len = len(self.layers)
        for i in range(epochs):
            cost = 0
            grad_a2, grad_a1, grad_W3, grad_W2, grad_W1, grad_b3, grad_b2, grad_b1 = self.create_gradients_zeros()
                
            for X, y in train_set:
                y = y.reshape(-1, 1)
                X = X.reshape(-1, 1)
                output, caches = self.forward_model(X, parameters, layers_len)
                gradients = self.backpropagation_for_loop(caches, output, y)

                # extract gradients and add them
                grad_b3 += gradients["db3"]
                grad_b2 += gradients["db2"]
                grad_b1 += gradients["db1"]
                grad_W3 += gradients["dW3"]
                grad_W2 += gradients["dW2"]
                grad_W1 += gradients["dW1"]

                # cost of this item in batch added to total cost oc this batch
                cost += self.compute_cost(output, y)/len(train_set)
                
                # update parameters, weights and biases
                for l in range(layers_len-1):
                    parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (gradients["dW" + str(l+1)]/ len(train_set))
                    parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (gradients["db" + str(l+1)]/ len(train_set))
            
            total_costs.append(cost)
            if i%10==0:
                print(f"EPOCH {i} | Loss : {cost}")
                    
        return parameters, total_costs

    def test(self, test_set, parameters):
        test_cost = 0
        predictions = []
        for Xtest, ytest in test_set:
            yp, _ = self.forward_model(Xtest, parameters, len(self.layers))
            predictions.append(yp)
            test_cost += self.compute_cost(yp, ytest)/len(test_set)
        return predictions, test_cost

