import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
np.random.seed(1)

class One_Hidden_Layer_Network:
    # Loading the data at initialization
    def __init__(self):
        self.X, self.Y = load_planar_dataset()
    # Check the functions and sanity checks
    def check(self):
        # Visualize the data:
        plt.scatter(self.X[0, :], self.X[1, :], c=self.Y.ravel(), s=40, cmap=plt.cm.Spectral)
        plt.show()
        print("Shape X  =" + str(self.X.shape) + ", Shape Y = " + str(self.Y.shape))
        X_assess, Y_assess = layer_sizes_test_case()
        (n_x, n_h, n_y) = self.layer_sizes(X_assess, Y_assess)
        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the hidden layer is: n_h = " + str(n_h))
        print("The size of the output layer is: n_y = " + str(n_y))

        n_x, n_h, n_y = initialize_parameters_test_case()
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))

        X_assess, parameters = forward_propagation_test_case()

        A2, cache = self.forward_propagation(X_assess, parameters)
        print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
        A2, Y_assess, parameters = compute_cost_test_case()
        print("cost = " + str(self.compute_cost(A2, Y_assess, parameters)))
        parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
        grads = self.backward_propagation(parameters, cache, X_assess, Y_assess)
        print ("dW1 = "+ str(grads["dW1"]))
        print ("db1 = "+ str(grads["db1"]))
        print ("dW2 = "+ str(grads["dW2"]))
        print ("db2 = "+ str(grads["db2"]))
        parameters, grads = update_parameters_test_case()
        parameters = self.update_parameters(parameters, grads)
        print("updated W1 = " + str(parameters["W1"]))
        print("updated b1 = " + str(parameters["b1"]))
        print("updated W2 = " + str(parameters["W2"]))
        print("updated b2 = " + str(parameters["b2"]))
        X_assess, Y_assess = nn_model_test_case()
        parameters = self.model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
        print("W1 = " + str(parameters["W1"]))
        print("b1 = " + str(parameters["b1"]))
        print("W2 = " + str(parameters["W2"]))
        print("b2 = " + str(parameters["b2"]))
        parameters, X_assess = predict_test_case()
        predictions = self.predict(parameters, X_assess)
        print("predictions mean = " + str(np.mean(predictions)))

    def execute(self):
        # Build a model with a n_h-dimensional hidden layer
        parameters = self.model(self.X, self.Y, n_h = 4, num_iterations = 10000, print_cost=True)

        # Plot the decision boundary
        plot_decision_boundary(lambda x: self.predict(parameters, x.T), self.X, self.Y)
        plt.title("Decision Boundary for hidden layer size " + str(4))
        # Print accuracy
        predictions = self.predict(parameters, self.X)
        print ('Accuracy: %d' % float((np.dot(self.Y,predictions.T) + np.dot(1-self.Y,1-predictions.T))/float(self.Y.size)*100) + '%')
        plt.show()


    # Apply Simple Logistic Regression on dataset and plot decision boundary
    def simpleLogReg(self):
        """
        Arguments:
        No explicit argument, just default fields of the class (self)
        
        Returns:
        Plot of the the logistic regression classifier decision boudry and accuracy
        """
        # Train the logistic regression classifier
        clf = sklearn.linear_model.LogisticRegressionCV()
        clf.fit(self.X.T, self.Y.T)
        # Plot the decision boundary for logistic regression
        plot_decision_boundary(lambda x: clf.predict(x), self.X, self.Y)
        plt.title("Logistic Regression")

        # Print accuracy
        LR_predictions = clf.predict(self.X.T)
        print ('Accuracy of logistic regression: %d ' % float((np.dot(self.Y,LR_predictions) + np.dot(1-self.Y,1-LR_predictions))/float(self.Y.size)*100) +
            '% ' + "(percentage of correctly labelled datapoints)")
        plt.show()
    # Check dataset sizes
    def layer_sizes(self, X, Y):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)
        
        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]
        return (n_x, n_h, n_y)
    # Initialize_parameters
    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        
        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    # Forward_propagation

    def forward_propagation(self, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]   #shape(4, 2)
        b1 = parameters["b1"]   #shape(4, 1)
        W2 = parameters["W2"]   #shape(1, 4)
        b2 = parameters["b2"]   #shape(1, 1)
        
        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(W1, X) + b1    #shape(4, 2) . shape(2, m) + shape(4, 1) = shape(4, m)
        A1 = np.tanh(Z1)           #shape(4, m)
        Z2 = np.dot(W2, A1) + b2   #shape(1, 4) . shape(4, m) + shape(1, 1) = shape(1, m)
        A2 = sigmoid(Z2)           #shape(1, m)
        
        assert(A2.shape == (1, X.shape[1]))
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache

    # Compute_cost
    def compute_cost(self, A2, Y, parameters):
        """
        Computes the cross-entropy
        
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        
        Returns:
        cost -- cross-entropy cost
        """
        
        m = Y.shape[1] # number of example

        # Compute the cross-entropy cost
        logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1-Y, np.log(1 - A2))
        cost = -np.sum(logprobs)/m
        
        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                    # E.g., turns [[17]] into 17 
        assert(isinstance(cost, float))
        
        return cost

    # Backward_propagation
    def backward_propagation(self, parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
        
        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters["W1"]    #shape(4, 2)
        W2 = parameters["W2"]    #shape(1, 4)
            
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]         #shape(4, m)
        A2 = cache["A2"]         #shape(1, m)
        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        dZ2 = A2 - Y             #shape(1, m)
        dW2 = np.dot(dZ2, A1.T)/m  #shape(1, 4)
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m #shape(1, 1)
        dZ1 = np.dot(W2.T, dZ2) * ( 1-np.power(np.tanh(cache["Z1"]), 2) ) #shape(4, m)
        dW1 = np.dot(dZ1, X.T)/m #shape(4, 2)
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m #shape(1, 1)
        
        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads

    # Update_parameters
    def update_parameters(self, parameters, grads, learning_rate = 1.2):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients 
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        
        # Update rule for each parameter
        W1 = W1 - learning_rate*dW1
        b1 = b1 - learning_rate*db1
        W2 = W2 - learning_rate*dW2
        b2 = b2 - learning_rate*db2
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters

    # Network Model
    def model(self, X, Y, n_h, num_iterations = 10000, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations
        
        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """
        
        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]
        
        # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self.forward_propagation(X, parameters)
            
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = self.compute_cost(A2, Y, parameters)
    
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = self.backward_propagation(parameters, cache, X, Y)
    
            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = self.update_parameters(parameters, grads)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return parameters

    # Predict
    def predict(self, parameters, X):
        """
        Using the learned parameters, predicts a class for each example in X
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        X -- input data of size (n_x, m)
        
        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.forward_propagation(X, parameters)
        predictions = np.array([ [0 if num<=0.5 else 1 for num in A2[0, :]] ])
        
        return predictions

    # Optimize number of hidden nodes
    def opt_hidden(self):
        plt.figure(figsize=(16, 32))
        hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
        for i, n_h in enumerate(hidden_layer_sizes):
            plt.subplot(5, 2, i+1)
            plt.title('Hidden Layer of size %d' % n_h)
            parameters = self.model(self.X, self.Y, n_h, num_iterations = 5000)
            plot_decision_boundary(lambda x: self.predict(parameters, x.T), self.X, self.Y)
            predictions = self.predict(parameters, self.X)
            accuracy = float((np.dot(self.Y,predictions.T) + np.dot(1-self.Y,1-predictions.T))/float(self.Y.size)*100)
            print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
        plt.show()