import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


class Logistic_Regression:
    # Loading the data at initialization
    def __init__(self):
        self.train_set_x_orig, self.train_set_y, self.test_set_x_orig, self.test_set_y, self.classes = load_dataset()
        self.train_set_x_flatten = self.train_set_x_orig.reshape(self.train_set_x_orig.shape[0], -1).T
        self.test_set_x_flatten = self.test_set_x_orig.reshape(self.test_set_x_orig.shape[0], -1).T
        self.train_set_x = self.train_set_x_flatten/255.
        self.test_set_x = self.test_set_x_flatten/255.
        self.w, self.b = self.initialize_with_zeros(2)


    # Check the functions and sanity checks
    def check(self, index):
        plt.imshow(self.train_set_x_orig[index])
        print ("y = " + str(self.train_set_y[:, index]) + ", it's a '" + self.classes[np.squeeze(self.train_set_y[:, index])].decode("utf-8") +  "' picture.")

        m_train = len(self.train_set_x_orig)
        m_test = len(self.test_set_x_orig)
        print("size of training set ( m_train = " + str(m_train) + " )")
        print("size of test set ( m_test = " + str(m_test) + " )")
        print ("train_set_x shape: " + str(self.train_set_x_orig.shape))
        print ("train_set_y shape: " + str(self.train_set_y.shape))
        print ("test_set_x shape: " + str(self.test_set_x_orig.shape))
        print ("test_set_y shape: " + str(self.test_set_y.shape))
        print ("train_set_x_flatten shape: " + str(self.train_set_x_flatten.shape))
        print ("test_set_x_flatten shape: " + str(self.test_set_x_flatten.shape))
        print ("sanity check after reshaping: " + str(self.train_set_x_flatten[0:5,0]))
        print ("sigmoid([0, 2]) = " + str(self.sigmoid(np.array([0,2]))))
        print ("w = " + str(self.w))
        print ("b = " + str(self.b))
        w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
        grads, cost = self.propagate(w, b, X, Y)
        print ("dw = " + str(grads["dw"]))
        print ("db = " + str(grads["db"]))
        print ("cost = " + str(cost))
        params, grads, costs = self.optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
        print ("w = " + str(params["w"]))
        print ("b = " + str(params["b"]))
        print ("dw = " + str(grads["dw"]))
        print ("db = " + str(grads["db"]))
        print ("predictions = " + str(self.predict(w, b, X)))
        plt.show()

    def execute(self, iter, lrate, print_bool):
        d = self.model(self.train_set_x, self.train_set_y, self.test_set_x, self.test_set_y, num_iterations = iter, learning_rate = lrate, print_cost = print_bool)


    def sigmoid(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """
        s = 1./(1. + np.exp(-z))
        return s

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        
        w = np.zeros((dim,1), float)
        b = 0.

        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        
        return w, b

    def propagate(self, w, b, X, Y):
        """
        This function implements the cost function and its gradient

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        
        """
        
        m = X.shape[1] #number of training examples
        
        # FORWARD PROPAGATION (FROM X TO COST)
        A = self.sigmoid( np.dot(w.T, X) + b )                                     # compute activation, shape (1, m)
        cost = -np.sum( Y*np.log(A)+(1-Y)*np.log(1-A) )/m                     # compute cost, shape (1, 1)
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = np.dot(X, (A-Y).T)/m
        db = np.sum(A-Y)/m

        # Sanity checks
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                "db": db}
        
        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate*dw
            b = b - learning_rate*db
            if i % 100 == 0:
                costs.append(cost)
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        params = {"w": w,
                "b": b}
        grads = {"dw": dw,
                "db": db}
        return params, grads, costs

    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = self.sigmoid( np.dot(w.T, X)+b )
        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            Y_prediction[0, i] = 0 if A[0, i]<=0.5 else 1
        assert(Y_prediction.shape == (1, m))
        return Y_prediction

    def model(self, X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        """
        Builds the logistic regression model by calling the function you've implemented previously
        
        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations
        
        Returns:
        d -- dictionary containing information about the model.
        """
        dim = X_train.shape[0]
        w, b = self.initialize_with_zeros(dim)
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)
        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}
        return d