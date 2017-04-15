import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha,plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    y_pred = nn.predict(x)
    mse = np.power(y - y_pred, 2)
    # print(mse.shape)
    return np.mean(mse)


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    # maximum number of hidden neurons
    n_hidden = 40
    nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden))
    nn.fit(x_train, y_train)
    y_pred_train = nn.predict(x_train)
    y_pred_test = nn.predict(x_test)
    plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
    ## TODO
    pass

def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    # store test and train error in numpy arrays

    # for every number n of hidden neurons store the test and train error
    for i in range(10):
        n_hidden = 10
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden), random_state=i)
        nn.fit(x_train, y_train)
        mse_test = calculate_mse(nn, x_test, y_test)
        mse_train = calculate_mse(nn, x_train, y_train)

        print("train mse: {}, test mse {}".format(mse_train, mse_test))
    
    # and plot the errors 
    ## TODO
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    seeds = 10
    n_values = [1, 2, 3, 4, 6, 8, 12, 20, 40]

    mse_test = np.empty((len(n_values), seeds))
    mse_train = np.empty((len(n_values), seeds))

    for i, n in enumerate(n_values):
        n_hidden = n
        for s in range(1, seeds + 1):
            nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden), random_state=s)
            nn.fit(x_train, y_train)
            mse_test[i, s - 1] = calculate_mse(nn, x_test, y_test)
            mse_train[i, s - 1] = calculate_mse(nn, x_train, y_train)
    ## TODO
    plot_mse_vs_neurons(mse_train, mse_test, n_values)
    pass

def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    n_values = [2, 8, 20]
    n_iter = 1000

    mse_test = np.empty((len(n_values), n_iter))
    mse_train = np.empty((len(n_values), n_iter))
    for solver in ['lbfgs', 'adam', 'sgd']:
        for index, n in enumerate(n_values):
            n_hidden = n
            nn = MLPRegressor(activation='logistic', solver=solver, hidden_layer_sizes=(n_hidden), warm_start=True, max_iter=1)
            for i in range(n_iter):
                nn.fit(x_train, y_train)
                mse_test[index, i] = calculate_mse(nn, x_test, y_test)
                mse_train[index, i] = calculate_mse(nn, x_train, y_train)

        plot_mse_vs_iterations(mse_train, mse_test, n_iter, n_values)
    ## TODO
    pass




def ex_1_2_a(x_train, x_test, y_train, y_test):

    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    seeds = 10
    
    alpha = [ 1e-8, 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100 ]
    n_hidden = 40 
    
    mse_test = np.empty((len(alpha), seeds))
    mse_train = np.empty((len(alpha), seeds))

    for i, n in enumerate(alpha):
        
        for s in range(1, seeds + 1):
            nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden), alpha = n, random_state = s )
            nn.fit(x_train, y_train)
            a = nn.coefs_
            a = np.mean(np.square(a[0]))
            a = a*n/2
            
            mse_test[i, s - 1] = calculate_mse(nn, x_test, y_test) + a
            mse_train[i, s - 1] = calculate_mse(nn, x_train, y_train) + a
            
            
    ## TODO
    plot_mse_vs_alpha(mse_train, mse_test, alpha)
    ## TODO
    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    a , b = shuffle(x_train,y_train)
    x_val = np.empty((int(len(x_train)/2),))
    y_val = np.empty((int(len(y_train)/2),))
    x_val = a[int(len(x_train)/2):]
    x_train = a[:int(len(x_train)/2)]
    y_val = b[int(len(y_train)/2):]
    y_train = b[:int(len(y_train)/2)]
    
    
    seeds = 10
    n_hidden = 40
    n_iter = 2000

    mse_test = np.empty((int(n_iter/20),))
    mse_val = np.empty((int(n_iter/20),))
#    mse_train = np.empty((int(n_iter/20),))
    result1 = np.empty((seeds,))
    result2 = np.empty((seeds,))
    result3 = np.empty((seeds,))
    for s in range(seeds):
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden), warm_start=True, max_iter=1,random_state=s)
        k=0
        for i in range(n_iter):
            nn.fit(x_train, y_train)
            if i%20 == 0:
                mse_test[k] = calculate_mse(nn, x_test, y_test)
                mse_val[k] = calculate_mse(nn, x_val, y_val)
 #               mse_train[k] = calculate_mse(nn, x_train, y_train)
                k = k + 1
        index = np.argmin(mse_val)
        index1 = np.argmin(mse_test)
        
        result1[s] = mse_test[int(n_iter/20)-1]
        result2[s] = mse_test[index]
        result3[s] = mse_test[index1]
        
#        print(mse_test[int(n_iter/20)-1],mse_test[index],index,mse_test[index1],index1)
#        
#    plt.plot(mse_val,'r')
#    plt.plot(mse_train,'g')
#    plt.plot(mse_test,'b')
    
    ## TODO
    pass

def ex_1_2_c(x_train, x_test, y_train, y_test):
    '''
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    ## TODO
    pass
