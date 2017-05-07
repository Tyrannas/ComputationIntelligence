import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha,plot_bars_early_stopping_mse_comparison
import matplotlib.pyplot as plt

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
    # compute the predicted target with the trained nn
    y_pred = nn.predict(x)
    mse = np.power(y - y_pred, 2)
    return np.mean(mse)


def ex_1_1_a(x_train, x_test, y_train, y_test,n_hidden):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    
    nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden,), max_iter=200, alpha=0)
    nn.fit(x_train, y_train)
    y_pred_train = nn.predict(x_train)
    y_pred_test = nn.predict(x_test)
    plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)

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
    mses_train = np.empty(10)
    mses_test = np.empty(10)

    n_hidden = 10       

    # for every different seed i, store the associated training and testing errors
    for i in range(10):
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden), random_state=i, max_iter=200)
        nn.fit(x_train, y_train)
        mses_test[i] = calculate_mse(nn, x_test, y_test)
        mses_train[i] = calculate_mse(nn, x_train, y_train)

    print("min_train: {}, max_train: {}, mean_train: {}, std_train: {}".format(np.min(mses_train), np.max(mses_train), np.mean(mses_train), np.std(mses_train)))
    print("min_test: {}, max_test: {}, mean_test: {}, std_test: {}".format(np.min(mses_test), np.max(mses_test), np.mean(mses_test), np.std(mses_test)))
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
    # number of different seeds
    seeds = 10
    # values of number of hidden neurons
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

    # for each different solver
    for solver in ['lbfgs', 'adam', 'sgd']:
        # for each value of n_hidden
        for index, n in enumerate(n_values):
            n_hidden = n
            nn = MLPRegressor(activation='logistic', solver=solver, hidden_layer_sizes=(n_hidden), warm_start=True, max_iter=1, alpha=0)
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
    # number of different seeds
    seeds = 10
    # values for alpha
    alphas = [ 1e-8, 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100 ]
    n_hidden = 40 
    
    mse_test = np.empty((len(alphas), seeds))
    mse_train = np.empty((len(alphas), seeds))

    for i, alpha in enumerate(alphas):
        for s in range(1, seeds + 1):
            nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden), alpha = alpha ,random_state=s)
            nn.fit(x_train, y_train)
            # computation of the msereg
            a = nn.coefs_
            a = np.mean(np.square(a[0]))
            a = a*alpha/2
            
            mse_test[i, s - 1] = calculate_mse(nn, x_test, y_test) + a
            mse_train[i, s - 1] = calculate_mse(nn, x_train, y_train) + a
            
    plot_mse_vs_alpha(mse_train, mse_test, alphas)
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
    # split the datas to get a validation set
    a, b = shuffle(x_train,y_train)
    x_val = np.empty((int(len(x_train)/2),))
    y_val = np.empty((int(len(y_train)/2),))
    x_val = a[int(len(x_train)/2):]
    x_train = a[:int(len(x_train)/2)]
    y_val = b[int(len(y_train)/2):]
    y_train = b[:int(len(y_train)/2)]

    # parameters
    seeds = 10
    n_hidden = 40
    n_iter = 2000

    # errors
    mse_test = np.empty((int(n_iter/20),))
    mse_val = np.empty((int(n_iter/20),))
    mse_train = np.empty((int(n_iter/20),))

    # results
    result1 = np.empty((seeds,))
    result2 = np.empty((seeds,))
    result3 = np.empty((seeds,))


    for s in range(seeds):
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden), warm_start=True, max_iter=1, random_state=s, alpha=0, momentum=0)
        k = 0

        for i in range(n_iter):
            nn.fit(x_train, y_train)
            if i%20 == 0:
                mse_test[k] = calculate_mse(nn, x_test, y_test)
                mse_val[k] = calculate_mse(nn, x_val, y_val)
                mse_train[k] = calculate_mse(nn, x_train, y_train)
                k = k + 1

        index = np.argmin(mse_val)
        index1 = np.argmin(mse_test)

        result1[s] = mse_test[int(n_iter/20)-1]
        result2[s] = mse_test[index]
        result3[s] = mse_test[index1]

    seeds = np.arange(seeds)

    # plot the datas
    ax = plt.subplot()
    rect1 = ax.bar(seeds - 0.2, result1, width=0.2, color='#5194ff', align='center')   
    rect2 = ax.bar(seeds, result2, width=0.2, color='#e83a3a', align='center')   
    rect3 = ax.bar(seeds + 0.2, result3, width=0.2, color='#e8993a', align='center')    

    # add lengend
    ax.legend( (rect1[0],rect2[0],rect3[0]),('Last iteration MSE of Test Set', 'Best MSE of Validation Set', 'Best MSE of Test set') )
    ax.set_xlim([-1,12])

    plt.show()
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
    # separate the datas
    a, b = shuffle(x_train,y_train, random_state=15)

    x_val = np.empty((int(len(x_train)/2),))
    y_val = np.empty((int(len(y_train)/2),))
    x_val = a[int(len(x_train)/2):]
    x_train = a[:int(len(x_train)/2)]
    y_val = b[int(len(y_train)/2):]
    y_train = b[:int(len(y_train)/2)]
    # parameters
    seeds = 100
    n_iter = 1000
    # errors
    mse_test = np.empty((seeds,))
    mse_val = np.empty((seeds,))
    mse_train = np.empty((seeds,))
    
    k = np.zeros((seeds,))
    
    for s in range(seeds):
        nn = MLPRegressor(alpha = 0.001 , activation='logistic', solver='lbfgs' , hidden_layer_sizes=(8,), warm_start=True, max_iter=1, random_state=s)
        mse_val[s] = 100

        for i in range(n_iter):
            nn.fit(x_train, y_train)
            k[s] = k[s] + 1
            if i%5 == 0:
                a = nn.coefs_
                a = np.mean(np.square(a[0]))
                a = a*0.001/2
                mse_reg = calculate_mse(nn, x_val, y_val) + a
                if mse_reg > mse_val[s]:
                    mse_val[s]  = mse_reg
                    mse_test[s] = calculate_mse(nn, x_test, y_test) + a
                    mse_train[s] = calculate_mse(nn, x_train, y_train) + a            
                    break
                else:
                    mse_val[s] = mse_reg

    mean_train = np.mean(mse_train)
    std_train = np.std(mse_train)
    mean_val = np.mean(mse_val)
    std_val = np.std(mse_val)
    mean_test = np.mean(mse_test)
    std_test = np.std(mse_test)
    
    print('mean train: {}, mean validation: {}, mean test: {}'.format(mean_train, mean_val, mean_test))
    print('std train: {}, std validation: {}, std test: {}'.format(std_train, std_val, std_test))
    
    index = np.argmin(mse_val)

    print('mse_train: {}, mse validation: {}, mse test: {}, number of iterations: {}, best seed: {}'.format(mse_train[index], mse_val[index], mse_test[index], k[index], index))
    pass