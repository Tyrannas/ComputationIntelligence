#!/usr/bin/env python
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gradient_descent as gd
import logreg as lr
import logreg_toolbox

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression

This is the main file that loads the data, compute the solution and plot the results.
"""


def main():
    # Set parameters
    degree = 5
    eta = 1
    max_iter = 10

    # Load data and expand with polynomial features
    f = open('data_logreg.json', 'r')
    data = json.load(f)
    for k, v in data.items(): data[k] = np.array(v)  # Encode list into numpy array

    # Expand with polynomial features
    X_train = logreg_toolbox.poly_2D_design_matrix(data['x1_train'], data['x2_train'], degree)
    X_test = logreg_toolbox.poly_2D_design_matrix(data['x1_test'], data['x2_test'], degree)
    n = X_train.shape[1]

    # Define the functions of the parameter we want to optimize
    def f(theta): return lr.cost(theta, X_train, data['y_train'])

    def df(theta): return lr.grad(theta, X_train, data['y_train'])



    # Test to verify if the computation of the gradient is correct
    logreg_toolbox.check_gradient(f, df, n)

    # Point for initialization of gradient descent
    theta0 = np.zeros(n)
    
    #### VARIANT 1: Optimize with gradient descent
    theta_opt, E_list = gd.gradient_descent(f, df, theta0, eta, max_iter)
    """
    nb_iters =[]
    eta_values =[]
    train_errors =[]
    test_errors = []

    # Loop for visualising the relationship between the different errors, the number of iterations and the learning rate
    # for nb_iter in range(1,50):
    #     print(nb_iter)
    #     for eta_value in range(0,110):
    #         theta_opt, E_list = gd.gradient_descent(f, df, theta0, eta_value/10, nb_iter)
    #         train_error = lr.cost(theta_opt, X_train, data['y_train'])
    #         test_error = lr.cost(theta_opt, X_test, data['y_test'])
    #         nb_iters.append(nb_iter)
    #         eta_values.append(eta_value)
    #         train_errors.append(train_error)
    #         test_errors.append(test_errors)
    # print(train_errors)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # eta_values, nb_iters = np.meshgrid(eta_values, nb_iters)
    # surf = ax.plot_surface(eta_values, nb_iters, train_errors, rstride=1, cstride=1, cmap=cm.RdBu,linewidth=1, antialiased=False)

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()
    """
    #### VARIANT 2: Optimize with adaptive gradient descent
    #theta_opt, E_list, l_rate_final = gd.adaptative_gradient_descent(f, df, theta0, eta, max_iter)
    #print('Adaptative gradient, final learning rate: {:.3g}'.format(l_rate_final))

    #### VARIANT 3: Optimize with gradient descent on scipy
    # E_list = []
    # res = minimize(f, x0=theta0, jac=df, options={'disp': True}, method='Nelder-Mead')
    # theta_opt = res.x.reshape((n, 1))



    logreg_toolbox.plot_logreg(data, degree, theta_opt, E_list)
    plt.show()


if __name__ == '__main__':
    main()
