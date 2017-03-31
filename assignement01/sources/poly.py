#!/usr/bin/env python
import numpy as np
from numpy.linalg import pinv

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file contains the main work to be done.
The functions are:
- TODO design_matrix: Create the design matrix including the polynomial expansions and the constant feature
- TODO train: finds the analytical solution of linear regression
- TODO compute_error: return the cost function of linear regression Mean Square Error
- train_and_test: call the compute error function and all sets and return the corresponding errors

"""


def design_matrix(x, degree):
    """
    Creates the design matrix given the data x.
    The design matrix is built of all polynomials of x from degree 0 to 'degree' minus one.

    EX: for the data x = [0,1,2] and degree 2
    the function should return: [[1,1,1],[0,1,2],[0,1,4]]

    :param x: numpy array of shape (N,1)
    :param degree: Higher degree of the polynomial
    :return: Expanded data in a numpy array of shape (N,degree)
    """

    return np.hstack([np.power(x, i) for i in range(degree + 1)])


def train(x, y, degree):
    """
    Returns the optimal coefficients theta that minimizes the error
    ||  X * theta - y ||**2
    when X is the polynomial expansion of x_train of degree 'degree'.

    :param x: numpy array on the input
    :param y: numpy array containing the output
    :param degree: maximum polynomial degree in the polynomial expansion
    :return: a numpy array containing the coefficient of each polynomial degree in the regression
    """

    X = design_matrix(x, degree)
    theta_opt = pinv(X).dot(y)
    return theta_opt


def compute_error(theta, degree, x, y):
    """
    Predict the value of y given by the model given by theta and degree.
    Then compare the predicted value to y and provide the mean square error.

    :param theta: Coefficients of the linear regression
    :param degree: Degree in the polynomial expansion
    :param x: Input data
    :param y: Output data to be compared to prediction
    :return: err: Mean square error
    """

    X = design_matrix(x, degree)
    y_predict = X.dot(theta)
    err = np.subtract(y_predict,y)
    err2 = np.power(err,2)
    return np.mean(err2)


def train_and_test(data, degree):
    """
    Train the model with degree 'degree' and provide the MSE for the training, validation and testing sets

    :param data:
    :param degree:
    :return:
    """

    theta = train(data['x_train'], data['y_train'], degree)

    err_train = compute_error(theta, degree, data['x_train'], data['y_train'])
    err_val = compute_error(theta, degree, data['x_val'], data['y_val'])
    err_test = compute_error(theta, degree, data['x_test'], data['y_test'])

    return theta, err_train, err_val, err_test
