#!/usr/bin/env python
import json
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import poly
from plot_poly import plot_poly
from plot_poly import plot_errors
import numpy as np

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This files:
1) loads the data from 'data_linreg.json'
2) trains and test a linear regression model for a given number of degree
3) plots the results

TODO boxes are to be found in 'poly.py'
"""


def main():
    # Set the degree of the polynomial expansion
    degree = 13
    data_path = 'data_linreg.json'
    # Load the data
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))
    
    #Print the training and testing errors
    ts = time.time()
    theta, err_train, err_val, err_test = poly.train_and_test(data, degree)
    print('Execution time: {}'.format(ts - time.time()))
    print(err_test)
    #plot the data
    plot_poly(data, degree, theta)
    plt.show()


if __name__ == '__main__':
    main()
