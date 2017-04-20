from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc
import numpy as np



__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    ## TODO
    target2 = np.transpose(target2)
    target2 = target2[1]
    nn = MLPClassifier(hidden_layer_sizes=(8, ), activation='tanh', solver='adam', max_iter=200)
    print(target2)
    model = nn.fit(input2,target2)
    
    y_predict = model.predict(input2)
    
    C = confusion_matrix(y_predict,target2)
    print(C) #first row = down , second row = right , third row = left , fourth row = up 
    hidden_layer_weights = model.coefs_
    
    plot_hidden_layer_weights(hidden_layer_weights[0])
    
    pass


def ex_2_2(input1, target1, input2, target2):
    ## TODO
    pass


def calculate_mse(nn, x, y):
    ## TODO
    return 0
