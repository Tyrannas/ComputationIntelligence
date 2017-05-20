import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import pylab
from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec, subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    # create and train the support vector machine.
    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(x, y)

    # plot results 
    plot_svm_decision_boundary(clf, x, y)

    pass


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    # add the point to the datas
    x = np.vstack((x, [4, 0]))
    y = np.hstack((y, 1))

    # create and train the SVM.
    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(x, y)

    # plot results
    plot_svm_decision_boundary(clf, x, y)
    pass


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    Objective: test the influence of the differents 
               values for C, the positive cost factor
    :param x: The x values
    :param y: The y values
    :return:
    """

    # add the point to the datas
    x = np.vstack((x, [4, 0]))
    y = np.hstack((y, 1))

    # values for C 
    Cs = [1e6, 1, 0.1, 0.001]

    # create, train and plot for every C value
    for C in Cs:
        clf = svm.SVC(C, kernel='linear')
        clf.fit(x, y)


        plot_svm_decision_boundary(clf, x, y)

def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    # create and train the SVM.
    clf = svm.SVC(C=1., kernel='linear')
    clf.fit(x_train, y_train)
    print(np.shape(clf.coef_))
    # plot the decision boundary on both datasets
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)

    # compute the mean accuracy of the test datas
    a = clf.support_vectors_
    print("SVC's score: {}".format(clf.score(x_test, y_test)))
    print("number of SV:", len(a))

    pass


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    # parameters
    r_value = 1
    degrees = range(1, 21)
    C = 1

    # store the scores
    train_scores = []
    test_scores = []

    # store the created svm so we don't have to train the best twice. 
    clfs = []

    # create and train the svm with a polynomial kernel for every d value
    for d in degrees:
        clf = svm.SVC(C=C, kernel='poly', degree=d, coef0=r_value)
        clf.fit(x_train, y_train)
        clfs.append(clf)
        # compute the scores
        train_scores.append(clf.score(x_train, y_train))
        test_scores.append(clf.score(x_test, y_test))


    # find the svm with the better test score
    max_index = test_scores.index(max(test_scores))
    clf = clfs[max_index]
    a = clf.support_vectors_
    print("best d value: {}, with an accuracy of {}".format(degrees[max_index], test_scores[max_index]))
    print("number of SV:",len(a) )
    # plot the decision boundary on both datasets for the best svm
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)

    # plot the score depending of d
    plot_score_vs_degree(train_scores, test_scores, degrees)

def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    # parameters
    gammas = np.arange(0.01, 2,0.02 )
    C = 1

    # store the scores
    train_scores = []
    test_scores = []

    # store the created svm so we don't have to train the best twice. 
    clfs = []

    # create and train the svm with a polynomial kernel for every d value
    for g in gammas:
        clf = svm.SVC(C=C, kernel='rbf', gamma=g)
        clf.fit(x_train, y_train)
        clfs.append(clf)
        # compute the scores
        train_scores.append(clf.score(x_train, y_train))
        test_scores.append(clf.score(x_test, y_test))


    # find the svm with the better test score
    max_index = test_scores.index(max(test_scores))
    clf = clfs[max_index]
    a = clf.support_vectors_
    print("best g value: {}, with an accuracy of {}".format(gammas[max_index], test_scores[max_index]))
    print("number of SV:", len(a))

    # plot the decision boundary on both datasets for the best svm
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)

    # plot the score depending of g
    plot_score_vs_gamma(train_scores, test_scores, gammas)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**-3
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########

    # parameters

    C = 0.0003

    # store the scores
    train_scores = []
    test_scores = []

    # store the created svm so we don't have to train the best twice.
    clfs = []
    #first the linear and then the rbf
    clf = svm.SVC(C=C, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    clfs.append(clf)

    # compute the scores
    train_scores.append(clf.score(x_train, y_train))
    test_scores.append(clf.score(x_test, y_test))

    # gammas = np.linspace(0.00001, 0.001, 10)
    # for g in gammas:
    #     clf = svm.SVC(C=C, kernel='rbf', gamma=g,decision_function_shape='ovr')
    #     clf.fit(x_train, y_train)
    #     clfs.append(clf)
    #
    #
    #
    #     # compute the scores
    #
    #     train_scores.append(clf.score(x_train, y_train))
    #     test_scores.append(clf.score(x_test, y_test))
    #
    # # plot the score depending of g , with linear score
    # plot_score_vs_gamma(train_scores[1:], test_scores[1:], gammas,train_scores[0],test_scores[0])



def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    C=0.0003
    clf = svm.SVC(C=C, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    labels = range(1, 6)

    plot_confusion_matrix(confusion_matrix(y_test,y_pred),labels)



    sel_error = np.array([0])  # Numpy indices to select images that are misclassified.
    i = 0   # should be the label number corresponding the largest classification error
    #in order to find the most missclassified we sum up the missclassified of every label and then we find the one with maximum error
    sums = np.zeros((5,))
    k=0
    for j in y_pred:
        if j!= y_test[k]:
            sums[y_test[k]-1] +=1
            sel_error = np.append(sel_error,k)
        k+=1
    i = np.argmax(sums)


    # Plot with mnist plot
    plot_mnist(x_test[sel_error], y_pred[sel_error], labels=labels[i], k_plots=10, prefix='Predicted class')
