#!/usr/bin/env python3
#Filename skeleton_HW5.py
#Author: Christian Knoll, Philipp Gabler
#Edited: 01.6.2017
#Edited: 02.6.2017 -- naming conventions, comments, ...

import numpy as np
import numpy.random as rd
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
from math import pi, exp
<<<<<<< HEAD
from scipy.stats import multivariate_normal, norm
=======
from scipy.stats import multivariate_normal
import warnings
>>>>>>> 66ab80e2e2354796cf866a2ec1977d5c0ae87d13


## -------------------------------------------------------    
## ---------------- HELPER FUNCTIONS ---------------------
## -------------------------------------------------------

def sample_discrete_pmf(X, PM, N):
    X, PM = np.asarray(X), np.asarray(PM)

    assert np.isclose(PM.sum(), 1.0)
    assert np.logical_and(0.0 <= PM, PM <= 1.0).all()

    y = np.zeros(N, dtype = X.dtype)
    cumulativePM = np.cumsum(PM)
    offset = rd.uniform(0, 1) * (1 / N)
    comb = np.arange(offset, 1 + offset, 1 / N)

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]:
            j += 1
        y[i] = X[j]

    return rd.permutation(y)




def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax):
    """Show contour plot for bivariate Gaussian with given mu and cov in the range specified.

    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01], [cov_10, cov_11]]
    xmin, xmax, ymin, ymax ... range for plotting
    """
    
    npts = 500
    deltaX = (xmax - xmin) / npts
    deltaY = (ymax - ymin) / npts
    stdev = [0, 0]

    stdev[0] = np.sqrt(cov[0][0])
    stdev[1] = np.sqrt(cov[1][1])
    x = np.arange(xmin, xmax, deltaX)
    y = np.arange(ymin, ymax, deltaY)
    X, Y = np.meshgrid(x, y)

    Z = mlab.bivariate_normal(X, Y, stdev[0], stdev[1], mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline = 1, fontsize = 10)
    # plt.title(title)


def likelihood_bivariate_normal(X, mu, cov):
    """Returns the likelihood of X for bivariate Gaussian specified with mu and cov.

    X  ... vector to be evaluated -- np.array([[x_00, x_01], ..., [x_n0, x_n1]])
    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01],[cov_10, cov_11]]
    """
    
    dist = multivariate_normal(mu, cov)
    P = dist.pdf(X)
    return P



## -------------------------------------------------------    
## ------------- START OF  ASSIGNMENT 5 ------------------
## -------------------------------------------------------



def EM(X, M, alpha_0, mu_0, Sigma_0, max_iter):

    # Step 1, Init the parameters 
    alpha = alpha_0
    mu = mu_0
    sigma = Sigma_0
    L1 = 10000
    L = []
    prob = np.zeros((M, 20000))
    rm = np.zeros((M, 20000))
    Nm = np.zeros(M,)
    P = np.zeros((M,20000 ))
    softclass = np.zeros((20000,))
    for i in range(max_iter):

        for j in range(M):

            prob[j] = alpha[j]*multivariate_normal(mu[j], sigma[j]).pdf(X)

        rm = prob/(np.sum(prob,axis = 0))

        for j in range(M):

            Nm[j] = np.sum(rm[j])

            mu[j] = np.dot(rm[j],X)/Nm[j]

            d = X - mu[j]
            # sigma[j] = np.dot(rm[j] * d, d.T) / Nm[j]


        alpha = Nm/20000
        sigma = sigma + [[50,0],[0,50]]
        for j in range(M):
            P[j] = alpha[j]*multivariate_normal(mu[j], sigma[j]).pdf(X)
        L += [np.log(np.sum(P))]


        if np.abs(L1-L[i]) > 0.000001:
            L1= L[i]
        else:
            # print(i)
            break
    # question 6
    softclass = np.argmax(rm, axis=0)

    return alpha,mu,sigma,L,softclass



def k_means(X, M, mu_0, max_iter):
    mu = mu_0
    D = np.empty(max_iter)
    # for every iteration
    for it in range(max_iter):
        mu_temp = [[X[0]] for i in range(M)] # intial regularization
        # for every value
        for x in X:
            arr = []
            # for every center in mu
            for m in mu:
                # lets compute the distance between the center and the value
                arr.append(np.linalg.norm(np.array(x) - np.array(m)))

            # store the minimum distance in D
            D[it] += min(arr)
            # store x in the center's array

            mu_temp[np.argmin(arr)].append(x)
        # update mu
        mu = np.array([np.mean(m, axis=0) for m in mu_temp])

    return mu, D


def sample_GMM(alpha, mu, Sigma, N,X):
    # TODO
    PM = [multivariate_normal(mu[0], Sigma[0]).pdf(mu[0]+(45000*N/2-i)*np.array([0.005,0.005])) for i in range(45000*N)]
    print(PM,np.sum(PM))
    y = sample_discrete_pmf(X, PM, N)
    return y


def main():
    # load data
    X = np.loadtxt('data/X.data', skiprows = 0) # unlabeled data
    a = np.loadtxt('data/a.data', skiprows = 0) # label: a
    e = np.loadtxt('data/e.data', skiprows = 0) # label: e
    i = np.loadtxt('data/i.data', skiprows = 0) # label: i
    o = np.loadtxt('data/o.data', skiprows = 0) # label: o
    y = np.loadtxt('data/y.data', skiprows = 0) # label: y

    # 1.) EM algorithm for GMM:
    # TODO
    M = 5

    X = np.array(X)
    alpha_0 = np.array([1/M for i in range(M)])
    mu_0 = np.random.uniform(200,3000,(M,2))
    Sigma_0 = 10000*np.array([np.identity(2) for i in range(M)])
    max_iter = 1000
    # Question 5
    # Sigma_0 = [[100000,5000],[5000,100000]] * np.array([np.abs(np.random.random((2,2))) for i in range(M)])
    alpha, mu, Sigma, L , softclass =  EM(X, M, alpha_0, mu_0, Sigma_0, max_iter)

    # Question  4
    plt.plot(L)
    plt.show()

    # Question 2, 3, 5
    colors = ['red', 'green', 'yellow', 'blue', 'orange']
    for index, data in enumerate([a, e, i, o, y]):
        plt.scatter(data[:, 0], data[:, 1], c=colors[index])

    for j in range(M):
        plot_gauss_contour(mu[j],Sigma[j],0,1200,0,3000)

    plt.show()

    # Question 6
    a1 = [list(X[0])]
    b1 = [list(X[0])]
    c1 = [list(X[0])]
    d1 = [list(X[0])]
    e1 = [list(X[0])]
    for j, clas in enumerate(softclass):
        if clas == 0:
            a1 += [list(X[j])]
        elif clas == 1:
            b1 += [list(X[j])]
        elif clas == 2:
            c1 += [list(X[j])]
        elif clas == 3:
            d1 += [list(X[j])]
        elif clas == 4:
            e1 += [list(X[j])]
    a1 = np.array(a1)
    b1 = np.array(b1)
    c1 = np.array(c1)
    d1 = np.array(d1)
    e1 = np.array(e1)

    colors = ['red', 'green', 'yellow', 'blue', 'orange']

    for index, data in enumerate([a1, b1, c1, d1,  e1]):
        plt.scatter(data[:,0], data[:,1], c=colors[index])
    plt.show()

    # 2.) K-means algorithm:
    max_iter = 10


    mu, D = k_means(X, M, mu_0, max_iter)
    colors = ['red', 'green', 'yellow', 'blue', 'orange']


    for index, data in enumerate([a, e, i, o, y]):
        plt.scatter(data[:,0], data[:,1], c=colors[index])

    plt.show()
    plot_kmeans_results(X, mu, D)

    #Using as initial mean values the result from k-means
    max_iter = 1000

    alpha, mu, Sigma, L, softclass = EM(X, M, alpha_0, mu, Sigma_0, max_iter)
    plt.plot(L)
    plt.show()

    # Question 2, 3, 5
    colors = ['red', 'green', 'yellow', 'blue', 'orange']
    for index, data in enumerate([a, e, i, o, y]):
        plt.scatter(data[:, 0], data[:, 1], c=colors[index])

    for j in range(M):
        plot_gauss_contour(mu[j], Sigma[j], 0, 1200, 0, 3000)

    plt.show()


    # # 3.) Sampling from GMM
    # # TODO
    #
    #
    # y = sample_GMM(alpha_0,mu_0,Sigma_0,5,X)
    # plt.scatter(y[:, 0], y[:, 1])
    # pass

def plot_kmeans_results(X, mu, D):
    colors = ['red', 'green', 'yellow', 'blue', 'orange']
    plt.plot(D)
    plt.show()
    c = [colors[np.argmin([np.linalg.norm(np.array(x) - np.array(m)) for m in mu])] for x in X]
    plt.scatter(X[:,0], X[:,1], c=c)
    plt.scatter(mu[:,0], mu[:,1], c='black')
    plt.show()

def plot_EM_results(X, rm):
    print(rm)

def sanity_checks():
    # likelihood_bivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_bivariate_normal(x, mu, cov)
    print(P)

    # plot_gauss_contour(mu, cov, -2, 2, -2, 2, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)
    
    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))


if __name__ == '__main__':
    # to make experiments replicable (you can change this, if you like)
    # rd.seed(1)
    # rd.seed(233151758)

    # sanity_checks()
    main()
