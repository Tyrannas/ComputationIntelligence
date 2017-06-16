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
from scipy.stats import multivariate_normal


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




def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, title):
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
    plt.title(title)


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
    # TODO
    alpha = alpha_0
    mu = mu_0
    sigma = Sigma_0
    L1 = 1000
    L = 0
    rm = np.zeros((M,20000))
    Nm = np.zeros(M,)
    P = np.zeros((M,20000 ))
    for i in range(max_iter):

        for j in range(M):

            rm[j] = alpha[j]*multivariate_normal(mu[j], sigma[j]).pdf(X)





        rm = rm/(np.sum(rm))
        for j in range(M):

            Nm[j] = np.sum(rm[j])


        for j in range(M):

            mu[j] = np.dot(rm[j],X)/Nm[j]


        for j in range(M):
            d = X - mu[j]
            sigma[j] = np.dot(rm[j]*d.T,d)/Nm[j]



        alpha = Nm/20000

        # sigma = sigma + 150
        for j in range(M):
            P[j] = alpha[j]*multivariate_normal(mu[j], sigma[j]).pdf(X)
        L = np.log(np.sum(P))

        print(L)

        if np.abs(L1-L) > 0.00001:
            L1= L
        if np.abs(L1 - L) < 0.00001:
            print('afsj')
            break
    # j=0
    # cl = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
    # for i in range(20000):
    #     j = np.argmax(rm[:,i])
    #     cl[j] += X[i,:]
    # print(cl)

    return alpha,mu,sigma,L



def k_means(X, M, mu_0, max_iter):
    # TODO
    pass


def sample_GMM(alpha, mu, Sigma, N):
    # TODO
    pass


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
    max_iter = 100


    alpha, mu, Sigma, L =  EM(X, M, alpha_0, mu_0, Sigma_0, max_iter)
    print(alpha,mu,Sigma,L)

    plt.scatter(a[:, 0], a[:,1])
    plt.scatter(e[:, 0], e[:, 1])
    plt.scatter(i[:, 0], X[:, 1])
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(X[:, 0], X[:, 1])
    for i in range(M):
        plot_gauss_contour(mu[i],Sigma[i],0,1200,0,3000,"Sourmpinator")


    plt.show()

    # 2.) K-means algorithm:
    # TODO


    # 3.) Sampling from GMM
    # TODO

    pass


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
    # rd.seed(2361317)
    
    sanity_checks()
    main()
    
