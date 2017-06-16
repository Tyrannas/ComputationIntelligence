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
    plt.show()


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
    for i in range(max_iter):

        rm1 = alpha*multivariate_normal(mu, sigma).pdf(X)
        rm = rm1/(np.sum(rm1))

        print(np.shape(rm1))
        Nm = np.sum(rm)
        mu = 1/Nm*np.dot(rm,X)

        sigma = 1/Nm*np.dot(np.dot(rm,(X-mu)),(X-mu).T)

        alpha = Nm/len(X)
        print(np.shape(sigma),np.shape(alpha),np.shape(mu))

        sigmainv = np.invert(sigma)


        L = np.sum(np.log(alpha*multivariate_normal(mu, sigma).pdf(X)))
        if np.abs(L1-L) < 0.1:
            break
    return alpha,mu,sigma,L


def normal(X,sigma,mu,sigmainv,d):
    return np.exp(-0.5*np.dot(np.dot(d,sigmainv),(X-mu)))/np.sqrt(2*pi*np.linalg.norm(sigma))


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
    alpha_0 = np.array([0.1,0.2,0.3,0.2,0.2])
    mu_0 = np.random.random((5,))
    Sigma_0 = np.identity(M)
    max_iter = 10

    alpha, mu, Sigma, L =  EM(X[0].T, M, alpha_0, mu_0, Sigma_0, max_iter)
    print(alpha,mu,Sigma,L)
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
    rd.seed(23434345)
    
    sanity_checks()
    main()
    
