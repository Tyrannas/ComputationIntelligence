#!/usr/bin/env python3
# Filename skeleton_HW6.py
# Author: Christian Knoll, Philipp Gabler
# Edited: 20.6.2017

import numpy as np
import numpy.random as rd
import math


## -------------------------------------------------------
## ---------------- HELPER FUNCTIONS ---------------------
## -------------------------------------------------------

def is_probability_distribution(p):
    """Check if p represents a valid probability distribution."""
    p = np.array(p)
    return np.isclose(p.sum(), 1.0) and np.logical_and(0.0 <= p, p <= 1.0).all()


def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over
    the support X.

    X ... Support of the RV -- (S,)
    PM ... Probabilities P(X) -- (S,)
    N ... Number of samples -- scalar
    """

    X, PM = np.asarray(X), np.asarray(PM)

    assert is_probability_distribution(PM)

    y = np.zeros(N, dtype=X.dtype)
    cumulativePM = np.cumsum(PM)  # build CDF based on PMF
    offset = rd.uniform(0, 1) * (1 / N)  # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offset, 1 + offset, 1 / N)  # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]:  # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return rd.permutation(y)  # permutation of all samples

def assess_hmm(path,prob):
    prod = 1
    sum = 0
    for j in range(len(path)):
        for i, o in enumerate(path[j,:]):
            prod *= prob[int(o), i]
        prod *= prob[j, i + 1]  # the last state is not in the path matrix
        sum += prod
    return sum

class HMM:
    def __init__(self, A, B, pi):
        """Represents a Hidden Markov Model with given parameters.

        pi ... Prior/initial probabilities -- (N_s,)
        A ... Transition probabilities -- (N_s, N_s)
        B ... Emission probabilities -- (N_s, N_o)
        """

        A, B, pi = np.asarray(A), np.asarray(B), np.asarray(pi)

        assert A.shape[0] == B.shape[0] == pi.shape[0]
        assert is_probability_distribution(pi)
        assert all(is_probability_distribution(p) for p in A)
        assert all(is_probability_distribution(p) for p in B)

        self.A = A
        self.B = B
        self.pi = pi
        self.N_s = pi.shape[0]  # number of states
        self.N_o = B.shape[1]  # number of possible observations

    def viterbi_discrete(self, X):
        """Viterbi algorithm for an HMM with discrete emission probabilities.
        Returns the optimal state sequence q_opt for a given observation sequence X.

        X ... Observation sequence -- (N,)
        """
        # init the variables
        # observations
        X = np.asarray(X)
        n_obs = len(X)
        # probabilities of the most likely path
        probs = np.zeros((self.N_s, n_obs))
        # most likely path
        path = np.zeros((self.N_s, n_obs - 1))
        # q_opt is the optimal state sequence; this is a default value
        q_opt = np.zeros(n_obs, dtype=int)

        # values for initial state
        probs[:, 0] = self.pi.T * self.B[:, X[0]]
        # improved initialization
        # path[:, 0] = probs[:, 0].argmax()

        # for every other observation
        for o in range(1, n_obs):
            # for every state
            for s in range(self.N_s):
                # compute the probability of the transition
                trans_prob = np.array(probs[:, o - 1] * self.A[:, s])
                # get the highest probability
                probs[s, o] = self.B[s, X[o]] * trans_prob.max()
                # get the state with highest probability
                path[s, o - 1] = trans_prob.argmax()

        # we then start from the end with the state with maximum probability
        q_opt[n_obs - 1] = probs[:, n_obs - 1].argmax()
        # then getting back to zero
        for o in range(n_obs - 1, 0, -1):
            q_opt[o - 1] = path[q_opt[o], o - 1]

        return q_opt,path,probs

    def sample(self, N):
        """Draw a random state and corresponding observation sequence of length N from the model."""
        # TODO: implement sampling from HMM
        PM = [1./N for i in range(N)]
        X = [np.random.randint(0,2) for i in range(N)]
        Y = sample_discrete_pmf(X, PM, N)
        qopt,nan,non  = self.viterbi_discrete(Y)
        return qopt, Y


## -------------------------------------------------------
## ------------- START OF  ASSIGNMENT 6 ------------------
## -------------------------------------------------------
def main():
    # define states
    states = ['s', 'r', 'f']  # 3 States: Sun, Rain, Fog

    # define HMM 1
    A1 = np.array([[0.8, 0.05, 0.15],
                   [0.2, 0.6, 0.2],
                   [0.2, 0.3, 0.5]])  # Transition Prob. Matrix
    B1 = np.array([[0.1, 0.9],
                   [0.8, 0.2],
                   [0.3, 0.7]])  # Emission Prob. rows correspond to states, columns to observations
    pi1 = np.array([1 / 3, 1 / 3, 1 / 3])  # Prior
    hmm1 = HMM(A1, B1, pi1)

    # define HMM 2
    A2 = np.array([[0.6, 0.20, 0.20],
                   [0.05, 0.7, 0.25],
                   [0.05, 0.6, 0.35]])  # Transition Prob. Matrix
    B2 = np.array([[0.3, 0.7],
                   [0.95, 0.05],
                   [0.5, 0.5]])  # Emission prob. rows correspond to states, columns to observations
    pi2 = np.array([1 / 3, 1 / 3, 1 / 3])  # Prior
    hmm2 = HMM(A2, B2, pi2)

    # define observation sequences
    X_test = np.array([0, 0, 0, 0, 0, 0, ])
    X1 = np.array([0, 0, 1, 1, 1, 0])  # 0 = umbrella
    X2 = np.array([0, 0, 1, 1, 1, 0, 0])  # 1 = no umbrella

    # 1.1.) apply Viterbi to find the optimal state sequence and assign the corresponding states
    # TODO: implement in HMM.viterbi_discrete

    # --- example usage of viterbi_discrete:
    optimal_state_sequence1, path1, prob1 = hmm1.viterbi_discrete(X2)

    # print(optimal_state_sequence1)
    print([states[i] for i in optimal_state_sequence1])

    #count the cost of more paths to see the better model

    optimal_state_sequence2, path2, prob2 = hmm2.viterbi_discrete(X2)

    # print(optimal_state_sequence2)
    print([states[i] for i in optimal_state_sequence2])



    # 1.2.) Sequence Classification
    # TODO
    evalhmm1 = assess_hmm(path1,prob1)
    print("Likelihood of hmm1:",evalhmm1)

    evalhmm2 = assess_hmm(path2, prob2)
    print("Likelihood of hmm2:",evalhmm2)


    # 1.3.) Sample from HMM
    # TODO: implement in HMM.sample
    Q, X = hmm1.sample(6)
    print(X, Q)


    # 1.4) Markov Model
    # define HMM 1
    A1 = np.array([[0.8, 0.05, 0.15],
                   [0.2, 0.6, 0.2],
                   [0.2, 0.3, 0.5]])  # Transition Prob. Matrix
    pi1 = np.array([1 / 3, 1 / 3, 1 / 3])  # Prior

    pi2 = np.dot(pi1,A1)
    pi3 = np.dot(pi2, A1)
    print(" Day 1:",pi1,"\n","Day 2:",pi2,"\n","Day 3:",pi3)

    pi1 = [0.4,0.2,0.4]
    pin = 0
    converge = np.array([2, 2, 2])
    while converge.all() > 0.001:
        pin =  np.dot(pi1,A1)
        converge = abs(pin -pi1)
        pi1 = pin

    print("Final State",pin)

if __name__ == '__main__':
    main()