#!/usr/bin/python

import os
from plot import *

import numpy as np

def load_data(filename = '.points.dat'):
    # check file existence
    if not os.path.exists(filename):
        raise Exception('No such file: ' + filename)

    # read all lines
    lines = [line.rstrip('\n').strip() for line in open(filename)]

    # split each line into two variables
    dataset = np.array([np.array([float(n) for n in x.split()]) for x in lines])

    # split train and dev set
    train_set = dataset[:int(0.9 * len(dataset))]
    dev_set = dataset[int(0.9 * len(dataset))::]

    return train_set.T, dev_set.T

def Gaussian(x, m, s):
    """
    Given mean and covariance, compute the multi-dimensinal Gaussian probability of x
    :param x: a point, viewd as a column vector
    :param m: mean, the same shape of x
    :param s: covariance, a square matrix
    :return: a probability value at point x
    """
    x, m = x[:, None], m[:, None]   # expand to be a column vector
    dim = len(x)    # sample dimension
    scalar = np.asscalar(1 / np.sqrt(np.power(2 * np.pi, dim) * np.linalg.det(s)))
    if np.linalg.det(s) == 0:
        s = np.eye(dim)
    index = -0.5 * np.dot(np.dot(np.transpose(x - m), np.linalg.inv(s)), x - m).diagonal()   # shape: (N)

    return scalar * np.exp(index)


def Initialization(dim, state_num):
    """
    Initialize start states, transition probability, and emission probability
    :param dim: dimension of input data
    :param state_num: number of hidden states
    :return: start states, transition matrix, mean and covariance
    """
    # start states, a column vector of state_num elements
    PI = np.ones((state_num, 1)) / state_num

    # Transition matrix, trans(i,j) = P(s_t = j | s_{t-1} = i)
    trans = np.ones((state_num, state_num))
    trans /= np.sum(trans, 1)

    # Assume emission probability for each hidden state is a Gaussian distribution
    # mu is mean, a dim * state_num matrix, and each column represents for a state
    # sigma is a list of covariance with the shape of dim * dim, and each elements represents for a state
    mu = np.random.rand(dim, state_num)
    # Because each hidden state is independent, so the covariance matrix is digonal matrix
    sigma = [np.eye(dim) for _ in range(state_num)]

    return PI, trans, mu, sigma


def createAlpha(dataset, PI, trans, mu, sigma):
    """
    Create alpha matrix, a K * N matrix; alpha(i,j) is the forward probability of state i at timestep j
    :param dataset: each column is point in dataset
    :param PI: state states
    :param trans: transition matrix
    :param mu: mean, a dim * state_num matrix
    :param sigma: a list of covariance matrix
    :return: alpha matrix
    """
    N = np.size(dataset, 1)
    state_num = np.size(trans, 0)
    # Forward matrix; alpha(i,j) means i state and j timestep
    alpha = np.zeros((state_num, N))
    col_sum = np.zeros(N)

    # Build the first column (first timestep) of alpha
    for i in range(state_num):
        alpha[i, 0] = PI[i] * Gaussian(dataset[:, 0], mu[:, i], sigma[i])
    col_sum[0] = np.sum(alpha[:, 0])
    # Normalization column
    alpha[:, 0] = alpha[:, 0] / np.sum(alpha[:, 0])

    # Build all following columns (timestep)
    for t in range(1, N):
        for i in range(state_num):
            for j in range(state_num):
                alpha[i, t] += alpha[j, t - 1] * trans[j, i]
            alpha[i, t] *= Gaussian(dataset[:, t], mu[:, i], sigma[i])
        col_sum[t] = np.sum(alpha[:, t])
        # Normalization of each column
        alpha[:, t] = alpha[:, t] / np.sum(alpha[:, t])

    return alpha, col_sum


def createBeta(dataset, PI, trans, mu, sigma):
    """
    Create beta matrix, a K * N matrix; beta(i,j) is the forward probability of state i at timestep j
    :param dataset: each column is point in dataset
    :param PI: state states
    :param trans: transition matrix
    :param mu: mean, a dim * state_num matrix
    :param sigma: a list of covariance matrix
    :return: beta matrix
    """
    N = np.size(dataset, 1)
    state_num = np.size(PI, 0)
    # Backward matrix; Beta(i,j) means i state and j timestep
    beta = np.zeros((state_num, N))

    # Build the last column (first timestep) of beta
    for i in range(state_num):
        beta[i, N - 1] = 1.0

    # Build the previous column (timestep)
    for t in range(N - 2, -1, -1):
        for i in range(state_num):
            for j in range(state_num):
                beta[i, t] += beta[j, t + 1] * trans[i, j] * Gaussian(dataset[:, t + 1], mu[:, j], sigma[j])
        # Normalization of each column
        beta[:, t] /= np.sum(beta[:, t + 1])

    return beta

def Estep(dataset, PI, trans, mu, sigma):
    """
    Define and estimate parameters
    :param dataset: each column is point in dataset
    :param PI: state states
    :param trans: transition matrix
    :param mu: mean, a dim * state_num matrix
    :param sigma: a list of covariance matrix
    :return: alpha * beta, a probability matrix, and an auxiliary vector
    """
    N = np.size(dataset, 1)
    state_num = np.size(PI, 0)
    # probabilities of generating each point at each timestep
    prob = np.zeros((state_num, state_num, N))

    # Compute forward and backward matrix
    alpha, col_sum = createAlpha(dataset, PI, trans, mu, sigma)
    beta = createBeta(dataset, PI, trans, mu, sigma)

    # For each time step
    for t in range(1, N):
        prob[:, :, t] = (1 / col_sum[t]) * np.dot(alpha[:, t - 1][None].T, beta[:, t][None]) * trans
        # For each hidden state
        for state in range(state_num):
            prob[:, state, t] *= Gaussian(dataset[:, t], mu[:, state], sigma[state])

    return alpha * beta, prob, col_sum


def Mstep(dataset, ab, prob):
    """
    Update parameters: probability of start states, transition matrix, mean and covariance
    :param dataset: each column is point in dataset
    :param ab: alpha * beta
    :param prob: a matrix, probabilities of generating each point at each timestep
    :return: sprobability of tart states, transition matrix, transition matrix, mean and covariance
    """
    dim = np.size(dataset, 0)
    state_num = np.size(ab, 0)

    PI = (ab[:, 0] / np.sum(ab[:, 0]))[None].T
    # Sum along timestep
    t_sum = np.sum(prob[:, :, 1:], axis = 2)
    trans = t_sum / np.sum(t_sum, axis = 1)[None].T

    mu = np.zeros((dim, state_num))
    # Sum along timestep
    ab_t_sum = np.sum(ab, axis = 1)[None].T
    sigma = []
    # For each hidden state, update parameters of mean and covariance
    for i in range(state_num):
        mu[:, i] = np.sum(ab[i, :] * dataset, axis = 1) / ab_t_sum[i]
        # Data points minus mean
        x_m = dataset - mu[:, i][None].T
        # Compute covariance for each state, and add it to a list
        sigma.append(np.dot(x_m, (x_m * (ab[i, :][None])).T) / ab_t_sum[i])

    return PI, trans, mu, sigma

def em_hmm(train_set, dev_set, model_num, max_iter_num):
    _, train_num = train_set.shape  # number of samples in train set
    dim, dev_num = dev_set.shape  # number of samples in dev set

    train_llh = np.empty([len(model_num), max_iter_num])
    dev_llh = np.empty([len(model_num), max_iter_num])

    for id, m in enumerate(model_num):
        # Initializaltion
        PI, trans, mu, sigma = Initialization(dim, m)
        for iter in range(max_iter_num):
            # E-step
            ab, prob, col_sum = Estep(train_set, PI, trans, mu, sigma)
            # M-step
            PI, trans, mu, sigma = Mstep(train_set, ab, prob)

            # log likelyhood for train set
            train_llh[id, iter] = np.sum(np.log(col_sum)) / train_num
            print('Number of states: %d, number of iteration: %d log likelyhood for train set: %f'
                  % (m, (iter + 1), np.sum(np.log(col_sum)) / train_num))

            # log likelyhood for dev set
            ab_dev, prob_dev, col_sum_dev = Estep(dev_set, PI, trans, mu, sigma)
            dev_llh[id, iter] = np.sum(np.log(col_sum_dev)) / dev_num
            print('Number of states: %d, number of iteration: %d, log likelyhood for dev set: %f'
                  % (m, (iter + 1), np.sum(np.log(col_sum_dev)) / dev_num))

    return train_llh, dev_llh


if __name__ == '__main__':
    input_file = open('points.dat')
    lines = input_file.readlines()
    allData = np.array([line.strip().split() for line in lines]).astype(np.float)
    (m, n) = np.shape(allData)

    train_set, dev_set = load_data('points.dat')

    model_num = [2, 4, 6, 8]  # number of Gaussian (hidden states)
    max_iter_num = 25

    train_llh, dev_llh = em_hmm(train_set, dev_set, model_num, max_iter_num)

    draw(train_llh, dev_llh, model_num)
