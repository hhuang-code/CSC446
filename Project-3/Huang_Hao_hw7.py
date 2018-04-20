#!/usr/bin/python

import os.path
import numpy as np

from plot import *

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

    return train_set, dev_set

def prob(x, mu, sigma):
    dim = x.shape[1]    # sample dimension
    scalar = np.asscalar(1 / np.sqrt(np.power(2 * np.pi, dim) * np.linalg.det(sigma)))
    index = -0.5 * np.dot(np.dot(x - mu, np.linalg.inv(sigma)), np.transpose(x - mu)).diagonal()   # shape: (N)

    return scalar * np.exp(index)

def em(train_set, dev_set, model_num, max_iter_num, type = 'seperated'):
    train_num, _ = train_set.shape    # number of samples in train set
    dev_num, _ = dev_set.shape  # number of samples in dev set

    train_llh = np.empty([len(model_num), max_iter_num])
    dev_llh = np.empty([len(model_num), max_iter_num])
    for id, m in enumerate(model_num):
        # initialize w
        w = np.random.rand(train_num, m)
        w = w / np.sum(w, axis = 1)[:, None]  # normalize
        for iter in range(max_iter_num):
            # M-step
            phi = np.mean(w, axis = 0)  # shape: (m)
            mu = np.dot(np.transpose(w), train_set) / np.sum(w, axis = 0)[:, None]  # shape: (m, 2)
            sigma = np.empty([m, train_set.shape[1], train_set.shape[1]])   # shape: (m, 2, 2)
            for j in range(m):
                s = np.dot(np.multiply(np.repeat(w[:, j][None, :], 2, axis = 0), np.transpose(train_set - mu[j])), train_set - mu[j])   # shape: (2, 2)
                sigma[j] = s / np.sum(w[:, j])

            # handle 'tied' covariance
            if type == 'tied':
                sigma = np.repeat(np.mean(sigma, axis = 0)[None, :], m, axis = 0)

            # E-step
            for j in range(m):
                w[:, j] = prob(train_set, mu[j], sigma[j])
            w = np.multiply(w, phi)
            w = w / np.sum(w, axis = 1)[:, None] # normalize

            # log likelyhood for train set
            llh = np.empty([train_num, m])
            for j in range(m):
                llh[:, j] = prob(train_set, mu[j], sigma[j]) * phi[j]
            train_llh[id, iter] = np.sum(np.log(np.sum(llh, axis = 1))) / train_num

            # log likelyhood for dev set
            llh = np.empty([dev_num, m])
            for j in range(m):
                llh[:, j] = prob(dev_set, mu[j], sigma[j]) * phi[j]
            dev_llh[id, iter] = np.sum(np.log(np.sum(llh, axis = 1))) / dev_num

    return train_llh, dev_llh


if __name__ == '__main__':
    # load data
    train_set, dev_set = load_data('./points.dat')  # shape: (N, 2)

    model_num = [3, 6, 12, 25, 50]   # number of Gaussian
    max_iter_num = 300

    train_llh_sep, dev_llh_sep = em(train_set, dev_set, model_num, max_iter_num, type = 'seperated')
    train_llh_tie, dev_llh_tie = em(train_set, dev_set, model_num, max_iter_num, type = 'tied')

    draw(train_llh_sep, dev_llh_sep, train_llh_tie, dev_llh_tie, model_num)