#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

def draw(train_llh_sep, dev_llh_sep, train_llh_tie, dev_llh_tie, model_num):

    m, iter = train_llh_sep.shape   # number of Gaussian, number of iteration

    # plot seperated covariance
    for j in range(m):
        plt.subplot(2, m, j + 1)
        x = np.linspace(1, iter, num = iter)
        ty = train_llh_sep[j]
        plt.plot(x, ty, label = 'train')
        dy = dev_llh_sep[j]
        plt.plot(x, dy, label = 'dev')
        plt.xlabel('#Iteration')
        plt.ylabel('Log Likelyhood')
        plt.title('#Gaussian: ' + str(model_num[j]))
        plt.legend(loc = 'upper right', borderaxespad = 0)

    # plot tied covariance
    for j in range(m):
        plt.subplot(2, m, m + j + 1)
        x = np.linspace(1, iter, num = iter)
        ty = train_llh_tie[j]
        plt.plot(x, ty, label = 'train')
        dy = dev_llh_tie[j]
        plt.plot(x, dy, label = 'dev')
        plt.xlabel('#Iteration')
        plt.ylabel('Log Likelyhood')
        plt.title('#Gaussian: ' + str(model_num[j]))
        plt.legend(loc = 'upper right', borderaxespad = 0)


    plt.suptitle('Top row: seperated\n Bottom row: tied')
    plt.subplots_adjust(left = 0.125, bottom = 0.15, right = 0.9, top = 0.9, wspace = 0.3, hspace = 0.3)
    plt.show()
