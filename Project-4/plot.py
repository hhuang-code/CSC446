#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

def draw(train_llh, dev_llh, model_num):

    m, iter = train_llh.shape   # number of Gaussian, number of iteration

    # For each number of model
    for j in range(m):
        plt.subplot(m, 2, 2 * j + 1)
        x = np.linspace(1, iter, num = iter)
        ty = train_llh[j]
        plt.plot(x, ty, color='blue', label = 'train')
        plt.xlabel('#Iteration')
        plt.ylabel('Log Likelyhood of Train Set')
        plt.title('#State: ' + str(model_num[j]))
        plt.legend(loc = 'upper right', borderaxespad = 0)

        plt.subplot(m, 2, 2 * j + 2)
        dy = dev_llh[j]
        plt.plot(x, dy, color = 'red', label = 'dev')
        plt.xlabel('#Iteration')
        plt.ylabel('Log Likelyhood of Dev Set')
        plt.title('#State: ' + str(model_num[j]))
        plt.legend(loc = 'upper right', borderaxespad = 0)

    plt.subplots_adjust(left = 0.125, bottom = 0.15, right = 0.9, top = 0.9, wspace = 0.3, hspace = 0.6)
    plt.show()