#!/usr/bin/python

import numpy as np

import matplotlib.pyplot as plt

from Huang_Hao_hw3 import *

def draw(x, dev, test, title, x_label, y_label):
    plt.xscale('log')
    plt.plot(x, dev, label = 'dev')
    plt.plot(x, test, label = 'test')
    # axis ranges
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # legend
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.show()

if __name__ == '__main__':

    # load data
    train_feat, train_gt, dev_feat, dev_gt, test_feat, test_gt = load_data()

    # training
    C = np.logspace(-3, 4, num = 20, endpoint = True)
    dev_acc_arr = []
    test_acc_arr = []
    for c in C:
        w = train(train_feat, train_gt, 5, c, 0.1)
        dev_acc_arr.append(test(dev_feat, dev_gt, w))
        test_acc_arr.append(test(test_feat, test_gt, w))

    # plot
    draw(C, dev_acc_arr, test_acc_arr, 'Accuracy v. Capacity', 'Capacity', 'Accuracy')
