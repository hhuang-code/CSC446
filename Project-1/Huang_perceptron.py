#!/usr/bin/python

from __future__ import print_function

import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def split_data(df):
    # convert to 123 features + 1 dummy feature
    feat = np.zeros((len(df), 124))
    cnt = 0
    for sample in df:  # for every sample
        for i in range(1, len(sample)):  # for every feature of a sample
            if not pd.isnull(sample[i]):
                [idx, value] = sample[i].split(':')
                feat[cnt][int(idx) - 1] = value
        cnt += 1
    assert (cnt == len(df))
    feat[:, -1] = 1  # set dummy feature

    gt = df[:, 0].astype(np.int)

    return feat, gt

def load_data(data_dir = '/u/cs246/data/adult'):
    # load data
    train_df = pd.read_table(os.path.join(data_dir, 'a7a.train'), header = None, delim_whitespace = True, index_col = False).as_matrix()
    dev_df = pd.read_table(os.path.join(data_dir, 'a7a.dev'), header = None, delim_whitespace = True, index_col = False).as_matrix()
    test_df = pd.read_table(os.path.join(data_dir, 'a7a.test'), header = None, delim_whitespace = True, index_col = False).as_matrix()

    train_feat, train_gt = split_data(train_df)
    dev_feat, dev_gt = split_data(dev_df)
    test_feat, test_gt = split_data(test_df)

    return train_feat, train_gt, dev_feat, dev_gt, test_feat, test_gt

def draw(x, y, title, x_label, y_label):
    plt.plot(x, y)
    # axis ranges
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def train(train_feat, train_gt, dev_feat, dev_gt, iterations, lr, need_dev = False):
    # iterations be int, and lr be float
    assert (isinstance(args['iterations'], (int, long)) and isinstance(args['lr'], (float)))

    # initialize weights
    w = np.zeros(124)  # the last one is bias

    best_w = None
    best_acc = 0.0
    accs = []

    for _ in range(iterations):
        for i in range(len(train_feat)):
            # # update weights
            if train_gt[i] != np.sign(np.dot(w.T, train_feat[i])):
                w += lr * train_gt[i] * train_feat[i]
        if need_dev:
            acc = test(dev_feat, dev_gt, w)
            accs.append(acc)
            if acc > best_acc:
                best_acc, best_w = acc, np.copy(w)  # deep copy

    if need_dev:
        w = best_w

    return w, accs

def test(test_feat, test_gt, w):
    # using matrix multiplication for testing
    res = np.squeeze(np.dot(np.expand_dims(w, axis=1).T, test_feat.T))
    res = map(lambda x: np.sign(x), res)

    # calculate accuracy
    acc = float(np.sum(res == test_gt)) / float(len(res))

    return acc


if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type= int, default = 20)
    parser.add_argument('--lr', type = float, default = 1.0)
    parser.add_argument('--noDev', action='store_true', default = False)
    args = vars(parser.parse_args())

    # load data
    train_feat, train_gt, dev_feat, dev_gt, test_feat, test_gt = load_data()

    # training
    w = None    # declaration
    if args['noDev'] is True:
        w, _ = train(train_feat, train_gt, dev_feat, dev_gt, args['iterations'], args['lr'], need_dev = False)
    else:
        w, accs = train(train_feat, train_gt, dev_feat, dev_gt, args['iterations'], args['lr'], need_dev = True)

    acc = test(test_feat, test_gt, w)

    print('Test accuracy: ', acc)
    print('Feature weights (bias last):', end = '')
    map(lambda x: print(' ', x, end = ''), w)
    print()

    if args['noDev'] is False:
        # plot accuracy v. #iterations
        draw(np.linspace(1, args['iterations'], num = args['iterations']), accs, 'Accuracy v. #Iterations', '#Iterations', 'Accuracy')
