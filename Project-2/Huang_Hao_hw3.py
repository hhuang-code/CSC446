#!/usr/bin/python

from __future__ import print_function

import os
import argparse

import pandas as pd
import numpy as np

def split_data(df):
    # convert to 123 features + 1 dummy feature
    feat = np.zeros((len(df), 124))
    cnt = 0
    for sample in df:  # for every sample
        for i in range(1, len(sample)):  # for every feature of a sample
            if not pd.isnull(sample[i]):
                [idx, value] = sample[i].split(':')
                feat[cnt][int(idx)] = value
        cnt += 1
    assert (cnt == len(df))

    feat[:, 0] = 1  # set dummy feature

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

def train(train_feat, train_gt, epochs, C, lr):
    # epochs be int, and capacity be float
    assert (isinstance(epochs, (int, long)) and isinstance(C, float))

    # initialize weights
    w = np.zeros(124)  # the last one is bias

    # the number of samples
    N = len(train_feat)

    for _ in range(epochs):
        for i in range(N):
            # update weights
            if 1 - train_gt[i] * (np.dot(w.T, train_feat[i])) > 0:
                w[1:] -= lr * (1.0 / N * w[1:] - C * train_gt[i] * train_feat[i][1:])   # update weights
                w[0] -= lr * (-C * train_gt[i]) # update bias
            else:
                w[1:] -= lr * 1.0 / N * w[1:]   # update weights

    return w

def test(test_feat, test_gt, w):
    # using matrix multiplication for testing
    res = np.squeeze(np.dot(np.expand_dims(w, axis = 1).T, test_feat.T))
    res = map(lambda x: np.sign(x), res)

    # calculate accuracy
    acc = float(np.sum(res == test_gt)) / float(len(res))

    return acc

if __name__ == '__main__':

    np.set_printoptions(precision = 16, suppress = True)

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type= int, default = 1)
    parser.add_argument('--capacity', type = float, default = 0.868)
    parser.add_argument('--lr', type = float, default = 0.1)
    args = vars(parser.parse_args())

    # load data
    train_feat, train_gt, dev_feat, dev_gt, test_feat, test_gt = load_data()

    # training
    w = train(train_feat, train_gt, args['epochs'], args['capacity'], args['lr'])

    print('EPOCHS: ', args['epochs'])
    print('CAPACITY: ', args['capacity'])
    print('TRAINING_ACCURACY: ', test(train_feat, train_gt, w))
    print('TEST_ACCURACY: ', test(test_feat, test_gt, w))
    print('DEV_ACCURACY: ', test(dev_feat, dev_gt, w))
    print('FINAL_SVM: ', w)
