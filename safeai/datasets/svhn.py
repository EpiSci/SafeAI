# Copyright (c) 2018 Episys Science, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SVHN dataset: 32x32 labeled digit images
# Reference: Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng
# Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on
# Deep Learning and Unsupervised Feature Learning 2011, http://ufldl.stanford.edu/housenumbers

import os

import scipy.io
import numpy as np
from tensorflow.keras.utils import get_file

# Todo: broken shape parameter
def load_data(shape=(32, 32)):
    dirname = os.path.join('datasets', 'svhn')
    base = 'http://ufldl.stanford.edu/housenumbers/'
    files = ['train_32x32.mat', 'test_32x32.mat']

    paths = []
    for fname in files:
        paths.append(get_file(fname,
                              origin=base + fname,
                              cache_subdir=dirname))
    train = scipy.io.loadmat(paths[0])
    test = scipy.io.loadmat(paths[1])

    def transpose_to_nwhc(x):
        assert x.shape[:3] == (32, 32, 3)
        nwhc = x.transpose((3, 0, 1, 2))
        assert nwhc.shape[1:4] == (32, 32, 3)
        return nwhc

    train_x, train_y = transpose_to_nwhc(train['X']), train['y']
    test_x, test_y = transpose_to_nwhc(test['X']), test['y']

    # integer '10' in labels represents digit 0 in image
    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0


    # Label index starts at 1 in SVHN dataset
    return (train_x, train_y), (test_x, test_y)
