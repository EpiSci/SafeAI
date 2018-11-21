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

# STL-10 dataset, contains 96x96 labeled, unlabeled images
# Reference:: Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer Networks 
# in Unsupervised Feature Learning AISTATS, 2011, http://cs.stanford.edu/~acoates/stl10
import os
import tarfile

import scipy.io
import skimage
import numpy as np
from tensorflow.keras.utils import get_file


def load_data(shape=(96, 96)):
    width, height = shape

    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("resize_shape must be 2d tuple\
                        consists of (width, heights)")
    if width > 96 or height > 96:
        raise ValueError("Cannot be resized bigger than the original image")

    dirname = os.path.join('datasets', 'stl-10')
    matlab_dir_name = 'stl10_matlab'
    tarname = 'stl10_matlab.tar.gz'
    base = 'http://ai.stanford.edu/~acoates/stl10/'
    train_data_name = 'train.mat'
    test_data_name = 'test.mat'

    stl_path = get_file(tarname,
                        origin=base + tarname,
                        cache_subdir=dirname)

    if not os.path.isdir(os.path.join(os.path.dirname(stl_path),
                                      matlab_dir_name)):
        tarf = tarfile.open(stl_path)
        tarf.extractall(os.path.dirname(stl_path))

    train_path = os.path.join(os.path.dirname(stl_path),
                              matlab_dir_name,
                              train_data_name)
    test_path = os.path.join(os.path.dirname(stl_path),
                             matlab_dir_name,
                             test_data_name)

    train = scipy.io.loadmat(train_path)
    test = scipy.io.loadmat(test_path)

    (train_x, train_y), (test_x, test_y) =\
        (train['X'], train['y']), (test['X'], test['y'])

    # Reshape column-major order images
    train_x = np.reshape(train_x, (-1, 3, 96, 96))
    test_x = np.reshape(test_x, (-1, 3, 96, 96))

    # Transpose to NCWH data format
    train_x = np.transpose(train_x, (0, 3, 2, 1))
    test_x = np.transpose(test_x, (0, 3, 2, 1))

    # integer '10' in labels represents class 0 in image
    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0

    if shape == (96, 96):
        return (train_x, train_y), (test_x, test_y)

    cache_dir_train = os.path.join(
        os.path.dirname(stl_path),
        "cached{}x{}_train.npy".format(width, height))

    cache_dir_test = os.path.join(
        os.path.dirname(stl_path),
        "cached{}x{}_test.npy".format(width, height))

    if os.path.exists(cache_dir_train) and os.path.exists(cache_dir_test):
        train_x = np.load(cache_dir_train)
        test_x = np.load(cache_dir_test)
        return (train_x, train_y), (test_x, test_y)

    resized_train = np.empty((5000, width, height, 3), dtype='uint8')
    resized_test = np.empty((8000, width, height, 3), dtype='uint8')

    for i, image in enumerate(train_x):
        resized_train[i, :, :, :] = skimage.transform.resize(
            image,
            (width, height, 3),
            mode='constant',
            anti_aliasing=True,
            preserve_range=True)

    for i, image in enumerate(test_x):
        resized_test[i, :, :, :] = skimage.transform.resize(
            image,
            (width, height, 3),
            mode='constant',
            anti_aliasing=True,
            preserve_range=True)

    np.save(cache_dir_train, resized_train)
    np.save(cache_dir_test, resized_test)

    return (resized_train, train_y), (resized_test, test_y)
