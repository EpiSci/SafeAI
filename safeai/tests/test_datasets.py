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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from safeai.datasets import svhn
from safeai.datasets import cifar10
from safeai.datasets import stl10
from safeai.datasets import tinyimagenet200
from safeai.datasets import mnist
from safeai.datasets import fashion_mnist


# Todo: Tests on dataset module needs to be more specific
class DatasetsTest(tf.test.TestCase):

    def check_image_shapes(self, data, image_shape, label_shape):
        (train_x, train_y), (test_x, test_y) = data
        self.assertEqual(train_x.shape[1:], image_shape)
        self.assertEqual(train_y.shape[1:], label_shape)
        self.assertEqual(test_x.shape[1:], image_shape)
        self.assertEqual(test_y.shape[1:], label_shape)

    # Todo: NWHC, NCHW based on keras.backend.image_data_foramt
    # Assumes 'channels_last' data format (Num_samples, Width, Height, Channel)
    # Todo: Need these datasets to be cached in ci settings Mon 19 Nov 2018 12:24:01 PM KST
    def test_svhn(self):
        svhn_data = svhn.load_data()
        self.check_image_shapes(svhn_data, (32, 32, 3), (1,))

    def test_cifar10(self):
        cifar10_data = cifar10.load_data()
        self.check_image_shapes(cifar10_data, (32, 32, 3), (1,))

    def test_stl10(self):
        stl10_data = stl10.load_data()
        self.check_image_shapes(stl10_data, (96, 96, 3), (1,))

    def test_stl10_32x32(self):
        stl10_data = stl10.load_data(shape=(32, 32))
        self.check_image_shapes(stl10_data, (32, 32, 3), (1,))

    def test_tiny_imagenet(self):
        tim = tinyimagenet200.load_data()
        self.check_image_shapes(tim, (64, 64, 3), (1,))

    def test_tiny_imagenet_32x32(self):
        tim = tinyimagenet200.load_data(shape=(32, 32))
        self.check_image_shapes(tim, (32, 32, 3), (1,))

    def test_mnist(self):
        mnist_data = mnist.load_data()
        self.check_image_shapes(mnist_data, (28, 28), ())

    def test_fashion_mnist(self):
        fashion_mnist_data = fashion_mnist.load_data()
        self.check_image_shapes(fashion_mnist_data, (28, 28), ())
