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

import time
import os
# Disable debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from safeai.utils.distribution import kl_divergence_with_uniform


class UtilsTest(tf.test.TestCase):
    def test_kl_divergence_with_uniform(self):

        batch_size, num_classes = 3, 10
        target_distribution = tf.random_normal((batch_size, num_classes))
        kld = kl_divergence_with_uniform(target_distribution)
        self.assertEqual(kld.shape[0], 3)
        self.assertEqual(len(kld.shape), 1)

        batch_size, num_classes = 8, 300
        target_distribution = tf.random_normal((batch_size, num_classes))
        kld = kl_divergence_with_uniform(target_distribution)
        self.assertEqual(kld.shape[0], 8)
        self.assertEqual(len(kld.shape), 1)
