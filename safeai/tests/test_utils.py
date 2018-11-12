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
