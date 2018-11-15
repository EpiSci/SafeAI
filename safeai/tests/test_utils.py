from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from safeai.utils.distribution import kl_divergence_with_uniform
import tensorflow as tf


class UtilsTest(tf.test.TestCase):

    def test_kl_divergence_with_uniform(self):
        with self.test_session() as sess:
            batch_size, num_classes = 3, 10
            target_distribution = tf.nn.softmax(tf.random_normal((batch_size, num_classes)))
            kld = kl_divergence_with_uniform(target_distribution)

            self.assertEqual(kld.shape[0], 3)
            self.assertEqual(len(kld.shape), 1)

            sess.run(tf.global_variables_initializer())
            self.assertNotAllClose(tf.zeros((3, 1)), tf.reshape(kld, (3, 1)))

        with self.test_session() as sess:
            batch_size, num_classes = 8, 300
            target_distribution = tf.nn.softmax(tf.random_normal((batch_size, num_classes)))
            kld = kl_divergence_with_uniform(target_distribution)

            self.assertEqual(kld.shape[0], 8)
            self.assertEqual(len(kld.shape), 1)

            sess.run(tf.global_variables_initializer())
            self.assertNotAllClose(tf.zeros((8, 1)), tf.reshape(kld, (8, 1)))
