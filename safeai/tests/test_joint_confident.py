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

from safeai.models.joint_confident import confident_classifier

import tensorflow as tf

# Disable debugging logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


SAMPLE_SIZE = 512
BATCH_SIZE = 64
MNIST_IMAGE_DIM = (28, 28)
NOISE_DIM = 100
MNIST_NUM_CLASSES = 10
TRAIN_STEPS = 1


def dummy_mnist_input_fn():
    image = tf.random_uniform([SAMPLE_SIZE, 28, 28])
    noise = tf.random_uniform([SAMPLE_SIZE, 100])
    labels = tf.random_uniform([SAMPLE_SIZE, 1], maxval=9, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'image': image, 'noise': noise}, labels))
    return dataset.repeat().batch(BATCH_SIZE)


def make_mnist_confident_classifier_estimator():
    image_feature = tf.feature_column.numeric_column('image',
                                                     shape=MNIST_IMAGE_DIM)
    noise_feature = tf.feature_column.numeric_column('noise',
                                                     shape=NOISE_DIM)
    return tf.estimator.Estimator(
        model_fn=confident_classifier,
        params={
            'image': image_feature,
            'noise': noise_feature,
            'classes': MNIST_NUM_CLASSES,
            'discriminator': None,
            'generator': None,
            'classifier': None,
            'learning_rate': 0.001,
            'beta': 1.0,
        })


class JointConfidentModelTest(tf.test.TestCase):

    def dummy_mnist_input_fn(self):
        return {'image': tf.random_uniform([3, 28, 28]),
                'noise': tf.random_uniform([3, 100])}

    def test_confident_classifier(self):
        classifier = make_mnist_confident_classifier_estimator()
        classifier.train(input_fn=dummy_mnist_input_fn, steps=8)
        eval_results = classifier.evaluate(input_fn=dummy_mnist_input_fn, steps=1)

        # Couldn't find the references of these two:
        nll_loss = eval_results['loss']
        global_step = eval_results['global_step']

        accuracy = eval_results['accuracy']

        self.assertEqual(nll_loss.shape, ())
        # Need to investigate the behavior of global_step, along with minimize
        # self.assertEqual(12, global_step)
        self.assertEqual(accuracy.shape, ())

        predictions_generator = classifier.predict(self.dummy_mnist_input_fn)
        for _ in range(3):
            predictions = next(predictions_generator)
            self.assertEqual(predictions['probabilities'].shape, (10, ))
            self.assertEqual(predictions['classes'].shape, ())

    def confident_classifier_model_fn_helper(self, mode):

        num_samples = 3
        features = {'image': tf.random_uniform([num_samples, 28, 28]),
                    'noise': tf.random_uniform([num_samples, 100])}
        labels = tf.random_uniform([num_samples, 1], minval=0, maxval=9,
                                   dtype=tf.int32)
        image_feature = tf.feature_column.numeric_column(
                                        'image', shape=MNIST_IMAGE_DIM)
        noise_feature = tf.feature_column.numeric_column(
                                        'noise', shape=NOISE_DIM)
        params = {
            'image': image_feature,
            'noise': noise_feature,
            'classes': MNIST_NUM_CLASSES,
            'learning_rate': 0.001,
            'beta': 1.0,
        }
        spec = confident_classifier(features, labels, mode, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = spec.predictions
            self.assertAllEqual(predictions['probabilities'].shape,
                                (num_samples, 10))
            self.assertEqual(predictions['probabilities'].dtype, tf.float32)
            self.assertAllEqual(predictions['classes'].shape, (num_samples,))
            self.assertEqual(predictions['classes'].dtype, tf.int64)

        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = spec.loss
            self.assertAllEqual(loss.shape, ())
            self.assertEqual(loss.dtype, tf.float32)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = spec.eval_metric_ops
            self.assertAllEqual(eval_metric_ops['accuracy'][0].shape, ())
            self.assertAllEqual(eval_metric_ops['accuracy'][1].shape, ())
            self.assertEqual(eval_metric_ops['accuracy'][0].dtype, tf.float32)
            self.assertEqual(eval_metric_ops['accuracy'][1].dtype, tf.float32)

    def test_fn_train_mode(self):
        self.confident_classifier_model_fn_helper(tf.estimator.ModeKeys.TRAIN)

    def test_fn_eval_mode(self):
        self.confident_classifier_model_fn_helper(tf.estimator.ModeKeys.EVAL)

    def test_fn_predict_mode(self):
        self.confident_classifier_model_fn_helper(
            tf.estimator.ModeKeys.PREDICT)


class Benchmarks(tf.test.Benchmark):

    def benchmark_train_step_time(self):
        classifier = make_confident_classifier_estimator()
        # Run one step to warmup any use of the GPU.
        classifier.train(input_fn=dummy_input_fn, steps=1)

        num_steps = 1000
        name = 'train_step_time'

        start = time.time()
        classifier.train(input_fn=dummy_input_fn, steps=num_steps)
        end = time.time()

        wall_time = (end - start) / num_steps
        self.report_benchmark(
            iters=num_steps,
            wall_time=wall_time,
            name=name,
            extras={
                'examples_per_sec': BATCH_SIZE / wall_time
            })


if __name__ == "__main__":
    tf.test.main()
