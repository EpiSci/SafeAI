
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from safeai.models.joint_confident import confident_classifier

import tensorflow as tf

BATCH_SIZE = 128
MNIST_IMAGE_DIM = (28, 28)
NOISE_DIM = 100
MNIST_NUM_CLASSES = 10
TRAIN_STEPS = 1


def mnist_dummy_input_fn():
    image = tf.random_uniform([BATCH_SIZE, 784])
    noise = tf.random_uniform([BATCH_SIZE, 100])
    labels = tf.random_uniform([BATCH_SIZE, 1], maxval=9, dtype=tf.int32)
    return {'image': image, 'noise': noise}, labels


def make_mnist_estimator():
    image_feature = tf.feature_column.numeric_column('image', shape=MNIST_IMAGE_DIM)
    noise_feature = tf.feature_column.numeric_column('noise', shape=NOISE_DIM)
    return tf.estimator.Estimator(
        model_fn=confident_classifier,
        params={
            'image': [image_feature],
            'noise': [noise_feature],
            'image_dim': MNIST_IMAGE_DIM, # Todo: Surplus param
            'noise_dim': NOISE_DIM, # Todo: Surplus param
            'num_classes': MNIST_NUM_CLASSES,
            'discriminator': None,
            'generator': None,
            'classifier': None,
            'learning_rate': 0.001,
            'beta': 1.0,
        })

class JointConfidentModelTest(tf.test.TestCase):

    def test_confident_classifier(self):
        classifier = make_mnist_estimator()
        classifier.train(input_fn=mnist_dummy_input_fn, steps=2)
        eval_results = classifier.evaluate(input_fn=mnist_dummy_input_fn, steps=1)

        # Couldn't find the references of these two:
        loss = eval_results['loss']
        global_step = eval_results['global_step']

        accuracy = eval_results['accuracy']
        
        self.assertEqual(loss.shape, ())
        self.assertEqual(2, global_step) # ?
        self.assertEqual(accuracy.shape, ())


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.test.main()
