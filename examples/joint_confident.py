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
import numpy as np
from safeai.models.joint_confident import confident_classifier
import tensorflow as tf
from tensorflow.keras.applications import vgg16

tf.logging.set_verbosity(tf.logging.DEBUG)
cifar10 = tf.keras.datasets.cifar10

def make_generator(images, noises, labels):
    def gen():
        for (image, noise), label in zip(zip(images, noises), labels):
            yield {'image': image, 'noise': noise}, label
    return gen


def train_input_fn(images, labels, noise_size, batch_size):
    """
    docstring
    """

    images = images / 255.
    noises = np.random.normal(0, 1, (images.shape[0], noise_size))
    labels = labels.astype(np.int32)

    gen = make_generator(images, noises, labels)

    output_tensor_types = (
        ({'image': tf.float32, 'noise': tf.float32}, tf.int32))
    output_tensor_shapes = (
        ({'image': tf.TensorShape(images.shape[1:]),
          'noise': tf.TensorShape(noises.shape[1:])},
         tf.TensorShape([1])))
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=output_tensor_types,
        output_shapes=output_tensor_shapes)

    dataset = dataset.cache().shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, noise_size, batch_size):
    """
    Used in both evaluation, prediction
    """
    noise = np.random.random((features.shape[0], noise_size))
    labels = labels.astype(np.int32)
    features_dict = {'image': features, 'noise': noise}
    if labels is None:
        inputs = features_dict
    else:
        inputs = (features_dict, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
    return dataset


def main():
    """
    docstring
    """
    batch_size = 256
    train_steps = 30000
    noise_dim = 100
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Todo: NWHC or NCWH? Tue 06 Nov 2018 09:36:29 PM KST
    image_shape = x_train.shape[1:]

    image_feature = tf.feature_column.numeric_column(
        'image', shape=image_shape)
    noise_feature = tf.feature_column.numeric_column(
        'noise', shape=noise_dim)

    # Todo: Reduce params['dim'] Tue 06 Nov 2018 05:05:10 PM KST
    joint_confident_classifier = tf.estimator.Estimator(
        model_fn=confident_classifier,
        params={
            'image': image_feature,
            'noise': noise_feature,
            'classes': num_classes,
            #'classifier': (vgg16.VGG16, {'include_top': False, 'weights': None, 'input_shape': image_feature.shape}),
            'learning_rate': 0.001,
            'beta': 1.0,
        })

    joint_confident_classifier.train(
        input_fn=lambda: train_input_fn(x_train,
                                        y_train,
                                        noise_dim,
                                        batch_size),
        steps=train_steps)

    eval_result = joint_confident_classifier.evaluate(
        input_fn=lambda: eval_input_fn(x_test, y_test, noise_dim, batch_size),
        steps=train_steps)
    prediction = joint_confident_classifier.predict(
        input_fn=lambda: train_input_fn(x_test, y_test, noise_dim, batch_size)
    )

    tf.logging.info('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))
    tf.logging.info('Prediction: {}'.format(prediction))

if __name__ == "__main__":
    main()
