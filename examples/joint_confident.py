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

import argparse

import numpy as np
from safeai.models.joint_confident import confident_classifier
import safeai.metrics.confusion as confusion
from safeai.datasets import cifar10, svhn, stl10, mnist, tinyimagenet200

import tensorflow as tf
tf.logging.set_verbosity('INFO')


def make_generator(images, noises, labels):
    def gen():
        for (image, noise), label in zip(zip(images, noises), labels):
            yield {'image': image, 'noise': noise}, label
    return gen


def train_input_fn(images, labels, noise_size, batch_size):

    image_shape = images.shape[1:]
    noises = np.random.normal(0, 1, (images.shape[0], noise_size))
    labels = labels.astype(np.int32).squeeze()
    gen = make_generator(images, noises, labels)

    output_types = (({'image': tf.float32, 'noise': tf.float32},
                     tf.int32))
    output_shapes = (
        ({'image': tf.TensorShape(image_shape), 
          'noise': tf.TensorShape(noise_size)},
         tf.TensorShape([]))) # 1 or 0?

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=output_types,
        output_shapes=output_shapes)

    dataset = dataset.cache().shuffle(3000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, noise_size, batch_size=128):
    noise = np.random.random((features.shape[0], noise_size))
    features_dict = {'image': features, 'noise': noise}
    if labels is None:
        inputs = features_dict
    else:
        labels = labels.astype(np.int32)
        inputs = (features_dict, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size)
    return dataset


def convert_pred_generator_to_array(gen, num_features, num_classes):

    classes = np.empty(num_features, dtype=np.uint8)
    probabilities = np.empty((num_features, num_classes), dtype=np.float32)

    for i, pred in enumerate(gen):
        assert isinstance(pred, dict)
        classes[i] = pred['classes']
        probabilities[i, :] = np.array(pred['probabilities'])

    return classes, probabilities


def in_out_predictions(classifier, features_in, features_out, labels_in=None, noise_dim=100):

    gen_predictions_in = classifier.predict(
        input_fn=lambda: eval_input_fn(features_in, labels_in, noise_dim, batch_size=128))
    gen_predictions_out = classifier.predict(
        input_fn=lambda: eval_input_fn(features_out, None, noise_dim, batch_size=128))

    _, in_probabilties = convert_pred_generator_to_array(gen_predictions_in,
                                                         len(features_in),
                                                         classifier.params['classes'])
    _, out_probabilties = convert_pred_generator_to_array(gen_predictions_out,
                                                          len(features_out),
                                                          classifier.params['classes'])
    return in_probabilties, out_probabilties


def main(args):
    batch_size = 256
    amount_of_ten_epochs = 10
    noise_dim = 100
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    _, (x_test_out, y_test_out) = svhn.load_data()

    # Normalize image data, 0~255 to -1~1
    x_train = x_train.astype(np.float32)
    x_train = x_train/255.
    x_test = x_test.astype(np.float32)
    x_test = x_test/255.

    x_test_out = x_test_out.astype(np.float32)
    x_test_out = x_test_out/255.


    # Todo: NWHC or NCWH? Tue 06 Nov 2018 09:36:29 PM KST
    image_shape = list(x_train.shape[1:])

    # greyscale data should be expanded
    if len(image_shape) == 2:
        image_shape += [1]
        x_train = np.reshape(x_train, [-1] + image_shape)
        x_test = np.reshape(x_test, [-1] + image_shape)

    image_feature = tf.feature_column.numeric_column(
        'image', shape=image_shape)
    noise_feature = tf.feature_column.numeric_column(
        'noise', shape=noise_dim)

    # Todo: Reduce params['dim'] Tue 06 Nov 2018 05:05:10 PM KST
    classifier = tf.estimator.Estimator(
        model_fn=confident_classifier,
        model_dir=args.model_dir,
        params={
            'image': image_feature,
            'noise': noise_feature,
            'classes': num_classes,
            'learning_rate': 0.00009,
            'alpha': 1.0,
            'beta': 0.0,
            'train_classifier_only': args.train_classifier_only
        })

    if args.mode == 'train':
        for ten_epochs in range(amount_of_ten_epochs):
            classifier.train(
                input_fn=lambda: train_input_fn(x_train, y_train, noise_dim, batch_size),
                steps=len(x_train)//batch_size*10)

            eval_result = classifier.evaluate(
                input_fn=lambda: eval_input_fn(x_test, y_test, noise_dim, batch_size),
                steps=len(x_test)//batch_size)

            tf.logging.info('Test set accuracy: {accuracy:0.3f}'.format(**eval_result))

    elif args.mode == 'test':
        _, probs = in_out_predictions(classifier,
                                      x_test,
                                      x_test,
                                      labels_in=y_test)

        stats = confusion.get_inout_stats(probs, probs, labels_in=y_test)
        print('accuracy: {}'.format(stats['accuracy']))

    elif args.mode == 'inout':
        in_probs, out_probs = in_out_predictions(classifier,
                                                 x_test,
                                                 x_test_out,
                                                 labels_in=y_test)

        stats = confusion.get_inout_stats(in_probs, out_probs, labels_in=y_test)
        print(stats)


def get_args():
    default_model_path = '/tmp/joint_confident'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_dir', type=str, default=default_model_path)
    parser.add_argument('--train_classifier_only', type=str, default=False)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    if args.mode not in ['train', 'test', 'inout']:
        raise ValueError("'mode' must be one of 'train', 'test', 'inout'")
    main(args)
