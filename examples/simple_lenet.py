
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import tensorflow as tf

from datasets import mnist
from safeai.models.simple_vgg import lenet

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist',
                    'Directory with the mnist data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_batches', None,
                     'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def preprocess_image(image, output_height, output_width, is_training):
  """Preprocesses the given image.
  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
  Returns:
    A preprocessed image.
  """
  image = tf.to_float(image)
  image = tf.image.resize_image_with_crop_or_pad(
      image, output_width, output_height)
  image = tf.subtract(image, 128.0)
  image = tf.div(image, 128.0)
  return image

def load_batch(dataset, batch_size=32, height=28, width=28, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image, label = data_provider.get(['image', 'label'])

    image = preprocess_image(
        image,
        height,
        width,
        is_training)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, labels

def main(args):
    dataset = mnist.get_split('train', FLAGS.data_dir)

    # load batch of dataset
    images, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)

    # run the model
    predictions = lenet(images)

    # cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)
    total_loss = slim.losses.get_total_loss()

    optimizer = tf.train.AdamOptimizer()

    train_op = slim.learning.create_train_op(
        total_loss,
        optimizer,
        summarize_gradients=True)

    slim.learning.train(
        train_op,
        FLAGS.log_dir,
        save_summaries_secs=20)

if __name__ == '__main__':
    tf.app.run()
