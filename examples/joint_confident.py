import numpy as np
import tensorflow as tf
from safeai.models.joint_confident import confident_classifier
tf.logging.set_verbosity(tf.logging.INFO)
mnist = tf.keras.datasets.mnist


def dcgan():
    """
    docstring
    """
    pass


def vgg():
    """
    docstring
    """
    pass


def train_input_fn(features, labels, noise_size, batch_size):
    """
    docstring
    """
    noise = np.random.random((features.shape[0], noise_size))
    dataset = tf.data.Dataset.from_tensor_slices(
            ({'image': features, 'noise': noise}, labels)
    )
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    """
    docstring
    """
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def main():
    """
    docstring
    """
    batch_size = 128
    train_steps = 1000
    noise_dim = 100
    output_dim = 10
    image_dim = (28, 28)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255., x_test/255.
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
    image_width, image_height = x_train.shape[1:]

    image_feature = tf.feature_column.numeric_column('image', shape=[image_width, image_height])
    noise_feature = tf.feature_column.numeric_column('noise', shape=[noise_dim])

    # Todo: Reduce params['dim'] Tue 06 Nov 2018 05:05:10 PM KST
    joint_confident_classifier = tf.estimator.Estimator(
        model_fn=confident_classifier,
        params={
            'image': [image_feature],
            'noise': [noise_feature],
            'image_dim': image_dim,
            'noise_dim': noise_dim,
            'output_dim': output_dim,
            'discriminator': None,
            'generator': None,
            'classifier': None,
            'beta': 1,
        })

    joint_confident_classifier.train(
        input_fn=lambda: train_input_fn(x_train, y_train, noise_dim, batch_size),
        steps=train_steps
    )

    eval_result = joint_confident_classifier.evaluate(
        input_fn=lambda: eval_input_fn(x_test, y_test, batch_size),
        steps=train_steps
    )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == "__main__":
    main()
