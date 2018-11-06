import tensorflow as tf
from safeai.models import joint_confident
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


def train_input_fn(features, labels, batch_size):
    """
    docstring
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_width, image_height = x_train.shape[1:]
    image_feature = tf.feature_column.numeric_column('image', shape=[image_width, image_height])

    joint_confident_classifier = tf.estimator.Estimator(
        model_fn=joint_confident,
        params={
            'feature_columns': [image_feature],
            'num_classes': 10,
            'discriminator_fn': None,
            'generator_fn': None,
            'classifier_fn': None,
        }
    )
    joint_confident_classifier.train(
        input_fn=lambda: train_input_fn(x_train, y_train, batch_size),
        steps=train_steps
    )

    eval_result = joint_confident_classifier.evaluate(
        input_fn=lambda: eval_input_fn(x_test, y_test, batch_size),
        steps=train_steps
    )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == "__main__":
    main()
