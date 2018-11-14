from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.estimator import model_fn
from tensorflow.train import get_global_step
from safeai.utils import gradual_sequence
from safeai.utils.distribution import kl_divergence_with_uniform

def _default_classifier(input_tensor, classes):
    """Create default classifier.
        Args:
            input_tensor: (N, W, H, C) or (N, C, W, H) tf.float32 tensor
            classes: number of classes
        Returns:
            model: tf.keras.Model instance
    """
    image_shape = input_tensor.shape[1:] # (N, 32, 32, 3) => (32, 32, 3)
    image_dim_flatten = np.prod(image_shape) # (32, 32, 3) => 3072
    # gradual_sequence() returns a list of integers that gradually
    # decreases/increases. For example, gradual_sequence(784, 10, multiplier=3)
    # returns a list [270, 90, 30]
    units_in_layers = gradual_sequence(image_dim_flatten, classes)

    # Define default model
    inputs = tf.keras.layers.Input(shape=image_shape, name='input')
    net = tf.keras.layers.Dense(image_dim_flatten, activation='relu')(inputs)
    for hidden_units in units_in_layers:
        net = tf.keras.layers.Dense(hidden_units,
                                    activation='relu')(net)
    logits = tf.keras.layers.Dense(classes, activation=None)(net)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


def _default_generator(input_tensor, image_tensor):
    """Default generator network.
        Args:
            noise_input: (N, latent_space_size), tf.float32 tensor
            image_tensor: img tensor to get shape of fake img, tf.float32 tensor
        Returns:
            model: tf.keras.Model instance
    """
    image_dim_flatten = np.prod(image_tensor.shape[1:])
    assert len(input_tensor.shape) == 2  # (N, latent_space_size)
    noise_dim = input_tensor.shape[1]
    units_in_layers = gradual_sequence(noise_dim, image_dim_flatten)

    inputs = tf.keras.layers.Input(shape=(noise_dim,))
    net = tf.keras.layers.Dense(noise_dim, activation='relu')(inputs)
    for hidden_units in units_in_layers:
        net = tf.keras.layers.Dense(hidden_units, activation='relu')(net)
    image_flatten = tf.keras.layers.Dense(image_dim_flatten,
                                          activation=None)(net)
    model = tf.keras.Model(inputs=inputs, outputs=image_flatten)
    return model


def _default_discriminator(input_tensor):
    """Default discriminator network.
        Args:
            input_tensor: (None, input_image_dim), tf.float32 tensor
        Returns:
            model: tf.keras.Model instance
    """
    image_shape = input_tensor.shape[1:]
    image_dim_flatten = np.prod(image_shape)
    units_in_layers = gradual_sequence(image_dim_flatten, 1)

    inputs = tf.keras.layers.Input(shape=(image_shape))
    net = tf.keras.layers.Dense(image_dim_flatten, activation='relu')(inputs)
    for hidden_units in units_in_layers:
        net = tf.keras.layers.Dense(hidden_units, activation='relu')(net)
    logit = tf.keras.layers.Dense(1, activation=None)(net)
    model = tf.keras.Model(inputs=inputs, outputs=logit)
    return model


def _default_submodel_fn(submodel_id):
    submodels = {
        'classifier': _default_classifier,
        'generator': _default_generator,
        'discriminator': _default_discriminator
    }
    submodel_fn = submodels[submodel_id]
    return submodel_fn  # Returns one of functions above


"""
Expected params: {
    'image': [image_feature],
    'noise': [noise_feature],
    'classes': num_classes
    'discriminator': None,  <-
    'generator': None,      <- instantiated keras model, or chain of callable layers
    'classifier': None,     <-
    'learning_rate': 0.001,
    'beta': 1.0,
}
"""
def confident_classifier(features, labels, mode, params):

    image_feature = params['image']
    noise_feature = params['noise']

    image_input_layer = tf.feature_column.input_layer(features,
                                                      image_feature)
    noise_input_layer = tf.feature_column.input_layer(features,
                                                      noise_feature)

    default_model_args = {
        'classifier': [image_input_layer, params['classes']],
        'generator': [noise_input_layer, image_input_layer],
        'discriminator': [image_input_layer]
    }

    submodels = {}
    for submodel_id in ['classifier', 'generator', 'discriminator']:
        if not params[submodel_id] or submodel_id not in params:
            tf.logging.info("Fetching default {}".format(submodel_id))
            args = default_model_args[submodel_id]
            submodel_fn = _default_submodel_fn(submodel_id)
            submodels[submodel_id] = submodel_fn(*args)
        else:
            if not callable(params[submodel_id]):
                raise ValueError("Model should be callable.")
            submodels[submodel_id] = params[submodel_id]


    if mode not in [model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))

    # Combine three independent submodels
    # Todo: callable, shape check on each submodel Mon Nov  5 14:16:29 2018
    classifier = submodels['classifier']
    discriminator = submodels['discriminator']
    generator = submodels['generator']

    logits = classifier(image_input_layer)
    predicted_classes = tf.argmax(logits, axis=1)
    confident_score = tf.nn.softmax(logits)

    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_classes,
            'probabilities': confident_score,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)



    # Todo: Take loss from params Mon Nov  5 20:14:35 2018
    # Discriminator loss

    # Put real image to discriminator
    d_score_real = discriminator(image_input_layer)

    # Put fake image to discriminator
    generated_fake_image = generator(noise_input_layer)

    d_score_fake = discriminator(generated_fake_image)


    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_real,
            labels=tf.ones_like(d_score_real))
    )

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_fake,
            labels=tf.zeros_like(d_score_fake))
    )

    # Discriminator loss
    discriminator_loss = d_loss_real + d_loss_fake


    g_loss_from_d = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_score_fake,
            labels=tf.ones_like(d_score_fake))
    )

    logits_fake = classifier(generated_fake_image)

    confident_score_fake = tf.nn.softmax(logits_fake)
    classifier_uniform_kld_fake =\
        kl_divergence_with_uniform(confident_score_fake)

    # Generator loss
    generator_loss = g_loss_from_d +\
        (params['beta'] * classifier_uniform_kld_fake)

    # Classifier loss
    nll_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)
    classifier_loss = nll_loss + (params['beta'] * classifier_uniform_kld_fake)

    # Separate variables to applying gradient only to subgraph
    classifier_variables = classifier.trainable_variables
    discriminator_variables = discriminator.trainable_variables
    generator_variables = generator.trainable_variables

    # Define three training operations
    lr = params['learning_rate']
    optimizer_discriminator = tf.train.AdamOptimizer(lr)
    train_discriminator_op =\
        optimizer_discriminator.minimize(discriminator_loss,
                                         global_step=get_global_step(),
                                         var_list=discriminator_variables)

    optimizer_generator = tf.train.AdamOptimizer(lr)
    train_generator_op =\
        optimizer_generator.minimize(generator_loss,
                                     global_step=get_global_step(),
                                     var_list=generator_variables)

    optimizer_classifier = tf.train.AdamOptimizer(lr)
    train_classifier_op =\
        optimizer_classifier.minimize(classifier_loss,
                                      global_step=get_global_step(),
                                      var_list=classifier_variables)

    grouped_ops = tf.group([train_discriminator_op,
                            train_generator_op,
                            train_classifier_op])

    # Define accuracy, metrics
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes)

    metrics = {'accuracy': accuracy}

    # Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=nll_loss, eval_metric_ops=metrics)

    # Train: Alternative learning
    if mode == tf.estimator.ModeKeys.TRAIN:
        est_spec = tf.estimator.EstimatorSpec(mode,
                                              loss=nll_loss,
                                              train_op=grouped_ops)
        tf.summary.scalar('accuracy', accuracy[1])
        return est_spec

    raise ValueError("Invalid estimator mode: reached the end of the function")
