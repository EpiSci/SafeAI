from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.estimator import model_fn
from tensorflow.train import get_global_step
from safeai.utils.distribution import kl_divergence_with_uniform


def _gradual_sequence(start, end, multiplier=3):
    """Custom nodes number generator
    The function gives the list of exponentially increase/decrease
    integers from both 'start' and 'end' params, which can be later used as
    the number of nodes in each layer.
    _gradual_sequence(10, 7000, multiplier=5) gives [50, 250, 1250, 6250],
    _gradual_sequence(6000, 10, multiplier=5) gives [1250, 250, 50]
    as a return sequence.
        Args:
            start: lower limit(exclusive)
            end: upper limit
        Returns:
            num_nodes_list: list of integers
            or:
            reversed(num_nodes_list)
    """
    mode = 'incremental'
    if end < start:
        mode = 'decremental'
        start, end = end, start
    num_nodes_list = [start*(multiplier**x) for x in range(10)
                      if start*(multiplier**x) < end]
    if mode == 'incremental':
        return num_nodes_list
    else:
        return reversed(num_nodes_list)


def _default_classifier(input_feature_column, num_classes, reuse=True):
    # Todo: Capture custom layer units ([500, 200, 10])
    # at params['classifier_units']
    """Default classifier network.
        Args:
            input_feature_column: (None, latent_space_size), tf.float32 tensor
            num_classes: tf.int32 tensor-like object
            reuse: variable reuse for scoping
        Returns:
            net: the generator output node
    """
    image_dim = int(np.prod(input_feature_column.shape[1:]))
    units_in_layers = _gradual_sequence(image_dim, num_classes)
    with tf.variable_scope("classifier", reuse=reuse):
        net = input_feature_column
        for hidden_units in units_in_layers:
            net = tf.layers.dense(net,
                                  units=hidden_units,
                                  activation=tf.nn.relu)
        logits = tf.layers.dense(net, units=num_classes, activation=None)
        return logits


def _default_generator(noise_input, image_dim, reuse=True):
    """Default generator network.
        Args:
            noise_input: (None, latent_space_size), tf.float32 tensor
            output_dim
            reuse: variable reuse for scoping
        Returns:
            net: the generator output node
    """
    if type(image_dim) is list or type(image_dim) is tuple:
        image_dim = np.prod(image_dim)
    noise_dim = noise_input.shape[1]
    units_in_layers = _gradual_sequence(noise_dim, image_dim)
    with tf.variable_scope("generator", reuse=reuse):
        for hidden_units in units_in_layers:
            net = tf.layers.dense(noise_input,
                                  units=hidden_units,
                                  activation=tf.nn.relu)
        net = tf.layers.dense(net, units=image_dim, activation=None)
        return net


def _default_discriminator(input_feature_column, reuse=True):
    """Default discriminator network.
        Args:
            input_feature_column: (None, input_image_dim), tf.float32 tensor
            reuse: variable reuse for scoping
        Returns:
            net: the generator output node
    """
    image_dim = np.prod(input_feature_column.shape[1:])
    units_in_layers = _gradual_sequence(image_dim, 1)
    with tf.variable_scope('discriminator', reuse=reuse):
        net = input_feature_column
        for hidden_units in units_in_layers:
            net = tf.layers.dense(net,
                                  units=hidden_units,
                                  activation=tf.nn.relu)
        score = tf.layers.dense(net, units=1, activation=None)
        return score


def _default_submodel_fn(submodel_id):
    submodel_builders = {
        'classifier': _default_classifier,
        'generator': _default_generator,
        'discriminator': _default_discriminator
    }
    return submodel_builders[submodel_id]


"""
Expected params: {
    'image': [image_feature],
    'noise': [noise_feature],
    'image_dim': image_dim,
    'noise_dim': noise_dim,
    'num_classes': num_classes,
    'discriminator': None,
    'generator': None,
    'classifier': None,
    'learning_rate': 0.001,
    'beta': 0.0,
}
"""


def confident_classifier(features, labels, mode, params):

    if mode not in [model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))

    submodels = {}
    for submodel_id in ['discriminator', 'generator', 'classifier']:
        if not params[submodel_id] or submodel_id not in params:
            print('default model fetching: {} model'.format(submodel_id))
            submodels[submodel_id] = _default_submodel_fn(submodel_id)
        else:
            submodels[submodel_id] = params[submodel_id]

    # Todo: sanity check on each submodel Mon Nov  5 14:16:29 2018
    # Combine three independent submodels
    classifier_fn = submodels['classifier']
    discriminator_fn = submodels['discriminator']
    generator_fn = submodels['generator']

    image_input_layer = tf.feature_column.input_layer(features,
                                                      params['image'])
    logits = classifier_fn(image_input_layer, params['num_classes'],
                           reuse=False)
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

    labels = tf.cast(labels, tf.int32)
    noise = tf.feature_column.input_layer(features, params['noise'])

    # Todo: Take loss from params Mon Nov  5 20:14:35 2018
    # Todo: reuse reversed bool Tue 06 Nov 2018 05:08:16 PM KST
    # Discriminator loss
    # Real image
    d_score_real = discriminator_fn(image_input_layer, reuse=False)

    # Fake image
    generated_fake_image = generator_fn(noise, params['image_dim'],
                                        reuse=False)
    d_score_fake = discriminator_fn(generated_fake_image)

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

    logits_fake = classifier_fn(generated_fake_image, params['num_classes'])
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
    # valid
    classifier_variables = tf.trainable_variables(scope="classifier")
    discriminator_variables = tf.trainable_variables(scope="discriminator")
    generator_variables = tf.trainable_variables(scope="generator")

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
