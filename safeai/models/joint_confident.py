from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn


def _gradual_nodes(start, end, multiplier=5):
    """Custom nodes number generator
    The function gives the list of exponentially increase/decrease
    integers from both 'start' and 'end' params, which can be used later as
    the number of nodes in each layer.
    _gradual_nodes(10, 7000) gives [50, 250, 1250, 6250],
    _gradual_nodes(6000, 10) gives [1250, 250, 50] as a return object.
        Args:
            start: lower limit
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


def _default_classifier(input_feature_column, output_dim, reuse=False):
    # Todo: Capture custom layer units ([500, 200, 10])
    # at params['classifier_units']
    """Default classifier network.
        Args:
            input_feature_column: (None, latent_space_size), tf.float32 tensor
            output_dim: tf.int32 tensor-like object
            reuse: variable reuse for scoping
        Returns:
            net: the generator output node
    """
    image_dim = np.prod(input_feature_column.shape[1:])
    units_in_layers = _gradual_nodes(image_dim, output_dim)
    with tf.variable_scope("classifier", reuse=reuse):
        net = input_feature_column
        for hidden_units in units_in_layers:
            net = tf.layers.dense(net,
                                  units=hidden_units,
                                  activation=tf.nn.relu)
        logits = tf.layers.dense(net, units=output_dim, activation=None)
        return logits


def _default_generator(noise_input, output_dim, reuse=False):
    """Default generator network.
        Args:
            noise_input: (None, latent_space_size), tf.float32 tensor
            output_dim
            reuse: variable reuse for scoping
        Returns:
            net: the generator output node
    """
    noise_dim = noise_input.shape[1]
    units_in_layers = _gradual_nodes(noise_dim, output_dim)
    with tf.variable_scope("generator", reuse=reuse):
        for hidden_units in units_in_layers:
            net = tf.layers.dense(noise_input,
                                  units=hidden_units,
                                  activation=tf.nn.relu)
        net = tf.layers.dense(net, units=output_dim, activation=None)
        return net


def _default_discriminator(input_feature_column, reuse=False):
    """Default discriminator network.
        Args:
            input_feature_column: (None, input_image_dim), tf.float32 tensor
            reuse: variable reuse for scoping
        Returns:
            net: the generator output node
    """
    image_dim = np.prod(input_feature_column.shape[1:])
    units_in_layers = _gradual_nodes(image_dim, 1)
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
    'image_column',
    'noise_column',
    'image_dim',
    'noise_dim',
    'num_classes',
    'classifier',
    'discriminator',
    'generator',
}
"""
def confident_classifier(features, labels, mode, params):
    
    if mode not in [model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))
    
    submodels = {}
    for submodel_id in ['discriminator', 'generator', 'classifier']:
        if not submodel_id or submodel_id not in params:
            print('default model fetching: {} model'.format(submodel_id))
            submodels[submodel_id] = _default_submodel_fn(submodel_id)
        else:
            submodels[submodel_id] = params[submodel_id]

    # Todo: sanity check on each submodel Mon Nov  5 14:16:29 2018
    # Combine three independent submodels
    classifier_fn = submodels['classifier']
    discriminator_fn = submodels['discriminator']
    generator_fn = submodels['generator']

    # Separate varialbes to applying gradient only to subgraph
    classifier_variables = tf.trainable_variables(scope="classifier")
    discriminator_variables = tf.trainable_variables(scope="discriminator")
    generator_variables = tf.trainable_variables(scope="generator")

    # Training Discriminator(1)
    net = tf.feature_column.input_layer(features, params['image_column'])
    noise = tf.feature_column.input_layer(features, params['noise'])
    score_real = discriminator_fn(input_image)
    generated_image = generator_fn(noise)
    score_fake = discriminator_fn(generated_image)


    # Training Generator(2)







    train_discriminator = tf.train.AdamOptimizer(1e-4).minimize(D_loss, var_list=D_vars)
    train_generator = tf.train.AdamOptimizer(1e-4).minimize(G_loss, var_list=G_vars)


    predicted_classes = tf.argmax(logits, 1)
    confident_score = tf.nn.softmax(logits)

    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': confident_score,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Define loss, metrics for EstimatorSpec
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.003)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    raise ValueError("Invalid estimator mode: reached the end of the function")
