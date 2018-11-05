from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn


def _default_classifier_fn(features, params):
    # Todo: Capture custom layer units ([500, 200, 10] ) at params['classifier_units']
    scope_name = 'Classifier'

    image_dim = np.prod(features['image'].shape[1:])
    output_dim = params['num_classes']
    units_in_layers = reversed([output_dim*(5**x) for x in range(10) if output_dim*(5**x) < image_dim])
    with tf.variable_scope(scope_name) as scope:
        net = tf.feature_column.input_layer(features, params['image_column'])
        for hidden_units in units_in_layers:
            net = tf.layers.dense(net, units=hidden_units, activation=tf.nn.relu)
        logits = tf.layers.dense(net, params['num_classes'], activation=None)
        return logits


def _default_generator(features, params):
    # Todo: fixing Mon Nov  5 10:57:10 2018
    scope_name = 'Generator'


def _default_discriminator_fn(features, params):
    scope_name = 'Discriminator'
    pass

def _build_default_model_fn(model_string, features, params):
    submodel_builders = {
        'classifier': _default_classifier_fn,
        'generator': _default_generator_fn,
        'discriminator': _default_discriminator_fn
    }
    return submodel_builders[model_string](features, params)


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
    submodels = {}
    for submodel_id in ['discriminator', 'generator', 'classifier']:
        if not submodel_id or submodel_id not in params:
            print('default model fetching: {} model'.format(submodel_id))
            submodels[model] = _build_default_model_fn(submodel_id, features, params)
        else:
            submodels[model] = params[model]

    if mode not in [model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))

    # Combine three independent submodels
    classifier = submodels['classifier']
    discriminator = submodels['discriminator']
    generator = submodels['generator']

    logits =
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
