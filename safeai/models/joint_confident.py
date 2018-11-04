from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn

def _check_gan():
    pass


def _check_classifier():
    pass


def dcgan_fn():
    pass


def vgg_fn():
    pass


def _build_default_classifier():
    scope_name = 'Classifier'


def _build_default_generator():
    scope_name = 'Generator'


def _build_default_discriminator():
    scope_name = 'Discriminator'
    pass


def _build_default_model_fn(model_string):
    submodel_builders = {
        'classifier': _build_default_classifier,
        'generator': _build_default_generator,
        'discriminator': _build_default_discriminator
    }
    return submodel_builders[model_string]()

def confident_classifier(features, labels, mode, params):

    submodels = {}
    for model in ['discriminator', 'generator', 'classifier']:
        if not model or model not in params:
            print('default model fetching: {} model'.format(model))
            submodels[model] = _build_default_model_fn(model)
        else:
            submodels[model] = params[model]

    if mode not in [model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))

    # Combine three independent submodels
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    logits = tf.layers.dense(net, params['num_classes'], activation=None)
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
