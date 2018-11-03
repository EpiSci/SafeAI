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
    pass


def _build_default_generator():
    scope_name = 'generator_scope'


def _build_default_discriminator():
    pass


def _build_default_model_fn(model_string):
    pass

def confident_classifier(features, labels, mode, params):

    models_dict = {}
    for model in ['discriminator', 'generator', 'classifier']:
        if not model or model not in params:
            print('default model fetching: {} model'.format(model))
            models_dict[model] = _build_default_model_fn(model)
        else:
            models_dict[model] = params[model]

    if mode not in [model_fn.ModeKeys.TRAIN, model_fn.ModeKeys.EVAL,
                    model_fn.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: {}'.format(mode))
    
    net = tf.feature_column.input_layer(features, params['feature_columns'])

