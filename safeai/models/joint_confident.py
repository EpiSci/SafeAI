from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _check_gan():
    pass


def _check_classifier():
    pass


def confident_classifier(features, labels, mode, params):

    discriminator_fn = params['discriminator']
    generator_fn = params['generator']
    classifier_fn = params['classifier']

    if mode not in [model_fn_lib.ModeKeys.TRAIN, model_fn_lib.ModeKeys.EVAL,
                    model_fn_lib.ModeKeys.PREDICT]:
        raise ValueError('Mode not recognized: %s' % mode)
