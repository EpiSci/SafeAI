from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import array_ops, init_ops, nn_ops, variable_scope


def smodel_arg_scope(weight_decay=0.0005):
    with arg_scope(
        [layers.conv2d, layers_lib.fully_connected],
        activation_fn=nn_ops.relu,
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        biases_initializer=init_ops.zeros_initializer()
    ):
        with arg_scope([layers.conv2d], padding='SAME') as arg_sc:
            return arg_sc

