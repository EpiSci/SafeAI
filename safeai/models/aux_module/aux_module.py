# Copyright (c) 2018 Episys Science, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Custom layer for merging 3 layers into.
class Merge3Ways(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Merge3Ways, self).__init__(**kwargs)

    def build(self, input_shape):

        # Destructuring shapes
        shape_x, shape_h2, shape_logits, shape_rec =\
                list(map(lambda shape: int(shape[1]), input_shape))

        self.weight1 = self.add_weight(name='h2_to_merged',
                                       shape=(shape_h2, self.output_dim))
        self.weight2 = self.add_weight(name='logits_to_merged',
                                       shape=(shape_logits, self.output_dim))
        self.weight3 = self.add_weight(name='rec_to_merged',
                                       shape=(shape_rec, self.output_dim))
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,))
        super(Merge3Ways, self).build(input_shape)

    def call(self, inputs):

        x, h2, logits_out, reconstruction = inputs
        a1 = K.dot(h2, self.weight1)
        a2 = K.dot(logits_out, self.weight2)
        a3 = K.dot(K.square(reconstruction-x), self.weight3)
        return (a1 + a2 + a3) + self.bias

def aux_base_model():
    pass
