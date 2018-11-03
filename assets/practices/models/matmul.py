from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import matmul
import tensorflow as tf

def matmul_example(input1, input2,
                   name=None):
    net = matmul(input1, input2, transpose_b=True, name=name)
    return net

if __name__ == '__main__':
    test1 = tf.ones((2,2))
    test2 = tf.ones((2,2))
    result = matmul_example(test1, test2, name='test_yes')
    print(result.name)