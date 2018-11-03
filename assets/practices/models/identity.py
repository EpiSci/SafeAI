# ==============================================================================
"""Contains Identity model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import identity

def identity_example(inputs,
		    name=None):
    net = identity(inputs, name)
    return net

if __name__ == '__main__':
    test = [1,2,3]
    result = identity_example(test, name='test_yes')
    print(result.name)
