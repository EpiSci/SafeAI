from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath('../safeai')))

import tensorflow as tf
from safeai.models.identity import identity_example as identity

def main():
    test = [1,2,3,4,5]
    halloween = identity(test, name='halloween')
    with tf.Session() as sess:
        print(sess.run(halloween))
        print(halloween.name)

if __name__ == '__main__':
    main()

