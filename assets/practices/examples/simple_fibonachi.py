from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath('../safeai')))

from safeai.model.fibonachi import fib1
from safeai.model.fibonachi import fib2

import tensorflow as tf

def main():
    n=100
    value1 = fib1(n)
    value2 = fib2(n)
    
    print(value1)
    print(value2)

if __name__ == '__main__':
    main()
