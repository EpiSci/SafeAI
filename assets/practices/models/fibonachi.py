from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def fib1(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print('   ')

def fib2(n):
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result

if __name__ == '__main__':
    
    m = 20
    
    c = fib1(m)
    d = fib2(m)
    
    print(c)
    print(d)
