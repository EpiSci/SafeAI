from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def gradual_sequence(start, end, multiplier=3):
    """Custom nodes number generator
    The function gives the list of exponentially increase/decrease
    integers from both 'start' and 'end' params, which can be later used as
    the number of nodes in each layer.
    _gradual_sequence(10, 7000, multiplier=5) gives [50, 250, 1250, 6250],
    _gradual_sequence(6000, 10, multiplier=5) gives [1250, 250, 50]
    as a return sequence.
        Args:
            start: lower limit(exclusive)
            end: upper limit
        Returns:
            num_nodes_list: list of integers
            or:
            reversed(num_nodes_list)
    """
    mode = 'incremental'
    if end < start:
        mode = 'decremental'
        start, end = end, start
    num_nodes_list = [start*(multiplier**x) for x in range(10)
                      if start*(multiplier**x) < end]
    if mode == 'incremental':
        return num_nodes_list
    return reversed(num_nodes_list)
