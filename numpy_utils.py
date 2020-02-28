import numpy as np


def fast_concatenate_0(a_tuple):
    if len(a_tuple) == 1:
        return a_tuple[0]
    else:
        return np.concatenate(a_tuple, axis=0)
