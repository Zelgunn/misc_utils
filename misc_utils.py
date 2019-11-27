import tensorflow as tf
import numpy as np
from typing import List


def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def to_constant_list(inputs, name: str) -> List[tf.Tensor]:
    inputs = to_list(inputs)
    if len(inputs) == 1:
        return [tf.constant(inputs[0], name=name)]
    else:
        return [tf.constant(x, name="{}_{}".format(name, i)) for i, x in enumerate(to_list(inputs))]


def int_ceil(value, epsilon=1e-5) -> int:
    # noinspection PyUnresolvedReferences
    return int(np.ceil(value - epsilon))


def int_floor(value, epsilon=1e-5) -> int:
    # noinspection PyUnresolvedReferences
    return int(np.floor(value + epsilon))


def get_known_shape(tensor: tf.Tensor):
    dyn_shape = tf.shape(tensor)
    outputs_shape = [dyn_shape[i] if tensor.shape[i] is None else tensor.shape[i]
                     for i in range(len(tensor.shape))]
    return outputs_shape
