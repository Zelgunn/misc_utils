import tensorflow as tf
from tensorflow.python.keras import backend
from typing import List, Tuple
import numpy as np


@tf.function
def diff(tensor, axis=-1):
    if axis < 0:
        axis = tf.rank(tensor) + axis

    offset = tf.zeros([axis], dtype=tf.int32)
    offset = tf.concat([offset, [1]], axis=0)

    partial_shape = tf.shape(tensor)[:axis + 1]

    left = tf.strided_slice(tensor, begin=tf.zeros_like(offset), end=partial_shape - offset)
    right = tf.strided_slice(tensor, begin=offset, end=partial_shape)

    return right - left


@tf.function
def lerp(a, b, x):
    return (1.0 - x) * a + b * x


def join_distributions_statistics(counts: List[int],
                                  means: List[float],
                                  stddevs: List[float]
                                  ) -> Tuple[int, float, float]:
    count, mean, stddev = counts[0], means[0], stddevs[0]

    for i in range(1, len(counts)):
        count, mean, stddev = join_two_distributions_statistics(count_1=count, count_2=counts[i],
                                                                mean_1=mean, mean_2=means[i],
                                                                stddev_1=stddev, stddev_2=stddevs[i])

    return count, mean, stddev


def join_two_distributions_statistics(count_1: int, count_2: int,
                                      mean_1: float, mean_2: float,
                                      stddev_1: float, stddev_2: float,
                                      ) -> Tuple[int, float, float]:
    count = count_1 + count_2
    mean = (mean_1 * count_1 + mean_2 * count_2) / count

    variance_1 = np.square(stddev_1)
    variance_2 = np.square(stddev_2)
    variance_weighted_sum = (count_1 - 1) * variance_1 + (count_2 - 1) * variance_2
    mean_distance = np.square(mean_1 - mean_2)
    variance = (variance_weighted_sum + (count_1 * count_2) * mean_distance / count) / (count - 1)
    stddev = np.sqrt(variance)

    return count, mean, stddev


def squash(tensor: tf.Tensor, axis: int) -> tf.Tensor:
    squared_norm = tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=True)
    norm = tf.sqrt(squared_norm + backend.epsilon())

    squash_factor = squared_norm / (1 + squared_norm)
    unit_vector = tensor / norm

    result = unit_vector * squash_factor
    return result
