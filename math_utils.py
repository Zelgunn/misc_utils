import tensorflow as tf
from tensorflow.python.keras import backend
from typing import List, Tuple, Callable
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


def reduce_mean_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_mean, keepdims=keepdims)


def reduce_sum_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_sum, keepdims=keepdims)


def reduce_prod_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_prod, keepdims=keepdims)


def reduce_std_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.math.reduce_std, keepdims=keepdims)


def reduce_adjusted_std_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, reduce_adjusted_stddev, keepdims=keepdims)


def reduce_min_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_min, keepdims=keepdims)


def reduce_max_from(inputs: tf.Tensor, start_axis=1, keepdims=False) -> tf.Tensor:
    return reduce_from(inputs, start_axis, tf.reduce_max, keepdims=keepdims)


def reduce_from(inputs: tf.Tensor, start_axis: int, fn: Callable, **kwargs):
    if start_axis < 0:
        start_axis = inputs.shape.rank + start_axis
    reduction_axis = tuple(range(start_axis, inputs.shape.rank))
    return fn(inputs, axis=reduction_axis, **kwargs)


def reduce_adjusted_stddev(inputs: tf.Tensor, axis: int, keepdims=False) -> tf.Tensor:
    inputs_shape = tf.shape(inputs)
    sample_dims = tf.gather(inputs_shape, axis)
    sample_size = tf.math.reduce_prod(input_tensor=sample_dims)
    sample_stddev = tf.math.reduce_std(input_tensor=inputs, axis=axis, keepdims=keepdims)
    min_stddev = tf.math.rsqrt(tf.cast(sample_size, inputs.dtype))
    adjusted_stddev = tf.maximum(sample_stddev, min_stddev)
    return adjusted_stddev


def get_mean_and_stddev(inputs: tf.Tensor, start_axis=1) -> Tuple[tf.Tensor, tf.Tensor]:
    sample_means = reduce_mean_from(inputs=inputs, start_axis=start_axis, keepdims=True)
    sample_stddev = reduce_adjusted_std_from(inputs=inputs, start_axis=start_axis, keepdims=True)
    return sample_means, sample_stddev


def standardize_from(inputs: tf.Tensor, start_axis=1) -> tf.Tensor:
    sample_means, sample_stddev = get_mean_and_stddev(inputs=inputs, start_axis=start_axis)
    outputs = (inputs - sample_means) / sample_stddev
    return outputs
