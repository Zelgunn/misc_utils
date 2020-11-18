import tensorflow as tf
import tensorflow_probability as tfp

from misc_utils.general import int_floor, int_ceil


def extract_most_likely_background(frames, buffer_size=None, k=1, frame_size=None):
    frames = tf.convert_to_tensor(frames)

    if frames.shape.rank < 4:
        frames = tf.expand_dims(frames, -1)

    if frame_size is not None:
        frames = tf.image.resize(frames, frame_size)
    votes_shape = frames.shape[1:] + [256]
    votes = tf.zeros(shape=votes_shape, dtype=tf.int32)

    if frames.dtype != tf.int32:
        if (frames.dtype == tf.float32) and (tf.reduce_all(frames <= 1.0)):
            frames *= 255.0
        frames = tf.cast(frames, tf.int32)

    buffer_size = frames.shape[0] if buffer_size is None else buffer_size
    for j in range(0, frames.shape[0], buffer_size):
        frames_votes = tf.one_hot(frames[j:j + buffer_size], depth=256, dtype=tf.int32)
        frames_votes = tf.reduce_sum(frames_votes, axis=0)
        votes += frames_votes

    k_votes_weights, k_votes = tf.math.top_k(votes, k, sorted=False)
    k_votes_weights, k_votes = tf.cast(k_votes_weights, tf.float32), tf.cast(k_votes, tf.float32)
    k_votes_weights /= tf.reduce_sum(k_votes_weights, axis=-1, keepdims=True)
    background = tf.reduce_sum(k_votes * k_votes_weights, axis=-1)
    background = tf.cast(background, tf.uint8)
    background = background.numpy()

    return background


def remove_most_likely_background(self, buffer_size=None, frame_size=None, k=1):
    frame_size = self.frame_size if frame_size is None else frame_size

    frames = self.read_all()
    frames = tf.convert_to_tensor(frames)

    if frames.shape.rank < 4:
        frames = tf.expand_dims(frames, -1)

    frames = tf.image.resize(frames, frame_size)
    votes_shape = frames.shape[1:] + [256]
    votes = tf.zeros(shape=votes_shape, dtype=tf.int32)

    if frames.dtype != tf.int32:
        if (frames.dtype == tf.float32) and (tf.reduce_all(frames <= 1.0)):
            frames *= 255.0
        frames = tf.cast(frames, tf.int32)

    buffer_size = frames.shape[0] if buffer_size is None else buffer_size
    for j in range(0, frames.shape[0], buffer_size):
        frames_votes = tf.one_hot(frames[j:j + buffer_size], depth=256, dtype=tf.int32)
        frames_votes = tf.reduce_sum(frames_votes, axis=0)
        votes += frames_votes

    _, k_votes = tf.math.top_k(votes, k, sorted=False)
    k_votes = tf.one_hot(k_votes, depth=256) != 0
    k_votes = tf.reduce_any(k_votes, axis=-2)
    k_votes = tf.expand_dims(k_votes, axis=0)

    frequency_filter = tf.ones(shape=[3, 3, 1, 1], dtype=tf.int32)
    gaussian_filter = get_gaussian_kernel(size=5, std=1.0)
    gaussian_filter = tf.expand_dims(tf.expand_dims(gaussian_filter, axis=-1), axis=-1)

    output_frames = []
    for j in range(0, frames.shape[0], buffer_size):
        buffer = frames[j:j + buffer_size]
        frames_votes = tf.one_hot(buffer, depth=256) != 0
        vote_in_top_k = tf.logical_and(k_votes, frames_votes)
        vote_in_top_k = tf.reduce_any(vote_in_top_k, axis=-1)  # for color depth
        if vote_in_top_k.shape[-1] > 1:
            vote_in_top_k = tf.reduce_any(vote_in_top_k, axis=-1, keepdims=True)  # for rgb

        mask = 1 - tf.cast(vote_in_top_k, tf.int32)
        mask = tf.nn.conv2d(mask, frequency_filter, strides=1, padding="SAME") > 3
        mask = tf.cast(mask, tf.float32)
        mask = tf.nn.conv2d(mask, gaussian_filter, strides=1, padding="SAME")

        masked_frames = tf.cast(buffer, tf.float32) * mask
        output_frames.append(masked_frames)

    output_frames = tf.concat(output_frames, axis=0)
    output_frames = tf.cast(output_frames, tf.uint8)
    output_frames = output_frames.numpy()

    return output_frames


def get_gaussian_kernel(size: int, std: float) -> tf.Tensor:
    """Makes 2D gaussian Kernel for convolution."""

    distribution = tfp.distributions.Normal(0.0, std)

    left_size = int_floor(size / 2)
    right_size = int_ceil(size / 2)
    values = distribution.prob(tf.range(start=-left_size, limit=right_size, dtype=tf.float32))
    gaussian_kernel = tf.einsum('i,j->ij', values, values)
    gaussian_kernel / tf.reduce_sum(gaussian_kernel)
    gaussian_kernel = tf.constant(gaussian_kernel)
    return gaussian_kernel
