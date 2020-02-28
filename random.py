import tensorflow as tf
from typing import Dict

from modalities import RawVideo


@tf.function
def coin_flip(seed: int = None):
    return tf.random.uniform(shape=[], minval=0.0, maxval=1.0, seed=seed) < 0.5


def augment_raw_video(modalities: Dict[str, tf.Tensor],
                      seed: int = None,
                      ) -> Dict[str, tf.Tensor]:
    if RawVideo.id() not in modalities:
        return modalities

    raw_video = modalities[RawVideo.id()]

    # raw_video = random_video_vertical_flip(raw_video, seed=seed)
    raw_video = random_video_horizontal_flip(raw_video, seed=seed)
    raw_video = tf.image.random_hue(raw_video, max_delta=0.1, seed=seed)
    raw_video = tf.image.random_brightness(raw_video, max_delta=0.1, seed=seed)

    modalities[RawVideo.id()] = raw_video
    return modalities


# region Random video flip
def random_video_vertical_flip(video: tf.Tensor,
                               seed: int = None,
                               scope_name: str = "random_video_vertical_flip"
                               ) -> tf.Tensor:
    return random_video_flip(video, 1, seed, scope_name)


def random_video_horizontal_flip(video: tf.Tensor,
                                 seed: int = None,
                                 scope_name: str = "random_video_horizontal_flip"
                                 ) -> tf.Tensor:
    return random_video_flip(video, 2, seed, scope_name)


def random_video_flip(video: tf.Tensor,
                      flip_index: int,
                      seed: int,
                      scope_name: str
                      ) -> tf.Tensor:
    """Randomly (50% chance) flip an video along axis `flip_index`.
    Args:
        video: 5-D Tensor of shape `[batch, time, height, width, channels]` or
               4-D Tensor of shape `[time, height, width, channels]`.
        flip_index: Dimension along which to flip video. Time: 0, Vertical: 1, Horizontal: 2
        seed: A Python integer. Used to create a random seed. See `tf.set_random_seed` for behavior.
        scope_name: Name of the scope in which the ops are added.
    Returns:
        A tensor of the same type and shape as `video`.
    Raises:
        ValueError: if the shape of `video` not supported.
    """
    with tf.name_scope(scope_name) as scope:
        video = tf.convert_to_tensor(video, name="video")
        shape = video.get_shape()

        if shape.ndims == 4:
            uniform_random = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, seed=seed)
            flip_condition = tf.less(uniform_random, 0.5)
            flipped = tf.reverse(video, [flip_index])
            outputs = tf.cond(pred=flip_condition,
                              true_fn=lambda: flipped,
                              false_fn=lambda: video,
                              name=scope)

        elif shape.ndims == 5:
            batch_size = tf.shape(video)[0]
            uniform_random = tf.random.uniform(shape=[batch_size], minval=0.0, maxval=1.0, seed=seed)
            uniform_random = tf.reshape(uniform_random, [batch_size, 1, 1, 1, 1])
            flips = tf.round(uniform_random)
            flips = tf.cast(flips, video.dtype)
            flipped = tf.reverse(video, [flip_index + 1])
            outputs = flips * flipped + (1.0 - flips) * video

        else:
            raise ValueError("`video` must have either 4 or 5 dimensions but has {} dimensions.".format(shape.ndims))

        return outputs


# endregion

def add_gaussian_noise(modalities: Dict[str, tf.Tensor],
                       noise_mean=0.0,
                       noise_stddev=0.1,
                       min_val=0.0,
                       max_val=1.0,
                       seed: int = None,
                       ) -> Dict[str, tf.Tensor]:
    if RawVideo.id() not in modalities:
        return modalities

    raw_video = modalities[RawVideo.id()]

    noise = tf.random.normal(tf.shape(raw_video), mean=noise_mean, stddev=noise_stddev,
                             name="gaussian_noise", seed=seed)
    if min_val is not None:
        raw_video = tf.clip_by_value(raw_video + noise, min_val, max_val)

    modalities[RawVideo.id()] = raw_video
    return modalities
