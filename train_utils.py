import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.engine.training_utils import MetricsAggregator
import os
from time import time
from abc import abstractmethod
from typing import Dict

from modalities import RawVideo


def get_log_dir(base_dir):
    log_dir = os.path.join(base_dir, "log_{0}".format(int(time())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_model_json(model: Model, log_dir):
    filename = "{0}_config.json".format(model.name)
    filename = os.path.join(log_dir, filename)
    with open(filename, "w") as file:
        file.write(model.to_json())


def save_model_summary(model: Model, log_dir):
    filename = "{0}_summary.txt".format(model.name)
    filename = os.path.join(log_dir, filename)
    with open(filename, "w") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))


def save_model_info(model: Model, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # keras.utils.plot_model(model, os.path.join(log_dir, "{0}.png".format(model.name)))
    save_model_json(model, log_dir)
    save_model_summary(model, log_dir)


def augment_raw_video(modalities: Dict[str, tf.Tensor]
                      ) -> Dict[str, tf.Tensor]:
    if RawVideo.id() not in modalities:
        return modalities

    raw_video = modalities[RawVideo.id()]

    # raw_video = random_video_vertical_flip(raw_video)
    raw_video = random_video_horizontal_flip(raw_video)
    raw_video = tf.image.random_hue(raw_video, max_delta=0.1)
    raw_video = tf.image.random_brightness(raw_video, max_delta=0.1)

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
                       max_val=1.0
                       ) -> Dict[str, tf.Tensor]:
    if RawVideo.id() not in modalities:
        return modalities

    raw_video = modalities[RawVideo.id()]

    noise = tf.random.normal(tf.shape(raw_video), mean=noise_mean, stddev=noise_stddev, name="gaussian_noise")
    if min_val is not None:
        raw_video = tf.clip_by_value(raw_video + noise, min_val, max_val)

    modalities[RawVideo.id()] = raw_video
    return modalities


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 learning_rate,
                 step_offset=0
                 ):
        super(CustomLearningRateSchedule, self).__init__()
        self.learning_rate = learning_rate
        self.step_offset = step_offset

    def __call__(self, step):
        return self.call(step + self.step_offset)

    def call(self, step):
        return self.get_learning_rate(step)

    def get_learning_rate(self, step):
        return self._get_learning_rate(step, self.learning_rate)

    @staticmethod
    def _get_learning_rate(step, learning_rate):
        if callable(learning_rate):
            learning_rate = learning_rate(step)
        return learning_rate

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "step_offset": self.step_offset,
        }
        return config


class WarmupSchedule(CustomLearningRateSchedule):
    def __init__(self,
                 warmup_steps: int,
                 learning_rate=1e-3,
                 **kwargs):
        super(WarmupSchedule, self).__init__(learning_rate=learning_rate,
                                             **kwargs)
        self.warmup_steps = warmup_steps

    def call(self, step):
        factor = (step + 1) / self.warmup_steps
        base_learning_rate = self.get_learning_rate(step)
        return base_learning_rate * tf.math.minimum(factor, 1.0)

    def get_config(self):
        base_config = super(WarmupSchedule, self).get_config()
        config = {
            "warmup_steps": self.warmup_steps,
        }
        return {**base_config, **config}


class CyclicSchedule(CustomLearningRateSchedule):
    def __init__(self,
                 cycle_length: int,
                 learning_rate=1e-3,
                 max_learning_rate=1e-2,
                 **kwargs):
        super(CyclicSchedule, self).__init__(learning_rate=learning_rate,
                                             **kwargs)
        self.cycle_length = cycle_length
        self.max_learning_rate = max_learning_rate

    def call(self, step):
        factor = step / self.cycle_length
        factor = tf.truncatemod(factor, 1.0)
        factor = tf.cond(pred=factor > 0.5,
                         true_fn=lambda: 1.0 - factor,
                         false_fn=lambda: factor)
        factor *= 2

        base_learning_rate = self.get_learning_rate(step)
        max_learning_rate = self.get_max_learning_rate(step)
        learning_rate = factor * max_learning_rate + (1 - factor) * base_learning_rate

        return learning_rate

    def get_max_learning_rate(self, step):
        return self._get_learning_rate(step, self.max_learning_rate)

    def get_config(self):
        base_config = super(CyclicSchedule, self).get_config()
        config = {
            "cycle_length": self.cycle_length,
            "max_learning_rate": self.max_learning_rate,
        }
        return {**base_config, **config}


class LossAggregator(MetricsAggregator):
    def aggregate(self, batch_outs, batch_start=None, batch_end=None):
        for i in range(len(self.results)):
            loss = batch_outs[i]
            if not self.use_steps:
                loss *= (batch_end - batch_start)
            self.results[i] += loss

    def finalize(self):
        if not self.results:
            raise ValueError("Empty training data.")

        for i in range(len(self.results)):
            self.results[i] /= (self.num_samples or self.steps)
