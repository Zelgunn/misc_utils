import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.engine.training_utils import MetricsAggregator
import os
from time import time


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


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 learning_rate,
                 step_offset=0
                 ):
        super(CustomLearningRateSchedule, self).__init__()
        self.learning_rate = learning_rate
        self._step_offset = tf.Variable(step_offset, trainable=False, dtype=tf.float32)

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

    @property
    def step_offset(self):
        return self._step_offset

    @step_offset.setter
    def step_offset(self, value):
        self._step_offset.assign(value)

    def get_config(self):
        config = {
            "learning_rate": self.learning_rate,
            "step_offset": self.step_offset.numpy(),
        }
        return config


class ScaledSchedule(CustomLearningRateSchedule):
    def __init__(self, learning_rate, scale, **kwargs):
        super(ScaledSchedule, self).__init__(learning_rate=learning_rate, **kwargs)
        self.scale = scale

    def call(self, step):
        scale = self.scale
        if callable(scale):
            scale = scale(step)

        return super(ScaledSchedule, self).call(step) * scale


class WarmupSchedule(CustomLearningRateSchedule):
    def __init__(self,
                 warmup_steps: int,
                 learning_rate=1e-3,
                 **kwargs):
        super(WarmupSchedule, self).__init__(learning_rate=learning_rate,
                                             **kwargs)
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
        self.steps = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, step):
        self.steps.assign_add(1)
        factor = self.steps / self.warmup_steps
        base_learning_rate = super(WarmupSchedule, self).call(step)
        return base_learning_rate * tf.math.minimum(factor, 1.0)

    def get_config(self):
        base_config = super(WarmupSchedule, self).get_config()
        config = {
            "warmup_steps": self.warmup_steps.numpy(),
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

        base_learning_rate = super(CyclicSchedule, self).call(step)
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
