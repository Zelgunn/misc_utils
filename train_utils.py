import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.engine.training_utils import MetricsAggregator
from tensorflow.python.keras.saving import hdf5_format, saving_utils
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.util import serialization
import h5py
import json
import os
from time import time
from typing import Union, Optional


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


# region Learning rate schedules
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


# endregion

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


# region HDF5 / H5PY
class SharedHDF5(object):
    def __init__(self, filepath: Union[str, h5py.File], mode: str = None):
        self.filepath = filepath
        self.mode = mode

        self.file: Optional[h5py.File] = None
        self.filepath_is_file = isinstance(filepath, h5py.File)

        if not self.filepath_is_file and mode is None:
            raise ValueError("Mode must be specified is filepath is not a h5py.File.")

    def __enter__(self):
        if self.filepath_is_file:
            self.file = self.filepath
        else:
            self.file = h5py.File(name=self.filepath, mode=self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.flush()
        if not self.filepath_is_file:
            self.file.close()
        self.file = None


def save_model_to_hdf5(hdf5_group: h5py.Group, model: Model, model_id: str):
    """Saves a model to a HDF5 file.

    The saved model contains:
        - the model"s configuration (topology)
        - the model"s weights

    Thus the saved model can be reinstantiated in the exact same state, without any of the code used for model
    definition or training.

    Arguments:
        hdf5_group: A pointer to a HDF5 group to save the model.
        model: Keras model instance to be saved.
        model_id: ID of the model.
    Raises:
        ImportError: if h5py is not available.
    """

    model_metadata = saving_utils.model_metadata(model=model, include_optimizer=False, require_config=True)
    for k, v in model_metadata.items():
        if isinstance(v, (dict, list, tuple)):
            v = json.dumps(v, default=serialization.get_json_type).encode("utf8")
        hdf5_group.attrs[k] = v

    model_weights_group = hdf5_group.create_group("model_{}_weights".format(model_id))
    model_layers = model.layers
    hdf5_format.save_weights_to_hdf5_group(model_weights_group, model_layers)


def save_optimizer_weights_to_hdf5_group(hdf5_group: h5py.Group, optimizer: OptimizerV2, optimizer_id: str):
    """Saves optimizer weights of a optimizer to a HDF5 group.

    Arguments:
        hdf5_group: A pointer to a HDF5 group to save the optimizer's weights.
        optimizer: Optimizer instance.
        optimizer_id: Name of the optimizer (for ID).
    """

    symbolic_weights = getattr(optimizer, "weights")
    if symbolic_weights:
        weight_names = [str(weights.name).encode("utf8") for weights in symbolic_weights]
        weight_values = optimizer.get_weights()

        weights_group = hdf5_group.create_group("optimizer_{}_weights".format(optimizer_id))
        hdf5_format.save_attributes_to_hdf5_group(weights_group, "weight_names", weight_names)

        for name, value in zip(weight_names, weight_values):
            weights_dataset = weights_group.create_dataset(name, value.shape, dtype=value.dtype)
            if not value.shape:
                weights_dataset[()] = value
            else:
                weights_dataset[:] = value


def load_optimizer_weights_from_hdf5_group(hdf5_group: h5py.Group, optimizer: OptimizerV2, optimizer_id: str):
    """Load optimizer weights from a HDF5 group.

    Arguments:
        hdf5_group: A pointer to a HDF5 group to load the optimizer's weights from.
        optimizer: Optimizer instance.
        optimizer_id: Name of the optimizer (for ID).

    """
    weights_group = hdf5_group["optimizer_{}_weights".format(optimizer_id)]
    optimizer_weight_names = hdf5_format.load_attributes_from_hdf5_group(weights_group, "weight_names")
    weights = [weights_group[weight_name] for weight_name in optimizer_weight_names]
    optimizer.set_weights(weights)

# endregion
