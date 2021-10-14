import tensorflow as tf
from tensorflow.keras.layers import Input
import numpy as np
import os
from typing import List, Union, Tuple

SingleTensor = Union[tf.Tensor, np.ndarray]
TensorList = Union[SingleTensor, List[SingleTensor]]


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


def get_known_shape(tensor: tf.Tensor) -> List[Union[tf.Tensor, int]]:
    dyn_shape = tf.shape(tensor)
    outputs_shape = [dyn_shape[i] if tensor.shape[i] is None else tensor.shape[i]
                     for i in range(len(tensor.shape))]
    return outputs_shape


def to_input_layers(input_shapes: List, names: Union[List, str] = None):
    if not isinstance(input_shapes, (tuple, list, tf.TensorShape)):
        raise TypeError("`input_shapes` must either be a list, a tuple or a TensorShape, got type {}"
                        .format(type(input_shapes)))
    elif len(input_shapes) == 0:
        raise ValueError("`input_shapes` or an element of `input_shapes` is empty")

    if any(isinstance(shape, (tuple, list, tf.TensorShape)) for shape in input_shapes):
        if names is None:
            layers = [to_input_layers(shapes) for shapes in input_shapes]
        else:
            layers = [to_input_layers(shapes, _names) for shapes, _names in zip(input_shapes, names)]
        return layers
    else:
        layer = Input(batch_shape=input_shapes, name=names)
        return layer


def list_dir_recursive(path: str, suffix: Union[Tuple[str, ...], str] = None) -> List[str]:
    listed_files = []
    no_suffix = suffix is None
    for root, dirs, files in os.walk(path):
        for file in files:
            if no_suffix or file.endswith(suffix):
                file = os.path.join(root, file)
                listed_files.append(file)
    return listed_files


@tf.function
def expand_dims_to_rank(x: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    rank_offset = target.shape.rank - x.shape.rank
    x = tf.reshape(x, tf.concat([tf.shape(x), [1] * rank_offset], axis=0))
    return x


def unpack_test_sample(sample: List[TensorList], modality_count: int) -> Tuple[TensorList, TensorList, tf.Tensor]:
    if modality_count == 1:
        if len(sample) == 3:
            inputs, outputs, labels = sample
        elif len(sample) == 2:
            inputs, labels = sample
            outputs = inputs
        else:
            raise ValueError("Expected a list/tuple containing either 2 or 3 elements. Got {}".format(len(sample)))
    elif modality_count == len(sample) + 1:
        if len(sample) == 2:
            sample = (sample[0], sample[0], sample[1])
        if len(sample) == 3:
            both_lists = isinstance(sample[0], (list, tuple)) and isinstance(sample[1], (list, tuple))
            one_list = isinstance(sample[0], (list, tuple)) or isinstance(sample[1], (list, tuple))
            if both_lists:
                lists_match = len(sample[0]) == len(sample[1])
                if lists_match:
                    inputs, outputs, labels = sample
                else:
                    raise ValueError("[Multimodal setup] Inputs and Outputs do not contain the same amount of "
                                     "modalities. Got {} and {}.".format(len(sample[0]), len(sample[1])))
            elif one_list:
                raise ValueError("[Multimodal setup] Inputs and Ouputs do not match, only one is a list/tuple.")
            else:
                *inputs, labels = sample
                outputs = inputs
        else:
            *inputs, labels = sample
            outputs = inputs

    if modality_count == 1:
        if isinstance(sample[0], tf.Tensor) and len(sample) == 3:
            inputs, outputs, labels = sample
    elif len(sample) == modality_count + 1:
        *inputs, labels = sample
        outputs = inputs
    else:
        raise ValueError("Length of sample does not match. Found {} but expected {} or 3.".format(len(sample),
                                                                                                  modality_count + 1))
    return inputs, outputs, labels


def get_model_inputs_count(model: tf.keras.Model) -> int:
    if model.inputs is not None:
        return len(model.inputs)

    # noinspection PyProtectedMember
    if model._saved_model_inputs_spec is not None:
        # noinspection PyProtectedMember
        if isinstance(model._saved_model_inputs_spec, tf.TensorSpec):
            return 1
        return len(model._saved_model_inputs_spec)

    raise RuntimeError("Could not determine the number of inputs for the model. Consider using model._set_inputs.")
