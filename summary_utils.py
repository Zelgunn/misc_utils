import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.ops import summary_ops_v2
import numpy as np
import matplotlib.cm
import time
from typing import List, Union

from misc_utils.general import to_input_layers, to_list
from misc_utils.math_utils import normalize_from


# region Image / Video
# Inspired by : https://github.com/alexlee-gk
def encode_gif(images: np.ndarray,
               fps: float or int):
    """Encodes numpy images into gif string.

    Args:
      images: A 4-D `uint8` `np.array` of shape `[time, height, width, channels]` where `channels` is 1 or 3.
      fps: frames per second of the animation

    Returns:
      The encoded gif string.

    Raises:
      IOError: If the ffmpeg command returns an error.
    """

    from subprocess import Popen, PIPE

    height, width, channels = np.shape(images)[1:]
    if channels not in [1, 3]:
        raise ValueError("Channels not in [1,3], got {} from inputs with shape {}".format(channels, np.shape(images)))
    channels_name = "gray" if channels is 1 else "rgb24"
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-r", "{0:.02f}".format(fps),
        "-s", "{0}x{1}".format(width, height),
        "-pix_fmt", channels_name,
        "-i", "-",
        "-filter_complex", "[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse",
        "-r", "{0:.02f}".format(fps),
        "-f", "gif",
        "-"]

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    for image in images:
        process.stdin.write(image.tostring())
    out, error = process.communicate()

    if process.returncode:
        error = '\n'.join([' '.join(cmd), error.decode('utf8')])
        raise IOError(error)

    del process
    return out


# Inspired by : https://github.com/alexlee-gk
def gif_summary(name: str,
                data: tf.Tensor,
                fps: int,
                step: int = None,
                max_outputs=3):
    """Write a gif summary.

    Args:
        name: A name for this summary. The summary tag used for TensorBoard will
            be this name prefixed by any active name scopes.
        data: A 5-D `uint8` `Tensor` of shape `[k, time, height, width, channels]`
            where `k` is the number of gifs and `channels` is either 1 or 3.
            Any of the dimensions may be statically unknown (i.e., `None`).
            Floating point data will be clipped to the range [0,1).
        fps: frames per second of the gif.
        step: Explicit `int64`-castable monotonic step value for this summary. If
            omitted, this defaults to `tf.summary.experimental.get_step()`, which must
            not be None.
        max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this
            many gifs will be emitted at each step. When more than
            `max_outputs` many gifs are provided, the first `max_outputs` many
            images will be used and the rest silently discarded.
    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.
    """
    summary_scope = tf.summary.experimental.summary_scope(name=name,
                                                          default_name='image_summary',
                                                          values=[data, max_outputs, step])

    batch_size, length, height, width, channels = data.shape
    batch_size = min(batch_size, max_outputs)

    with summary_scope as (tag, _):
        tf.debugging.assert_rank(data, 5)

        summary = summary_pb2.Summary()

        if tf.executing_eagerly():
            data = data.numpy()
        else:
            session = tf.compat.v1.keras.backend.get_session()
            data = session.run(data)

        for i in range(batch_size):
            ith_image_summary = summary_pb2.Summary.Image()
            ith_image_summary.height = height
            ith_image_summary.width = width
            ith_image_summary.colorspace = channels

            try:
                ith_image_summary.encoded_image_string = encode_gif(data[i], fps)
            except (IOError, OSError) as exception:
                raise IOError("Unable to encode images to a gif string because either ffmpeg is "
                              "not installed or ffmpeg returned an error: {}.".format(repr(exception)))

            summary_tag = "{}/gif".format(tag) if (batch_size == 1) else "{}/gif/{}".format(tag, i)

            summary.value.add(tag=summary_tag, image=ith_image_summary)

        event = event_pb2.Event(summary=summary)
        event.wall_time = time.time()
        event.step = step

        summary_ops_v2.import_event(event.SerializeToString(), name="scope")


def image_summary(name: str,
                  data: tf.Tensor,
                  fps=25,
                  step: int = None,
                  max_outputs=3):
    """Write an image or a gif summary.

    Args:
        name: A name for this summary. The summary tag used for TensorBoard will
            be this name prefixed by any active name scopes.
        data: A 4-D or a 5-D `uint8` `Tensor` of shape `[k, Optional[time], height, width, channels]`
            where `k` is the number of gifs and `channels` is either 1 or 3.
            Any of the dimensions may be statically unknown (i.e., `None`).
            Floating point data will be clipped to the range [0,1).
        fps: frames per second of the gif.
        step: Explicit `int64`-castable monotonic step value for this summary. If
            omitted, this defaults to `tf.summary.experimental.get_step()`, which must
            not be None.
        max_outputs: Optional `int` or rank-0 integer `Tensor`. At most this
            many gifs will be emitted at each step. When more than
            `max_outputs` many gifs are provided, the first `max_outputs` many
            images will be used and the rest silently discarded.
    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.
    """
    rank = data.shape.ndims
    assert rank >= 3

    if rank > 5:
        data = merge_video_time_dimensions(data)

    if rank == 3:
        data = tf.expand_dims(data, axis=-1)
        rank = 4

    if rank == 4:
        return tf.summary.image(name, data, step=step, max_outputs=max_outputs)
    else:
        return gif_summary(name, data, fps, step, max_outputs)


def merge_video_time_dimensions(video: tf.Tensor) -> tf.Tensor:
    video_shape = tf.shape(video)
    batch_size, height, width, channels = video_shape[0], video_shape[-3], video_shape[-2], video_shape[-1]
    length = tf.reduce_prod(video_shape[1:-3])
    return tf.reshape(video, [batch_size, length, height, width, channels])


def check_image_video_rank(data: Union[tf.Tensor, List[tf.Tensor]]):
    if isinstance(data, list) or isinstance(data, tuple):
        for sample in data:
            check_image_video_rank(sample)

    elif data.shape.rank < 3:
        raise ValueError("Incorrect rank for images/video, expected rank >= 3, got {} with rank {}."
                         .format(data.shape, data.shape.rank))


def use_video_summary(data: tf.Tensor) -> bool:
    return data.shape.rank >= 5


# endregion

# region Network packets
def network_packet_summary(name: str,
                           network_packets: tf.Tensor,
                           step: int = None,
                           max_outputs=3,
                           zoom: int = None,
                           normalize=True,
                           cmap="GnBu"
                           ):
    if cmap is not None:
        as_image = tf_apply_cmap(network_packets, cmap=cmap, color_count=256, normalize=normalize)
    else:
        as_image = tf.expand_dims(network_packets, axis=-1)
        if normalize:
            as_image = normalize_from(as_image, as_image.shape.rank - 2)

    if zoom is not None:
        image_size = tf.shape(network_packets)[1:3] * zoom
        as_image = tf.image.resize(as_image, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.summary.image(name, as_image, step=step, max_outputs=max_outputs)


# endregion

def convert_tensors_uint8(tensors: Union[tf.Tensor, List[tf.Tensor]]) -> List[tf.Tensor]:
    tensors = to_list(tensors)
    tensors = [convert_tensor_uint8(tensor) for tensor in tensors]
    return tensors


def convert_tensor_uint8(tensor) -> tf.Tensor:
    tensor: tf.Tensor = tf.convert_to_tensor(tensor)
    tensor_min = tf.reduce_min(tensor)
    tensor_max = tf.reduce_max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    normalized = tf.cast(tensor * tf.constant(255, dtype=tensor.dtype), tf.uint8)
    return normalized


def tf_function_summary(tf_function, input_shapes: List, step=None, name=None):
    inputs = to_input_layers(input_shapes)
    inputs = to_list(inputs)
    graph = tf_function.get_concrete_function(*inputs).graph
    return summary_ops_v2.graph(graph, step=step, name=name)


@tf.function
def tf_apply_cmap(tensor: tf.Tensor, cmap: str, color_count: int, normalize: bool = True):
    if normalize:
        tensor = normalize_from(tensor, tensor.shape.rank - 2)

    cmap = matplotlib.cm.get_cmap(cmap, color_count)
    color_range = np.arange(color_count, dtype=np.float32) / (color_count - 1)
    colors = cmap(color_range)
    colors = tf.constant(colors[:, :3], dtype=tf.float32)

    indices = tf.cast(tf.round(tensor * tf.cast(color_count - 1, tf.float32)), tf.int32)
    result = tf.gather(colors, indices)

    return result
