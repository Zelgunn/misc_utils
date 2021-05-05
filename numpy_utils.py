import numpy as np


def fast_concatenate_0(a_tuple):
    if len(a_tuple) == 1:
        return a_tuple[0]
    else:
        return np.concatenate(a_tuple, axis=0)


def normalize(x: np.ndarray, axis=None) -> np.ndarray:
    # noinspection PyArgumentList
    x_min = x.min(axis=axis, keepdims=True)
    # noinspection PyArgumentList
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def safe_normalize(x: np.ndarray, axis=None) -> np.ndarray:
    if np.any(np.isnan(x)):
        raise ValueError("Could not normalize array safely, it contains NaN values.")

    # noinspection PyArgumentList
    x_min = x.min(axis=axis, keepdims=True)
    # noinspection PyArgumentList
    x_max = x.max(axis=axis, keepdims=True)

    x_range = x_max - x_min
    if np.any(x_range == 0.0):
        x_range = np.where(x_range == 0.0, np.ones_like(x_range), x_range)

    return (x - x_min) / x_range


def standardize(x: np.ndarray, axis=None) -> np.ndarray:
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True)
    return (x - x_mean) / x_std


def safe_standardize(x: np.ndarray, axis=None) -> np.ndarray:
    if np.any(np.isnan(x)):
        raise ValueError("Could not standardize array safely, it contains NaN values.")

    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True)

    if np.any(x_std == 0.0):
        x_std = np.where(x_std == 0.0, np.ones_like(x_std), x_std)

    return (x - x_mean) / x_std
