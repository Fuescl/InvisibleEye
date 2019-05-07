import numpy as np


def angular_rotation(v: np.array, u: np.array) -> np.array:
    _v = np.concatenate([np.tan(np.deg2rad(v)), np.full((v.shape[0], 1), -1)], axis=1)
    _u = np.concatenate([np.tan(np.deg2rad(u)), np.full((u.shape[0], 1), -1)], axis=1)

    res = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(_v, _u), axis=1), np.sum(_v * _u, axis=1)))
    return res


def euclidean_distance(a: np.array, b: np.array) -> np.array:
    return np.linalg.norm(a - b, axis=1)


def global_error(y_true: np.array, y_pred: np.array, error_func=angular_rotation) -> np.array:
    return np.mean([np.mean(error_func(y_true[i], y_pred[i])) for i in range(len(y_true))])
