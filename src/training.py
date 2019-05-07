from .keras_utils import free_memory, Metrics
import logging
import numpy as np
from typing import List


def prepare_data_dense(x: np.array, use_cameras=(True, True, True, True)) -> List[np.array]:
    assert sum(use_cameras) > 0

    x_tr = x.reshape(x.shape[:-2] + (-1,))
    res = [x_tr[i] for i in range(x_tr.shape[0]) if use_cameras[i]]
    return res


def prepare_data_dense_concat_cameras(x: np.array, use_cameras=(True, True, True, True)) -> List[np.array]:
    assert sum(use_cameras) > 0

    x_tr = x.reshape(x.shape[:-2] + (-1,))
    res = [np.concatenate([x_tr[i] for i in range(x_tr.shape[0]) if use_cameras[i]], axis=-1)]
    return res


def prepare_data_cnn(x: np.array, use_cameras=(True, True, True, True)) -> List[np.array]:
    res = [x[i].reshape(x[i].shape + (1, )) for i in range(x.shape[0]) if use_cameras[i]]
    return res


def prepare_data_cnn_concat_cameras(x: np.array, use_cameras=(True, True, True, True)) -> List[np.array]:
    return [np.concatenate([x[i].reshape(x[i].shape + (1, )) for i in range(x.shape[0]) if use_cameras[i]], axis=-2)]


def train_subject_models(x_train, y_train, x_test, y_test, use_cameras, model_generator, x_preprocess, fit_kwargs):
    logging.info("Started training subject models with use_cameras: {}".format(use_cameras))
    results = []
    x_train = [x_preprocess(x_tr, use_cameras) for x_tr in x_train]
    x_test = [x_preprocess(x_te, use_cameras) for x_te in x_test]
    for i in range(len(x_train)):
        model = model_generator(x_train[i][0].shape, sum(use_cameras))
        metrics = Metrics()
        model.fit(x_train[i], y_train[i], shuffle=True, validation_data=(x_test[i], y_test[i]), callbacks=[metrics],
                  **fit_kwargs)
        results.append(metrics.angular_error)
        free_memory(model)
        logging.info("Subject {} error: {}".format(i, round(metrics.angular_error[-1], 2)))

    return np.array(results)
