from keras import layers, models, backend
import keras
import gc
from .metrics import angular_rotation
import numpy as np
import logging


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.angular_error = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.validation_data[0:-2])
        y_true = self.validation_data[-2]

        self.angular_error.append(np.mean(angular_rotation(y_true, y_pred_val)))

        return


def rmse_keras(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def free_memory(*args):
    """
    Takes in variables, deletes them and does manual garbage collect and clear session of keras backend

    :param *args: a variable amount of variables that you want deleted;

    """
    for var in args:
        del var
    gc.collect()
    backend.clear_session()


def create_parallel_cnn_model(x_shape, n_cameras, conv_layers, dense_layers, loss, optimizer):
    # ToDo: remove next condition and make a better solution
    if x_shape[-2] != x_shape[-3]:
        n_cameras = 1
        logging.info("Setting n_cameras to 1")

    input_shape = x_shape[1:]
    parallel_inputs = []
    parallel_outputs = []

    for i in range(n_cameras):
        model_input = layers.Input(input_shape)
        parallel_inputs.append(model_input)

        conv = model_input
        for conv_layer in conv_layers:
            conv = conv_layer(conv)

        parallel_outputs.append(conv)

    if n_cameras == 1:
        parallel_inputs = parallel_inputs[0]
        parallel_outputs = parallel_outputs[0]
        concat = parallel_outputs
    else:
        concat = layers.concatenate(parallel_outputs, axis=-1)

    flat = layers.Flatten()(concat)
    dense = flat
    for dense_layer in dense_layers:
        dense = dense_layer(dense)

    output_layer = layers.Dense(2)(dense)

    model = models.Model(inputs=parallel_inputs, outputs=output_layer)

    model.compile(loss=loss, optimizer=optimizer, metrics=[rmse_keras, 'accuracy'])
    return model


# conv(3x3, 4) -> conv(3x3, 4) -> concat -> flatten -> dense(64) -> dense(16) -> dense(2) -> rms loss
def create_basic_cnn_model(x_shape, n_cameras):
    conv_layers = [layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
                   layers.Conv2D(8, (3, 3), padding='same', activation='relu')]
    dense_layers = [layers.Dense(64, activation='relu'), layers.Dense(16, activation='relu')]

    return create_parallel_cnn_model(x_shape, n_cameras, conv_layers, dense_layers,
                                     loss='mean_squared_error', optimizer='adam')


def create_small_cnn_model(x_shape, n_cameras):
    conv_layers = [layers.Conv2D(4, (3, 3), padding='same', activation='relu'),
                   layers.Conv2D(4, (3, 3), padding='same', activation='relu')]
    dense_layers = [layers.Dense(32, activation='relu')]

    return create_parallel_cnn_model(x_shape, n_cameras, conv_layers, dense_layers,
                                     loss='mean_squared_error', optimizer='adam')


def create_big_cnn_model(x_shape, n_cameras):
    conv_layers = [layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
                   layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                   layers.Conv2D(64, (3, 3), padding='same', activation='relu')]
    dense_layers = [layers.Dense(256, activation='relu'), layers.Dense(64, activation='relu')]

    return create_parallel_cnn_model(x_shape, n_cameras, conv_layers, dense_layers,
                                     loss='mean_squared_error', optimizer='adam')


def create_dense_sep_channel_model(x_shape, n_cameras, layer_size, activation, loss, optimizer):
    # ToDo: remove next condition and make a better solution. It's exteremely dangerous
    if x_shape[-1] != 9:
        n_cameras = 1
        logging.info("Setting n_cameras to 1")

    input_shape = x_shape[1:]
    parallel_inputs = []
    parallel_outputs = []

    for i in range(n_cameras):
        model_input = layers.Input(input_shape)
        parallel_inputs.append(model_input)

        h_dense1 = layers.Dense(layer_size, activation=activation)(model_input)
        model_output = layers.Dense(layer_size, activation=activation)(h_dense1)

        parallel_outputs.append(model_output)

    if n_cameras == 1:
        parallel_inputs = parallel_inputs[0]
        parallel_outputs = parallel_outputs[0]
        concat = parallel_outputs
    else:
        concat = layers.concatenate(parallel_outputs)

    dense1 = layers.Dense(layer_size, activation=activation)(concat)
    dense2 = layers.Dense(2)(dense1)

    model = models.Model(inputs=parallel_inputs, outputs=dense2)

    model.compile(loss=loss, optimizer=optimizer, metrics=[rmse_keras, 'accuracy'])
    return model


# as close to the described model in the paper as possible
def create_original_model(x_shape, n_cameras):
    return create_dense_sep_channel_model(x_shape, n_cameras, layer_size=512, activation='relu',
                                          loss='mean_absolute_error', optimizer=keras.optimizers.Adagrad(lr=0.05)
                                          )


# changed loss and optimizer
def create_modified_original_model(x_shape, n_cameras):
    return create_dense_sep_channel_model(x_shape, n_cameras, layer_size=512, activation='relu',
                                          loss='mean_squared_error', optimizer='adam'
                                          )


# changed dense layer size, loss and optimizer
def create_smaller_modified_original_model(x_shape, n_cameras):
    return create_dense_sep_channel_model(x_shape, n_cameras, layer_size=64, activation='relu',
                                          loss='mean_squared_error', optimizer='adam'
                                          )



