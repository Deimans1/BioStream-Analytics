from signal import SIGINT, signal

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K
from tensorflow.keras import layers, metrics, constraints
from tensorflow import keras

#from tensorflow.python.keras.utils.generic_utils import register_keras_serializable

from utils.myconfig import myconfig


def handler(signalnum, frame):
    raise TypeError


@register_keras_serializable()
def MMSE_loss(y_true, y_pred):
    return tf.keras.backend.mean(
        tf.keras.backend.square(tf.abs(y_true - y_pred) + 1), axis=-1
    )


@register_keras_serializable()
def RSS_metric(y_true, y_pred):
    return tf.keras.backend.sum(tf.keras.backend.square((tf.abs(y_true - y_pred))))


@register_keras_serializable()
def coeff_determination(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def build_and_compile_model_LSTM(n_steps, n_features) -> tuple:
    model = keras.Sequential(
        [
            layers.LSTM(
                units=n_steps,
                activation="relu",
                input_shape=(n_steps, n_features),
                unroll=True,
            ),
            layers.Dropout(0.2),
            layers.Dense(10, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(
                1,
                kernel_constraint=constraints.NonNeg(),
                bias_constraint=constraints.NonNeg(),
            ),
        ]
    )

    model.compile(
        loss=MMSE_loss,
        optimizer=myconfig.custom_optimizer,
        metrics=[
            metrics.MeanAbsoluteError(),
            metrics.MeanSquaredError(),
            metrics.RootMeanSquaredError(),
            RSS_metric,
            coeff_determination,
        ],
    )
    # model.summary()
    return model


def ModelFit(model, train_x, train_y, valid_x, valid_y, early_stop) -> tuple:
    signal(SIGINT, handler)
    model.fit(
        train_x,
        train_y,
        validation_data=[valid_x, valid_y],
        verbose=0,
        batch_size=16,
        epochs=1000,
        callbacks=[early_stop],
    )

    results = model.evaluate(valid_x, valid_y)
    predictions = (model.predict(valid_x)).flatten()

    return (results, predictions, model)
