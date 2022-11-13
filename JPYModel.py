"""
File with USDJPY prediciton model subclassed from tf model
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import datetime as dt

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Normalization
from keras import losses, metrics, activations, optimizers

if TYPE_CHECKING:
    pass
    # import file with custom datatypes


class JPYModelLinaer(Model):

    def __init__(self, activation: str = "swish", **kwargs) -> None:
        """
        Declare layers and their specifications
        """
        super(JPYModelLinaer, self).__init__(**kwargs)
        # self.input_layer = Input(shape=29, name='input_layer')
        self.first_dense = Dense(units=128, activation=activation, name='first_dense')
        self.second_dense = Dense(units=256, activation=activation, name='second_dense')
        self.first_drop = Dropout(0.1, name='first_dropout')
        self.third_dense = Dense(units=512, activation=activation, name='third_dense')
        self.fourth_dense = Dense(units=512, activation=activation, name='fourth_dense')
        self.fifth_dense = Dense(units=128, activation=activation, name='fifth_dense')
        self.first_output = Dense(units=1, name='output_1')

    def call(self, inputs, training=None, mask=None):
        """
        Connect layers via functional API interface
        """

        # x = self.input_layer
        x = self.first_dense(inputs)
        x = self.second_dense(x)
        x = self.first_drop(x)
        x = self.third_dense(x)
        x = self.fourth_dense(x)
        x = self.fifth_dense(x)
        out1 = self.first_output(x)

        return out1

    def compile(self,
                optimizer=optimizers.Adam(),
                loss=losses.mean_absolute_error,
                metrics=[metrics.mean_absolute_error],
                epochs=30,
                batch_size=256,
                run_eagerly=True,
                **kwargs):
        super(JPYModelLinaer, self).\
            compile(optimizer=optimizer,
                    loss=loss,
                    metrics=metrics,
                    run_eagerly=True,
                    **kwargs)

    def fit(self,
            x=None,
            y=None,
            batch_size=128,
            epochs=30,
            **kwargs):
        super(JPYModelLinaer, self).\
            fit(x=x,
                y=y,
                epochs=epochs,
                batch_size=batch_size,
                **kwargs)

    @staticmethod
    def plot_results(true_y, pred_y):
        fig = plt.figure()
        plt.plot(true_y, color='green', label='True values')
        plt.plot(pred_y, color='red', label='Predicted values')
        plt.title('Linear prediction of values')
        plt.show()

    def save_model(self, model_name=None):
        if not model_name:
            model_name = f"JPYLinear_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
        super(JPYModelLinaer, self).save(f"models/models/{model_name}")

    @staticmethod
    def load_model(model_name=None):
        return tf.keras.models.load_model(f"models/models/{model_name}")


class JPYModelClasssifier(Model):

    def __init__(self, input_shape: int = 29, **kwargs) -> None:
        """
        Declare layers and their specifications
        """
        super(JPYModelClasssifier, self).__init__(**kwargs)
        self.first_dense = Dense(units=128, activation='relu', name='first_dense')
        self.second_dense = Dense(units=128, activation='relu', name='second_dense')
        self.first_drop = Dropout(0.1, name='first_dropout')
        self.third_dense = Dense(units=128, activation='relu', name='third_dense')
        self.first_output = Dense(units=1, name='output_1', activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        """
        Connect layers via functional API interface
        """

        # x = self.input_layer
        x = self.first_dense(inputs)
        x = self.second_dense(x)
        x = self.first_drop(x)
        x = self.third_dense(x)
        out1 = self.first_output(x)

        return out1

    def compile(self,
                optimizer='adam',
                loss='mean_absolute_error',
                metrics='mean_square_error',
                epochs=30,
                batch_size=128,
                **kwargs):
        super(JPYModelClasssifier, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

