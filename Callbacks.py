"""
Files with custom callbacks
"""

import datetime as dt
import os
import tensorflow as tf


from keras.callbacks import Callback



class InfoCallback(Callback):

    def on_train_batch_begin(self, batch, logs=None):
        print(f'Training: batch {batch} begins at {dt.datetime.now().time()}')

    def on_train_batch_end(self, batch, logs=None):
        print(f'Training: batch {batch} ends at {dt.datetime.now().time()}')


class DetectOverfittingCallback(Callback):

    def __init__(self, threshold=0.8):
        """
        Initialize parameters for future use
        :param threshold: how high ratio of validation loss to normal loss can be
        :return:
        """
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        ratio = logs['val_loss'] / logs['loss']
        print(f"\nEpoch: {epoch}, val/train loss ratio: {ratio}")

        if ratio > self.threshold:
            print("Stopped training. Model overfiting..")
            self.model.stop_training = True


class TensorBoardCallback(Callback):

    def __init__(self, directory="logs"):
        self.logdir = os.path.join(directory, dt.datetime.now().strftime("%Y%m%d_%H%M"))

    def on_epoch_end(self, epoch, logs=None):
        tf.keras.callbacks.TensorBoard(log_dir=self.logdir)


class EarlyStoppingCallback(Callback):
    def __init__(self, **kwargs):
        patience = kwargs.pop('patience', 5)
        min_delta = kwargs.pop('min_delta', 0.05)
        baseline = kwargs.pop('baseline', 0.8)
        mode = kwargs.pop('mode', 'min')
        monitor = kwargs.pop('monitor', 'val_loss')
        restore_best_weights = kwargs.pop('restore_best_weights', True)

        tf.keras.callbacks.EarlyStopping(patience=patience,
                                         min_delta=min_delta,
                                         baseline=baseline,
                                         mode=mode,
                                         monitor=monitor,
                                         restore_best_weights=restore_best_weights,
                                         **kwargs)
