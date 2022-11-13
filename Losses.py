"""
Custom losses for my Neural Networks
Loss function must take y_true, y_pred and return some form of value  "loss"
"""
import numpy as np
import tensorflow as tf
from keras.losses import Loss
import keras.backend as K


class MSLogScaledErrorLoss(Loss):

    def __init__(self, alpha: float = 1, epsilon: float = 10e-1):
        """
        Declare value of custom parameters
        """
        super(MSLogScaledErrorLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def call(self, y_true, y_pred) -> np.array:
        """
        Calculations of loss functions
        :param y_true: true values from dataset
        :param y_pred: values predicted by model
        :return:
        """
        first_bracket = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        second_bracket = tf.reduce_mean(tf.square(K.log(y_true + self.epsilon) - K.log(tf.abs(y_pred) + self.epsilon)), axis=-1)

        msls = first_bracket + (self.alpha * second_bracket)

        return msls
