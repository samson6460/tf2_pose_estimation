# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Distribution-based loss functions.
"""

import tensorflow as tf

epsilon = 1e-07


def balanced_categorical_crossentropy(class_weight=1):
    def _balanced_categorical_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        ce = -tf.reduce_sum(
            (y_true*tf.math.log(y_pred)*class_weight), axis=-1)
        return ce
    return _balanced_categorical_crossentropy


def balanced_binary_crossentropy(class_weight=1, binary_weight=1):
    def _balanced_binary_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        bce = -tf.reduce_mean(
            (y_true*tf.math.log(y_pred)
            + binary_weight*(1 - y_true)
            *tf.math.log(1 - y_pred))*class_weight, axis=-1)
        return bce
    return _balanced_binary_crossentropy