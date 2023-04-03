# Copyright 2020 Samson Woof. All Rights Reserved.
# =============================================================================

"""Distribution-based loss functions.
"""

import tensorflow as tf

EPSILON = 1e-07


def balanced_categorical_crossentropy(class_weight=1):
    """
    Args:
        class_weight: A float or a list of floats,
            acts as reduction weighting coefficient
            for the per-class losses. If a scalar is provided,
            then the loss is simply scaled by the given value.
    Return:
        A function.
    """
    def _balanced_categorical_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, EPSILON, 1 - EPSILON)
        crossentropy = -tf.reduce_sum(
            (y_true*tf.math.log(y_pred)*class_weight), axis=-1)
        return crossentropy
    return _balanced_categorical_crossentropy


def balanced_binary_crossentropy(class_weight=1, binary_weight=1):
    """
    Args:
        class_weight: A float or a list of floats,
            acts as reduction weighting coefficient
            for the per-class losses. If a scalar is provided,
            then the loss is simply scaled by the given value.
        binary_weight: A float,
        acts as reduction weighting coefficient
        for the positive and negative losses.

    Return:
        A function.
    """
    def _balanced_binary_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, EPSILON, 1 - EPSILON)
        bce = -tf.reduce_mean(
            (y_true*tf.math.log(y_pred)
            + binary_weight*(1 - y_true)
            *tf.math.log(1 - y_pred))*class_weight, axis=-1)
        return bce
    return _balanced_binary_crossentropy


def channeled_categorical_crossentropy(y_true, y_pred):
    """Calculate crossentropy and sum each keypoint heatmap.

    Return:
        loss tensor with shape: (N, keypoints).
    """
    y_pred = tf.clip_by_value(y_pred, EPSILON, 1 - EPSILON)
    crossentropy = -tf.reduce_sum(
        y_true*tf.math.log(y_pred), axis=(1, 2))
    return crossentropy
