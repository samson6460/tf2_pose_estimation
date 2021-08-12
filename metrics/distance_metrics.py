import tensorflow as tf


def distance_from_prob(y_true, y_pred):
    height, width = y_pred.get_shape()[1:3]

    index_map_x = tf.range(width, dtype=y_true.dtype)
    index_map_x = tf.reshape(index_map_x, (1, 1, -1, 1))
    index_map_y = tf.range(height, dtype=y_true.dtype)
    index_map_y = tf.reshape(index_map_y, (1, -1, 1, 1))

    y_true_map_x = y_true*index_map_x
    y_true_map_y = y_true*index_map_y
    y_pred_map_x = y_pred*index_map_x
    y_pred_map_y = y_pred*index_map_y

    y_true_x = tf.reduce_sum(y_true_map_x, axis=(1, 2))
    y_true_y = tf.reduce_sum(y_true_map_y, axis=(1, 2))
    y_pred_x = tf.reduce_sum(y_pred_map_x, axis=(1, 2))
    y_pred_y = tf.reduce_sum(y_pred_map_y, axis=(1, 2))

    dist = tf.math.sqrt(
        (y_true_x - y_pred_x)**2 + (y_true_y - y_pred_y)**2)

    dist = tf.reduce_mean(dist, axis=-1)

    return dist

