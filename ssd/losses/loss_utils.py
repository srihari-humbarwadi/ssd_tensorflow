import tensorflow as tf

_policy = tf.keras.mixed_precision.experimental.global_policy()

def _safe_mean(losses):
    num_values = tf.cast(tf.size(losses), dtype=tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(losses), num_values)


def _scale_loss(loss):
    strategy = tf.distribute.get_strategy()
    num_replicas = strategy.num_replicas_in_sync
    return loss * (1. / num_replicas)


def get_scaled_losses(loss, regularization_losses=None):
    loss = _scale_loss(_safe_mean(loss))
    if regularization_losses:
        regularization_losses = tf.math.add_n(regularization_losses)
        regularization_losses = _scale_loss(regularization_losses)
        loss = loss + regularization_losses
    return loss


def reduce_losses(losses_dict):
    return {loss: _safe_mean(value) for loss, value in losses_dict.items()}
