import tensorflow as tf

def relu(x: tf.Tensor) -> tf.Tensor:
    """Returns max(0, x)."""
    return tf.maximum(0.0, x)

def softmax(x: tf.Tensor) -> tf.Tensor:
    """
    Computes Exp(x_i) / Sum(Exp(x_j)).
    Includes numerical stability handling by subtracting the maximum value.
    """
    exp_x = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)