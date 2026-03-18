import tensorflow as tf

def relu(x: tf.Tensor) -> tf.Tensor:
    """Retorna max(0, x)."""
    return tf.maximum(0.0, x)

def softmax(x: tf.Tensor) -> tf.Tensor:
    """
    Exp(x_i) / Sum(Exp(x_j)). 
    Inclui estabilidade numérica subtraindo o máximo.
    """
    exp_x = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=-1, keepdims=True)