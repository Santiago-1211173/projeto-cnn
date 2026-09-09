import tensorflow as tf

def categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor, global_batch_size: int) -> tf.Tensor:
    """
    Loss function designed for multi-GPU classification.
    Returns the scalar sum of loss divided by the global batch size instead of simple average.
    """
    y_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
    epsilon = 1e-15
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # 1. Calculate loss per example (1D vector)
    per_example_loss = -tf.reduce_sum(y_one_hot * tf.math.log(y_pred), axis=-1)
    
    # 2. Sum everything and divide by the Global Batch Size (Multi-GPU mathematical rigor)
    return tf.reduce_sum(per_example_loss) / tf.cast(global_batch_size, tf.float32)