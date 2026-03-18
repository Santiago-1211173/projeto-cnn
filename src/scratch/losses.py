import tensorflow as tf

def categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor, global_batch_size: int) -> tf.Tensor:
    """
    Loss para classificação multi-GPU. 
    Retorna a soma escalar dividida pelo batch global em vez da média simples.
    """
    y_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
    epsilon = 1e-15
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # 1. Calcular a loss por exemplo (vetor 1D)
    per_example_loss = -tf.reduce_sum(y_one_hot * tf.math.log(y_pred), axis=-1)
    
    # 2. Somar tudo e dividir pelo Batch Global (Rigor Matemático Multi-GPU)
    return tf.reduce_sum(per_example_loss) / tf.cast(global_batch_size, tf.float32)