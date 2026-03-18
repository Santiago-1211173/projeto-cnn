import tensorflow as tf

class DenseLayer(tf.Module):
    """Implementação raw de uma camada Fully Connected."""
    def __init__(self, in_features: int, out_features: int, name=None):
        super().__init__(name=name)
        # Inicialização He/Xavier manual
        initializer = tf.initializers.GlorotUniform()
        self.w = tf.Variable(
            initializer(shape=(in_features, out_features), dtype=tf.float32), 
            trainable=True, name=f'{name}_W'
        )
        self.b = tf.Variable(
            tf.zeros([out_features], dtype=tf.float32), 
            trainable=True, name=f'{name}_b'
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Matemática Pura: MatMul na L40S
        return tf.matmul(x, self.w) + self.b