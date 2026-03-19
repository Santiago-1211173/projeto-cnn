import tensorflow as tf

class DenseLayer(tf.Module):
    """Implementação raw de uma camada Fully Connected (MatMul)."""
    def __init__(self, in_features: int, out_features: int, name=None):
        super().__init__(name=name)
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
        return tf.matmul(x, self.w) + self.b


class Conv2DLayer(tf.Module):
    """Implementação matemática de uma camada Convolucional 2D."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = 'VALID', name=None):
        super().__init__(name=name)
        self.stride = stride
        self.padding = padding

        # Inicialização He Normal (Matematicamente provada ser superior para ativações ReLU)
        initializer = tf.initializers.HeNormal()
        
        # O Tensor de Pesos tem 4 dimensões: [altura_kernel, largura_kernel, canais_in, canais_out]
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.w = tf.Variable(
            initializer(shape=shape, dtype=tf.float32), 
            trainable=True, name=f'{name}_W'
        )
        
        # O Bias tem 1 dimensão: um valor a somar por cada canal de saída
        self.b = tf.Variable(
            tf.zeros([out_channels], dtype=tf.float32), 
            trainable=True, name=f'{name}_b'
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # A operação core do Deep Learning espacial:
        # strides=[batch, height, width, channels]
        conv = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        return conv + self.b


class MaxPool2DLayer(tf.Module):
    """Implementação puramente espacial de redução de dimensionalidade."""
    def __init__(self, pool_size: int = 2, stride: int = 2, padding: str = 'VALID', name=None):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Não tem pesos (não precisa de aprender nada), apenas extrai o máximo local.
        return tf.nn.max_pool2d(
            x, 
            ksize=[1, self.pool_size, self.pool_size, 1], 
            strides=[1, self.stride, self.stride, 1], 
            padding=self.padding
        )