import tensorflow as tf

class DenseLayer(tf.Module):
    """Raw implementation of a Fully Connected (Dense) layer."""
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
    """Mathematical implementation of a 2D Convolutional layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = 'VALID', name=None):
        super().__init__(name=name)
        self.stride = stride
        self.padding = padding

        # He Normal initialization (proven mathematically superior for ReLU activations)
        initializer = tf.initializers.HeNormal()
        
        # The weight tensor has 4 dimensions: [kernel_height, kernel_width, in_channels, out_channels]
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.w = tf.Variable(
            initializer(shape=shape, dtype=tf.float32), 
            trainable=True, name=f'{name}_W'
        )
        
        # The bias tensor has 1 dimension: one bias value added per output channel
        self.b = tf.Variable(
            tf.zeros([out_channels], dtype=tf.float32), 
            trainable=True, name=f'{name}_b'
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # Spatial convolution operation:
        # strides=[batch, height, width, channels]
        conv = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        return conv + self.b


class MaxPool2DLayer(tf.Module):
    """Spatial implementation of 2D max pooling dimensionality reduction."""
    def __init__(self, pool_size: int = 2, stride: int = 2, padding: str = 'VALID', name=None):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        # No weights to learn, simply extracts the local maximum value.
        return tf.nn.max_pool2d(
            x, 
            ksize=[1, self.pool_size, self.pool_size, 1], 
            strides=[1, self.stride, self.stride, 1], 
            padding=self.padding
        )