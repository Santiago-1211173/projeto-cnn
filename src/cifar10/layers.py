"""Low-level layer primitives and residual building blocks for CIFAR-10 model."""
import tensorflow as tf

class BatchNorm2DLayer(tf.Module):
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.9, name: str = None):
        super().__init__(name=name)
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = tf.Variable(tf.ones([num_features], dtype=tf.float32), trainable=True, name=f"{name}_gamma")
        self.beta = tf.Variable(tf.zeros([num_features], dtype=tf.float32), trainable=True, name=f"{name}_beta")
        self.moving_mean = tf.Variable(tf.zeros([num_features], dtype=tf.float32), trainable=False, name=f"{name}_mmean")
        self.moving_var = tf.Variable(tf.ones([num_features], dtype=tf.float32), trainable=False, name=f"{name}_mvar")

    def __call__(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        if training:
            mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
            self.moving_mean.assign(self.momentum * self.moving_mean + (1.0 - self.momentum) * mean)
            self.moving_var.assign(self.momentum * self.moving_var + (1.0 - self.momentum) * variance)
            return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(x, self.moving_mean, self.moving_var, self.beta, self.gamma, self.epsilon)


class Conv2DLayer(tf.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = 'SAME', name=None):
        super().__init__(name=name)
        self.stride = stride
        self.padding = padding
        initializer = tf.initializers.HeNormal()
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.w = tf.Variable(initializer(shape=shape, dtype=tf.float32), trainable=True, name=f'{name}_W')
        self.b = tf.Variable(tf.zeros([out_channels], dtype=tf.float32), trainable=True, name=f'{name}_b')

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        conv = tf.nn.conv2d(x, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        return conv + self.b


class MaxPool2DLayer(tf.Module):
    def __init__(self, pool_size: int = 2, stride: int = 2, padding: str = 'SAME', name=None):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.max_pool2d(x, ksize=self.pool_size, strides=self.stride, padding=self.padding)


class DenseLayer(tf.Module):
    def __init__(self, in_features: int, out_features: int, name=None):
        super().__init__(name=name)
        initializer = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initializer(shape=(in_features, out_features), dtype=tf.float32), trainable=True, name=f'{name}_W')
        self.b = tf.Variable(tf.zeros([out_features], dtype=tf.float32), trainable=True, name=f'{name}_b')

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.w) + self.b


class ResidualBlock(tf.Module):
    """
    Standard Residual Block with He initialization and Batch Normalization.
    x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN + Shortcut(x) -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, name: str = None):
        super().__init__(name=name)
        self.conv1 = Conv2DLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding='SAME', name=f"{name}_c1")
        self.bn1 = BatchNorm2DLayer(out_channels, name=f"{name}_bn1")
        self.conv2 = Conv2DLayer(out_channels, out_channels, kernel_size=3, stride=1, padding='SAME', name=f"{name}_c2")
        self.bn2 = BatchNorm2DLayer(out_channels, name=f"{name}_bn2")

        if in_channels != out_channels or stride != 1:
            self.shortcut_conv = Conv2DLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding='SAME', name=f"{name}_sc")
            self.shortcut_bn = BatchNorm2DLayer(out_channels, name=f"{name}_sbn")
            self.has_shortcut = True
        else:
            self.has_shortcut = False

    def __call__(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        shortcut = x
        if self.has_shortcut:
            shortcut = self.shortcut_bn(self.shortcut_conv(x), training=training)

        residual = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        residual = self.bn2(self.conv2(residual), training=training)
        return tf.nn.relu(residual + shortcut)


class GlobalAvgPool2DLayer(tf.Module):
    """Global Average Pooling across spatial height and width."""
    def __init__(self, name: str = None):
        super().__init__(name=name)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(x, axis=[1, 2])
