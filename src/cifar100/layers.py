"""
Low-level layer primitives and residual building blocks for CIFAR-100 model.
Implemented using TensorFlow tf.Module with explicit parameter initialization.
"""

from typing import Optional
import tensorflow as tf


class BatchNorm2DLayer(tf.Module):
    """
    2D Batch Normalization layer maintaining running mean and variance.
    Supports both PyTorch-style and TensorFlow-style momentum arguments.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        epsilon: Optional[float] = None,
        momentum: float = 0.1,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        effective_eps = epsilon if epsilon is not None else eps
        self.eps = effective_eps
        self.epsilon = effective_eps
        # If momentum is e.g. 0.1 (PyTorch update rate), decay is 0.9.
        # If momentum is e.g. 0.9 (TF decay rate), decay is 0.9.
        self.decay = momentum if momentum > 0.5 else (1.0 - momentum)
        self.momentum = momentum

        prefix = f"{name}_" if name else ""
        self.gamma = tf.Variable(
            tf.ones([num_features], dtype=tf.float32),
            trainable=True,
            name=f"{prefix}gamma",
        )
        self.beta = tf.Variable(
            tf.zeros([num_features], dtype=tf.float32),
            trainable=True,
            name=f"{prefix}beta",
        )
        self.moving_mean = tf.Variable(
            tf.zeros([num_features], dtype=tf.float32),
            trainable=False,
            name=f"{prefix}mmean",
        )
        self.moving_var = tf.Variable(
            tf.ones([num_features], dtype=tf.float32),
            trainable=False,
            name=f"{prefix}mvar",
        )

    def __call__(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        if training:
            mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
            self.moving_mean.assign(
                self.decay * self.moving_mean + (1.0 - self.decay) * mean
            )
            self.moving_var.assign(
                self.decay * self.moving_var + (1.0 - self.decay) * variance
            )
            return tf.nn.batch_normalization(
                x, mean, variance, self.beta, self.gamma, self.eps
            )
        else:
            return tf.nn.batch_normalization(
                x, self.moving_mean, self.moving_var, self.beta, self.gamma, self.eps
            )


class Conv2DLayer(tf.Module):
    """
    2D Convolutional layer with He Normal weight initialization and zero bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "SAME",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        prefix = f"{name}_" if name else ""
        initializer = tf.initializers.HeNormal()
        shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.w = tf.Variable(
            initializer(shape=shape, dtype=tf.float32),
            trainable=True,
            name=f"{prefix}W",
        )
        self.b = tf.Variable(
            tf.zeros([out_channels], dtype=tf.float32),
            trainable=True,
            name=f"{prefix}b",
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        conv = tf.nn.conv2d(
            x, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding
        )
        return conv + self.b


class MaxPool2DLayer(tf.Module):
    """
    2D Max Pooling layer with configurable window and stride.
    """

    def __init__(
        self,
        pool_size: int = 2,
        stride: int = 2,
        padding: str = "SAME",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.max_pool2d(
            x, ksize=self.pool_size, strides=self.stride, padding=self.padding
        )


class DenseLayer(tf.Module):
    """
    Fully-connected linear layer with Glorot Uniform initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features

        prefix = f"{name}_" if name else ""
        initializer = tf.initializers.GlorotUniform()
        self.w = tf.Variable(
            initializer(shape=(in_features, out_features), dtype=tf.float32),
            trainable=True,
            name=f"{prefix}W",
        )
        self.b = tf.Variable(
            tf.zeros([out_features], dtype=tf.float32),
            trainable=True,
            name=f"{prefix}b",
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.w) + self.b


class ResidualBlock(tf.Module):
    """
    Standard Residual Block with He initialization and Batch Normalization.
    x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN + Shortcut(x) -> ReLU
    Includes 1x1 conv + BN projection when in_channels != out_channels or stride != 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        prefix = f"{name}_" if name else ""
        self.conv1 = Conv2DLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding="SAME",
            name=f"{prefix}c1",
        )
        self.bn1 = BatchNorm2DLayer(out_channels, name=f"{prefix}bn1")
        self.conv2 = Conv2DLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding="SAME",
            name=f"{prefix}c2",
        )
        self.bn2 = BatchNorm2DLayer(out_channels, name=f"{prefix}bn2")

        if in_channels != out_channels or stride != 1:
            self.shortcut_conv = Conv2DLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding="SAME",
                name=f"{prefix}sc",
            )
            self.shortcut_bn = BatchNorm2DLayer(out_channels, name=f"{prefix}sbn")
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
    """Global Average Pooling across spatial height and width axes [1, 2]."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(x, axis=[1, 2])
