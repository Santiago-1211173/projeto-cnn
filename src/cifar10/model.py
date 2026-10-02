"""
CIFAR-10 Residual CNN Feature Extractor with Batch Normalization.
Upgraded ResNet-style architecture designed to achieve 90%+ test accuracy.
Preserves the identical 128D latent contract required by episodic memory and RL agent.
"""
from typing import Dict
import tensorflow as tf
from src.cifar10.layers import Conv2DLayer, BatchNorm2DLayer, MaxPool2DLayer, DenseLayer, ResidualBlock, GlobalAvgPool2DLayer

class RawModelCIFAR10(tf.Module):
    """
    ResNet-9 architecture with invariant 128D latent bottleneck.
    Input: (N, 32, 32, 3)
    Output:
        latent_features: (N, 128)
        probabilities: (N, 10)
    """
    def __init__(self, name: str = "custom_cnn_cifar10"):
        super().__init__(name=name)

        # Prep Layer: 32x32x3 -> 32x32x64
        self.prep_conv = Conv2DLayer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding='SAME', name="prep_conv")
        self.prep_bn = BatchNorm2DLayer(num_features=64, name="prep_bn")

        # Stage 1: 32x32x64 -> ResBlock -> MaxPool -> 16x16x64
        self.res1 = ResidualBlock(in_channels=64, out_channels=64, stride=1, name="res1")
        self.pool1 = MaxPool2DLayer(pool_size=2, stride=2, padding='SAME', name="pool1")

        # Stage 2: 16x16x64 -> ResBlock -> MaxPool -> 8x8x128
        self.res2 = ResidualBlock(in_channels=64, out_channels=128, stride=1, name="res2")
        self.pool2 = MaxPool2DLayer(pool_size=2, stride=2, padding='SAME', name="pool2")

        # Stage 3: 8x8x128 -> ResBlock -> MaxPool -> 4x4x256
        self.res3 = ResidualBlock(in_channels=128, out_channels=256, stride=1, name="res3")
        self.pool3 = MaxPool2DLayer(pool_size=2, stride=2, padding='SAME', name="pool3")

        # Stage 4: 4x4x256 -> ResBlock -> 4x4x256
        self.res4 = ResidualBlock(in_channels=256, out_channels=256, stride=1, name="res4")

        # Global Average Pooling: 4x4x256 -> 256D
        self.gap = GlobalAvgPool2DLayer(name="gap")

        # Invariant Latent Bottleneck: 256 -> 128D
        self.latent_dense = DenseLayer(in_features=256, out_features=128, name="latent_space")

        # Classifier: 128 -> 10D
        self.classifier_dense = DenseLayer(in_features=128, out_features=10, name="classifier")

    def __call__(self, x: tf.Tensor, training: bool = True) -> Dict[str, tf.Tensor]:
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Prep
        x = tf.nn.relu(self.prep_bn(self.prep_conv(x), training=training))

        # Stages with Residuals
        x = self.pool1(self.res1(x, training=training))
        x = self.pool2(self.res2(x, training=training))
        x = self.pool3(self.res3(x, training=training))
        x = self.res4(x, training=training)

        # Global Pooling & 128D Latent Projection
        pooled = self.gap(x)
        raw_latent = self.latent_dense(pooled)
        latent_features = tf.nn.relu(raw_latent)

        # Classification
        logits = self.classifier_dense(latent_features)
        probabilities = tf.nn.softmax(logits)

        return {
            "latent_features": latent_features,
            "probabilities": probabilities
        }
