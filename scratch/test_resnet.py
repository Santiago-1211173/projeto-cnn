import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from src.cifar10.layers import Conv2DLayer, BatchNorm2DLayer, MaxPool2DLayer, DenseLayer

class ResidualBlock(tf.Module):
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

class TestResNet(tf.Module):
    def __init__(self):
        super().__init__(name="test_resnet")
        self.prep = Conv2DLayer(3, 64, kernel_size=3, padding='SAME', name="prep_conv")
        self.bn_prep = BatchNorm2DLayer(64, name="prep_bn")
        
        self.res1 = ResidualBlock(64, 64, name="res1")
        self.pool1 = MaxPool2DLayer(2, 2, padding='SAME')

        self.res2 = ResidualBlock(64, 128, name="res2")
        self.pool2 = MaxPool2DLayer(2, 2, padding='SAME')

        self.res3 = ResidualBlock(128, 256, name="res3")
        self.pool3 = MaxPool2DLayer(2, 2, padding='SAME')

        self.res4 = ResidualBlock(256, 256, name="res4")

        self.latent_dense = DenseLayer(256, 128, name="latent_space")
        self.classifier = DenseLayer(128, 10, name="classifier")

    def __call__(self, x: tf.Tensor, training: bool = True):
        x = tf.nn.relu(self.bn_prep(self.prep(x), training=training))
        x = self.pool1(self.res1(x, training=training))
        x = self.pool2(self.res2(x, training=training))
        x = self.pool3(self.res3(x, training=training))
        x = self.res4(x, training=training)

        gap = tf.reduce_mean(x, axis=[1, 2])
        latent = tf.nn.relu(self.latent_dense(gap))
        logits = self.classifier(latent)
        probs = tf.nn.softmax(logits)

        return {"latent_features": latent, "probabilities": probs}

m = TestResNet()
dummy = tf.random.normal([8, 32, 32, 3])
out = m(dummy, training=False)
print("Latent shape:", out["latent_features"].shape)
print("Probs shape:", out["probabilities"].shape)
assert out["latent_features"].shape == (8, 128)
assert out["probabilities"].shape == (8, 10)
print("ResNet test passed!")
