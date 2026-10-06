import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config
import tensorflow as tf
from src.cifar100.layers import (
    Conv2DLayer,
    BatchNorm2DLayer,
    DenseLayer,
    ResidualBlock,
    GlobalAvgPool2DLayer,
)


class RawModelCIFAR100V2(tf.Module):
    """
    Enhanced ResNet-18-style backbone for CIFAR-100:
    - 4 Residual Stages: 64 -> 128 -> 256 -> 512
    - Strided convolutions for learned downsampling (No blind MaxPool)
    - GAP: (N, 4, 4, 512) -> (N, 512)
    - Invariant Latent Bottleneck: (N, 512) -> (N, latent_dim) with BatchNorm
    - Classifier: (N, latent_dim) -> (N, 100)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 100,
        name: str = "resnet18_cifar100_v2",
    ):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Prep Layer: (N, 32, 32, 3) -> (N, 32, 32, 64)
        self.prep_conv = Conv2DLayer(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding="SAME",
            name=f"{name}_prep_c",
        )
        self.prep_bn = BatchNorm2DLayer(num_features=64, name=f"{name}_prep_bn")

        # Stage 1: (N, 32, 32, 64) -> 2x ResBlock(64->64, stride 1) -> (N, 32, 32, 64)
        self.s1_res1 = ResidualBlock(64, 64, stride=1, name=f"{name}_s1_r1")
        self.s1_res2 = ResidualBlock(64, 64, stride=1, name=f"{name}_s1_r2")

        # Stage 2: (N, 32, 32, 64) -> ResBlock(64->128, stride 2) -> ResBlock(128->128, stride 1) -> (N, 16, 16, 128)
        self.s2_res1 = ResidualBlock(64, 128, stride=2, name=f"{name}_s2_r1")
        self.s2_res2 = ResidualBlock(128, 128, stride=1, name=f"{name}_s2_r2")

        # Stage 3: (N, 16, 16, 128) -> ResBlock(128->256, stride 2) -> ResBlock(256->256, stride 1) -> (N, 8, 8, 256)
        self.s3_res1 = ResidualBlock(128, 256, stride=2, name=f"{name}_s3_r1")
        self.s3_res2 = ResidualBlock(256, 256, stride=1, name=f"{name}_s3_r2")

        # Stage 4: (N, 8, 8, 256) -> ResBlock(256->512, stride 2) -> ResBlock(512->512, stride 1) -> (N, 4, 4, 512)
        self.s4_res1 = ResidualBlock(256, 512, stride=2, name=f"{name}_s4_r1")
        self.s4_res2 = ResidualBlock(512, 512, stride=1, name=f"{name}_s4_r2")

        # Global Average Pooling: (N, 4, 4, 512) -> (N, 512)
        self.gap = GlobalAvgPool2DLayer(name=f"{name}_gap")

        # Latent Bottleneck: (N, 512) -> (N, latent_dim)
        self.latent_dense = DenseLayer(
            in_features=512, out_features=latent_dim, name=f"{name}_lat_dense"
        )
        self.latent_bn = BatchNorm2DLayer(num_features=latent_dim, name=f"{name}_lat_bn")

        # Classifier Head: (N, latent_dim) -> (N, num_classes)
        self.classifier_dense = DenseLayer(
            in_features=latent_dim,
            out_features=num_classes,
            name=f"{name}_clf_dense",
        )

    def __call__(
        self, x: tf.Tensor, training: bool = True
    ) -> dict[str, tf.Tensor]:
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Prep stage
        x = tf.nn.relu(self.prep_bn(self.prep_conv(x), training=training))

        # Stage 1: 32x32x64
        x = self.s1_res1(x, training=training)
        x = self.s1_res2(x, training=training)

        # Stage 2: 16x16x128 (learned strided downsampling)
        x = self.s2_res1(x, training=training)
        x = self.s2_res2(x, training=training)

        # Stage 3: 8x8x256 (learned strided downsampling)
        x = self.s3_res1(x, training=training)
        x = self.s3_res2(x, training=training)

        # Stage 4: 4x4x512 (learned strided downsampling)
        x = self.s4_res1(x, training=training)
        x = self.s4_res2(x, training=training)

        # Global Average Pooling -> 512D
        pooled = self.gap(x)

        # Latent projection: Dense(512 -> 128) + BatchNorm
        latent_raw = self.latent_dense(pooled)
        # Apply 1D batch norm by expanding spatial dims
        latent_expanded = latent_raw[:, tf.newaxis, tf.newaxis, :]
        latent_bn = self.latent_bn(latent_expanded, training=training)[:, 0, 0, :]
        latent_features = tf.nn.relu(latent_bn)

        # Classifier logits and probabilities
        logits = self.classifier_dense(latent_features)
        probabilities = tf.nn.softmax(logits)

        return {
            "latent_features": latent_features,
            "logits": logits,
            "probabilities": probabilities,
        }


# Test model instantiate and forward pass
x_test = tf.random.normal([8, 32, 32, 3])
m = RawModelCIFAR100V2(latent_dim=128)
out = m(x_test, training=True)
print("Forward pass successful!")
print("  latent_features shape:", out["latent_features"].shape)
print("  logits shape:         ", out["logits"].shape)
print("  probabilities shape:  ", out["probabilities"].shape)
print("  Total trainable params:", sum(v.numpy().size for v in m.trainable_variables))
