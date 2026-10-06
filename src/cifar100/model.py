"""
CIFAR-100 ResNet-14 Architecture with Dynamic Latent Bottleneck.
Designed for 100-class classification, dual-uncertainty OOD arbitration,
and episodic memory integration.
"""

from typing import Dict, Optional, Tuple
import tensorflow as tf
from src.cifar100.layers import (
    Conv2DLayer,
    BatchNorm2DLayer,
    MaxPool2DLayer,
    DenseLayer,
    ResidualBlock,
    GlobalAvgPool2DLayer,
)


class RawModelCIFAR100(tf.Module):
    """
    ResNet-14 architecture with 6 Residual Blocks (2 per stage)
    and configurable latent bottleneck dimension.

    Input: (N, 32, 32, 3)
    Output:
        latent_features: (N, latent_dim)
        logits:          (N, 100)
        probabilities:   (N, 100)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 100,
        name: str = "resnet14_cifar100",
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
            name=f"{name}_prep_conv",
        )
        self.prep_bn = BatchNorm2DLayer(num_features=64, name=f"{name}_prep_bn")

        # Stage 1: (N, 32, 32, 64) -> 2x ResBlock -> MaxPool -> (N, 16, 16, 64)
        self.stage1_res1 = ResidualBlock(
            in_channels=64, out_channels=64, stride=1, name=f"{name}_s1_res1"
        )
        self.stage1_res2 = ResidualBlock(
            in_channels=64, out_channels=64, stride=1, name=f"{name}_s1_res2"
        )
        self.stage1_pool = MaxPool2DLayer(
            pool_size=2, stride=2, padding="SAME", name=f"{name}_s1_pool"
        )

        # Stage 2: (N, 16, 16, 64) -> ResBlock(64->128) -> ResBlock(128->128) -> MaxPool -> (N, 8, 8, 128)
        self.stage2_res1 = ResidualBlock(
            in_channels=64, out_channels=128, stride=1, name=f"{name}_s2_res1"
        )
        self.stage2_res2 = ResidualBlock(
            in_channels=128, out_channels=128, stride=1, name=f"{name}_s2_res2"
        )
        self.stage2_pool = MaxPool2DLayer(
            pool_size=2, stride=2, padding="SAME", name=f"{name}_s2_pool"
        )

        # Stage 3: (N, 8, 8, 128) -> ResBlock(128->256) -> ResBlock(256->256) -> MaxPool -> (N, 4, 4, 256)
        self.stage3_res1 = ResidualBlock(
            in_channels=128, out_channels=256, stride=1, name=f"{name}_s3_res1"
        )
        self.stage3_res2 = ResidualBlock(
            in_channels=256, out_channels=256, stride=1, name=f"{name}_s3_res2"
        )
        self.stage3_pool = MaxPool2DLayer(
            pool_size=2, stride=2, padding="SAME", name=f"{name}_s3_pool"
        )

        # Global Average Pooling: (N, 4, 4, 256) -> (N, 256)
        self.gap = GlobalAvgPool2DLayer(name=f"{name}_gap")

        # Latent Bottleneck: (N, 256) -> (N, latent_dim)
        self.latent_dense = DenseLayer(
            in_features=256, out_features=latent_dim, name=f"{name}_latent_dense"
        )

        # Classifier Head: (N, latent_dim) -> (N, num_classes)
        self.classifier_dense = DenseLayer(
            in_features=latent_dim,
            out_features=num_classes,
            name=f"{name}_classifier_dense",
        )

    def __call__(
        self, x: tf.Tensor, training: bool = True
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass of ResNet-14.

        Args:
            x: Input tensor of shape (N, 32, 32, 3).
            training: Boolean indicating whether batch normalization should update running stats.

        Returns:
            Dictionary containing:
                - 'latent_features': Tensor of shape (N, latent_dim)
                - 'logits': Tensor of shape (N, num_classes)
                - 'probabilities': Tensor of shape (N, num_classes)
        """
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Prep stage
        x = tf.nn.relu(self.prep_bn(self.prep_conv(x), training=training))

        # Stage 1: 32x32x64 -> 16x16x64
        x = self.stage1_res1(x, training=training)
        x = self.stage1_res2(x, training=training)
        x = self.stage1_pool(x)

        # Stage 2: 16x16x64 -> 8x8x128
        x = self.stage2_res1(x, training=training)
        x = self.stage2_res2(x, training=training)
        x = self.stage2_pool(x)

        # Stage 3: 8x8x128 -> 4x4x256
        x = self.stage3_res1(x, training=training)
        x = self.stage3_res2(x, training=training)
        x = self.stage3_pool(x)

        # Global Average Pooling: -> 256D
        pooled = self.gap(x)

        # Latent space projection with ReLU activation
        latent_features = tf.nn.relu(self.latent_dense(pooled))

        # Classification logits and probabilities
        logits = self.classifier_dense(latent_features)
        probabilities = tf.nn.softmax(logits)

        return {
            "latent_features": latent_features,
            "logits": logits,
            "probabilities": probabilities,
        }


class RawModelCIFAR100V2(tf.Module):
    """
    Enhanced ResNet-18 architecture with 4 Residual Stages (8 Residual Blocks)
    and learned strided convolutions (replacing blind MaxPools).

    Input: (N, 32, 32, 3)
    Stages:
        Prep:    (N, 32, 32, 3)  -> Conv(3->64) + BN + ReLU  -> (N, 32, 32, 64)
        Stage 1: (N, 32, 32, 64) -> 2x ResBlock(64->64, s=1)  -> (N, 32, 32, 64)
        Stage 2: (N, 32, 32, 64) -> ResBlock(64->128, s=2) + ResBlock(128->128, s=1) -> (N, 16, 16, 128)
        Stage 3: (N, 16, 16, 128)-> ResBlock(128->256, s=2) + ResBlock(256->256, s=1) -> (N, 8, 8, 256)
        Stage 4: (N, 8, 8, 256)  -> ResBlock(256->512, s=2) + ResBlock(512->512, s=1) -> (N, 4, 4, 512)
        GAP:     (N, 4, 4, 512)  -> GlobalAvgPool             -> (N, 512)
        Bottleneck: (N, 512)     -> Dense(512->128) + BN + ReLU -> (N, latent_dim)
        Classifier: (N, 128)     -> Dense(128->100)           -> (N, 100)
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
            name=f"{name}_prep_conv",
        )
        self.prep_bn = BatchNorm2DLayer(num_features=64, name=f"{name}_prep_bn")

        # Stage 1: (N, 32, 32, 64) -> (N, 32, 32, 64)
        self.stage1_res1 = ResidualBlock(
            in_channels=64, out_channels=64, stride=1, name=f"{name}_s1_res1"
        )
        self.stage1_res2 = ResidualBlock(
            in_channels=64, out_channels=64, stride=1, name=f"{name}_s1_res2"
        )

        # Stage 2: (N, 32, 32, 64) -> (N, 16, 16, 128) via learned strided convolution
        self.stage2_res1 = ResidualBlock(
            in_channels=64, out_channels=128, stride=2, name=f"{name}_s2_res1"
        )
        self.stage2_res2 = ResidualBlock(
            in_channels=128, out_channels=128, stride=1, name=f"{name}_s2_res2"
        )

        # Stage 3: (N, 16, 16, 128) -> (N, 8, 8, 256) via learned strided convolution
        self.stage3_res1 = ResidualBlock(
            in_channels=128, out_channels=256, stride=2, name=f"{name}_s3_res1"
        )
        self.stage3_res2 = ResidualBlock(
            in_channels=256, out_channels=256, stride=1, name=f"{name}_s3_res2"
        )

        # Stage 4: (N, 8, 8, 256) -> (N, 4, 4, 512) via learned strided convolution
        self.stage4_res1 = ResidualBlock(
            in_channels=256, out_channels=512, stride=2, name=f"{name}_s4_res1"
        )
        self.stage4_res2 = ResidualBlock(
            in_channels=512, out_channels=512, stride=1, name=f"{name}_s4_res2"
        )

        # Global Average Pooling: (N, 4, 4, 512) -> (N, 512)
        self.gap = GlobalAvgPool2DLayer(name=f"{name}_gap")

        # Latent Bottleneck: (N, 512) -> (N, latent_dim) with BatchNorm
        self.latent_dense = DenseLayer(
            in_features=512, out_features=latent_dim, name=f"{name}_latent_dense"
        )
        self.latent_bn = BatchNorm2DLayer(
            num_features=latent_dim, name=f"{name}_latent_bn"
        )

        # Classifier Head: (N, latent_dim) -> (N, num_classes)
        self.classifier_dense = DenseLayer(
            in_features=latent_dim,
            out_features=num_classes,
            name=f"{name}_classifier_dense",
        )

    def __call__(
        self, x: tf.Tensor, training: bool = True
    ) -> Dict[str, tf.Tensor]:
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Prep stage
        x = tf.nn.relu(self.prep_bn(self.prep_conv(x), training=training))

        # Stage 1: 32x32x64
        x = self.stage1_res1(x, training=training)
        x = self.stage1_res2(x, training=training)

        # Stage 2: 16x16x128
        x = self.stage2_res1(x, training=training)
        x = self.stage2_res2(x, training=training)

        # Stage 3: 8x8x256
        x = self.stage3_res1(x, training=training)
        x = self.stage3_res2(x, training=training)

        # Stage 4: 4x4x512
        x = self.stage4_res1(x, training=training)
        x = self.stage4_res2(x, training=training)

        # Global Average Pooling -> 512D
        pooled = self.gap(x)

        # Latent space projection with BatchNorm and ReLU
        latent_raw = self.latent_dense(pooled)
        latent_exp = latent_raw[:, tf.newaxis, tf.newaxis, :]
        latent_bn = self.latent_bn(latent_exp, training=training)[:, 0, 0, :]
        latent_features = tf.nn.relu(latent_bn)

        # Classification logits and probabilities
        logits = self.classifier_dense(latent_features)
        probabilities = tf.nn.softmax(logits)

        return {
            "latent_features": latent_features,
            "logits": logits,
            "probabilities": probabilities,
        }


CIFAR100_MEAN = tf.constant([0.50707516, 0.48654887, 0.44091784], dtype=tf.float32)
CIFAR100_STD = tf.constant([0.26733429, 0.25643846, 0.27615047], dtype=tf.float32)


def load_cifar100_backbone(
    checkpoint_dir: str,
    latent_dim: int = 128,
) -> Tuple[tf.Module, bool]:
    """
    Loads and restores the CIFAR-100 CNN backbone from checkpoint_dir.
    Auto-detects architecture (RawModelCIFAR100 vs RawModelCIFAR100V2)
    and standardization requirements from model_meta.json.

    Returns:
        Tuple of (model, standardize_flag)
    """
    import os
    import json

    meta_path = os.path.join(checkpoint_dir, "model_meta.json")
    model_version = "v1"
    standardize = False
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_version = meta.get("model_version", "v1")
                standardize = bool(meta.get("standardize", False))
        except Exception:
            pass

    if model_version == "v2":
        model = RawModelCIFAR100V2(latent_dim=latent_dim)
    else:
        model = RawModelCIFAR100(latent_dim=latent_dim)

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest_ckpt:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}!")

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(latest_ckpt).expect_partial()
    return model, standardize

