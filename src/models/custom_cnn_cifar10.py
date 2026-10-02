"""
CIFAR-10 CNN Feature Extractor.
A custom Convolutional Neural Network adapted for 32x32x3 RGB images,
built from scratch using the same low-level TensorFlow primitives as the MNIST model.

Maintains the identical 128D latent space contract required by the downstream
episodic memory, Mahalanobis++ OOD detector, and RL agent.
"""

import tensorflow as tf
from typing import Dict
from src.scratch.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
from src.scratch.activations import relu, softmax


class RawModelCIFAR10(tf.Module):
    def __init__(self, name: str = "custom_cnn_cifar10"):
        super().__init__(name=name)

        # --- CONVOLUTIONAL BLOCK 1 ---
        # Input: 32x32x3 RGB image. Two conv layers before pooling for richer features.
        self.conv1 = Conv2DLayer(in_channels=3, out_channels=32, kernel_size=3, name="conv1")
        # 32x32x3 -> Conv1(3x3,VALID) -> 30x30x32
        self.conv2 = Conv2DLayer(in_channels=32, out_channels=32, kernel_size=3, name="conv2")
        # 30x30x32 -> Conv2(3x3,VALID) -> 28x28x32
        self.pool1 = MaxPool2DLayer(pool_size=2, stride=2, name="pool1")
        # 28x28x32 -> Pool1(2x2) -> 14x14x32

        # --- CONVOLUTIONAL BLOCK 2 ---
        # Deeper features: edges -> textures -> object parts.
        self.conv3 = Conv2DLayer(in_channels=32, out_channels=64, kernel_size=3, name="conv3")
        # 14x14x32 -> Conv3(3x3,VALID) -> 12x12x64
        self.conv4 = Conv2DLayer(in_channels=64, out_channels=64, kernel_size=3, name="conv4")
        # 12x12x64 -> Conv4(3x3,VALID) -> 10x10x64
        self.pool2 = MaxPool2DLayer(pool_size=2, stride=2, name="pool2")
        # 10x10x64 -> Pool2(2x2) -> 5x5x64

        # --- TRANSITION ---
        self.flatten = tf.keras.layers.Flatten()

        # --- DENSE BLOCK (Latent Space and Classification) ---
        # Spatial dimensions after conv/pool chain:
        # 32 -> Conv1 -> 30 -> Conv2 -> 28 -> Pool1 -> 14
        # 14 -> Conv3 -> 12 -> Conv4 -> 10 -> Pool2 -> 5
        # 5 x 5 x 64 = 1600 (same flattened dim as the MNIST model)
        self.latent_dense = DenseLayer(in_features=5 * 5 * 64, out_features=128, name="latent_space")
        self.classifier_dense = DenseLayer(in_features=128, out_features=10, name="classifier")

    def __call__(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        # --- Phase 1: Spatial Feature Extraction ---
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = relu(x)
        x = self.conv4(x)
        x = relu(x)
        x = self.pool2(x)

        # --- Phase 2: Latent Space ---
        x_flat = self.flatten(x)
        raw_latent = self.latent_dense(x_flat)
        latent_features = relu(raw_latent)

        # --- Phase 3: Decision ---
        logits = self.classifier_dense(latent_features)
        probabilities = softmax(logits)

        return {
            "latent_features": latent_features,
            "probabilities": probabilities
        }
