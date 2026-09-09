"""
Module containing the network architecture.
A custom Convolutional Neural Network (CNN) built from scratch.
"""

import tensorflow as tf
from typing import Dict
from src.scratch.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
from src.scratch.activations import relu, softmax

class RawModel(tf.Module):
    def __init__(self, name: str = "true_custom_cnn"):
        super().__init__(name=name)
        
        # --- CONVOLUTIONAL BLOCK 1 ---
        # Input image has 1 channel (Grayscale). Extract 32 features (edges/lines).
        self.conv1 = Conv2DLayer(in_channels=1, out_channels=32, kernel_size=3, name="conv1")
        self.pool1 = MaxPool2DLayer(pool_size=2, stride=2, name="pool1")
        
        # --- CONVOLUTIONAL BLOCK 2 ---
        # Transform the 32 basic features into 64 complex geometric shapes.
        self.conv2 = Conv2DLayer(in_channels=32, out_channels=64, kernel_size=3, name="conv2")
        self.pool2 = MaxPool2DLayer(pool_size=2, stride=2, name="pool2")
        
        # --- TRANSITION ---
        self.flatten = tf.keras.layers.Flatten()
        
        # --- DENSE BLOCK (Latent Space and Classification) ---
        # Calculating the 1600 input dimensions:
        # Image 28x28 -> Conv1(3x3) -> 26x26 -> Pool1(2x2) -> 13x13
        # 13x13 -> Conv2(3x3) -> 11x11 -> Pool2(2x2) -> 5x5. 
        # 5x5 pixels * 64 filters = 1600 raw latent dimensions.
        self.latent_dense = DenseLayer(in_features=5 * 5 * 64, out_features=128, name="latent_space")
        self.classifier_dense = DenseLayer(in_features=128, out_features=10, name="classifier")

    def __call__(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        # --- Phase 1: Spatial Feature Extraction ---
        x = self.conv1(x)
        x = relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
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