"""
Módulo contendo a arquitetura da rede.
Agora é uma VERDADEIRA Rede Neuronal Convolucional (CNN), construída do zero.
"""

import tensorflow as tf
from typing import Dict
from src.scratch.layers import DenseLayer, Conv2DLayer, MaxPool2DLayer
from src.scratch.activations import relu, softmax

class RawModel(tf.Module):
    def __init__(self, name: str = "true_custom_cnn"):
        super().__init__(name=name)
        
        # --- BLOCO CONVOLUCIONAL 1 ---
        # A imagem entra com 1 canal (Grayscale). Extraímos 32 características (bordas/linhas).
        self.conv1 = Conv2DLayer(in_channels=1, out_channels=32, kernel_size=3, name="conv1")
        self.pool1 = MaxPool2DLayer(pool_size=2, stride=2, name="pool1")
        
        # --- BLOCO CONVOLUCIONAL 2 ---
        # Transformamos as 32 linhas básicas em 64 formas geométricas complexas.
        self.conv2 = Conv2DLayer(in_channels=32, out_channels=64, kernel_size=3, name="conv2")
        self.pool2 = MaxPool2DLayer(pool_size=2, stride=2, name="pool2")
        
        # --- TRANSIÇÃO ---
        self.flatten = tf.keras.layers.Flatten()
        
        # --- BLOCO DENSO (Espaço Latente e Classificação) ---
        # Como calculamos o input de 1600?
        # Imagem 28x28 -> Conv1(3x3) -> 26x26 -> Pool1(2x2) -> 13x13
        # 13x13 -> Conv2(3x3) -> 11x11 -> Pool2(2x2) -> 5x5. 
        # 5x5 pixeis * 64 filtros = 1600 dimensões latentes raw.
        self.latent_dense = DenseLayer(in_features=5 * 5 * 64, out_features=128, name="latent_space")
        self.classifier_dense = DenseLayer(in_features=128, out_features=10, name="classifier")

    def __call__(self, x: tf.Tensor) -> Dict[str, tf.Tensor]:
        # --- Fase 1: Extração Espacial de Características ---
        x = self.conv1(x)
        x = relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = relu(x)
        x = self.pool2(x)
        
        # --- Fase 2: O Espaço Latente (A compressão do cérebro) ---
        x_flat = self.flatten(x)
        raw_latent = self.latent_dense(x_flat)
        latent_features = relu(raw_latent)
        
        # --- Fase 3: Decisão (A Caixa Preta) ---
        logits = self.classifier_dense(latent_features)
        probabilities = softmax(logits)
        
        return {
            "latent_features": latent_features,
            "probabilities": probabilities
        }